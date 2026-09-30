"""GPU-only RunPod job: S3 video -> MediaPipe artifacts and annotated video."""

from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

import requests

from contract import ARTIFACTS, MAX_INPUT_BYTES
from .analysis import run_analysis, sha256


MODEL_PATH = Path(os.environ.get(
    "POSE_MODEL_PATH", "/opt/models/pose_landmarker_heavy.task"
))
_RUN_ID = re.compile(r"^[A-Za-z0-9_-]{8,80}$")


def _s3_url(value: object) -> str:
    if not isinstance(value, str):
        raise ValueError("expected an HTTPS AWS S3 presigned URL")
    parsed = urlsplit(value)
    host = (parsed.hostname or "").lower()
    s3_host = host == "s3.amazonaws.com" or (
        host.endswith(".amazonaws.com")
        and (host.startswith("s3.") or ".s3." in host)
    )
    if parsed.scheme != "https" or not s3_host or not parsed.path or not parsed.query:
        raise ValueError("expected an HTTPS AWS S3 presigned URL")
    return value


def parse_payload(job: object) -> dict:
    if not isinstance(job, dict) or not isinstance(job.get("input"), dict):
        raise ValueError("RunPod job must contain input")
    data = job["input"]
    run_id = data.get("run_id")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("invalid run_id")
    source = data.get("source")
    if not isinstance(source, dict) or not all(
        isinstance(source.get(key), str) and source[key]
        for key in ("bucket", "key", "version_id", "etag")
    ):
        raise ValueError("versioned S3 source identity is required")
    size = source.get("size_bytes")
    if type(size) is not int or not 0 < size <= MAX_INPUT_BYTES:
        raise ValueError("invalid source size")
    frame_rate = data.get("processing_frame_rate", 30)
    if type(frame_rate) not in (int, float) or not 1 <= frame_rate <= 60:
        raise ValueError("processing_frame_rate must be 1 through 60")
    put_urls = data.get("output_put_urls")
    output_uris = data.get("output_s3_uris")
    if not isinstance(put_urls, dict) or not isinstance(output_uris, dict):
        raise ValueError("artifact destinations are required")
    outputs = {}
    for name, (filename, content_type) in ARTIFACTS.items():
        uri = output_uris.get(name)
        if not isinstance(uri, str) or not uri.endswith("/" + filename):
            raise ValueError(f"invalid output URI: {name}")
        outputs[name] = (_s3_url(put_urls.get(name)), uri, content_type)
    return {
        "run_id": run_id,
        "video_get_url": _s3_url(data.get("video_get_url")),
        "source": source,
        "processing_frame_rate": float(frame_rate),
        "outputs": outputs,
    }


def _gpu_info() -> dict[str, str]:
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10, check=True,
        )
    except (OSError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        raise RuntimeError("NVIDIA GPU unavailable; refusing CPU fallback") from None
    line = result.stdout.strip().splitlines()
    if not line or not line[0].strip():
        raise RuntimeError("NVIDIA GPU unavailable; refusing CPU fallback")
    name, _, driver = line[0].partition(",")
    return {"name": name.strip(), "driver_version": driver.strip()}


def _download(url: str, destination: Path, expected_size: int) -> None:
    total = 0
    try:
        with requests.get(url, stream=True, timeout=(15, 60), allow_redirects=False) as response:
            if response.status_code != 200:
                raise RuntimeError(f"S3 video download failed: HTTP {response.status_code}")
            with destination.open("wb") as target:
                for block in response.iter_content(chunk_size=1024 * 1024):
                    total += len(block)
                    if total > MAX_INPUT_BYTES:
                        raise ValueError("source video exceeds test limit")
                    target.write(block)
    except requests.RequestException:
        # Request exceptions may embed the signed URL; never include it in logs.
        raise RuntimeError("S3 video download failed") from None
    if total != expected_size:
        raise ValueError("downloaded video size differs from versioned S3 source")


def _upload(url: str, path: Path, content_type: str) -> None:
    if not path.is_file() or path.stat().st_size <= 0:
        raise ValueError(f"missing or empty output artifact: {path.name}")
    try:
        with path.open("rb") as artifact:
            response = requests.put(
                url, data=artifact, headers={"Content-Type": content_type},
                timeout=(15, 300), allow_redirects=False,
            )
    except requests.RequestException:
        raise RuntimeError(f"S3 upload failed: {path.name}") from None
    if response.status_code not in (200, 201, 204):
        raise RuntimeError(
            f"S3 upload failed: {path.name}, HTTP {response.status_code}"
        )


def handler(job: dict) -> dict:
    started = time.perf_counter()
    payload = parse_payload(job)
    gpu = _gpu_info()
    suffix = Path(urlsplit(payload["video_get_url"]).path).suffix.lower()
    if suffix not in {".mov", ".mp4"}:
        raise ValueError("input source must be .mov or .mp4")
    with tempfile.TemporaryDirectory(prefix="arcs-mediapipe-gpu-") as temp:
        root = Path(temp)
        video = root / f"source{suffix}"
        stage = time.perf_counter()
        _download(
            payload["video_get_url"], video, payload["source"]["size_bytes"]
        )
        download_seconds = time.perf_counter() - stage
        stage = time.perf_counter()
        analysis = run_analysis(
            video, root / "output", model_path=MODEL_PATH,
            source=payload["source"],
            processing_frame_rate=payload["processing_frame_rate"],
        )
        processing_seconds = time.perf_counter() - stage
        output_dir = root / "output"
        hashes = {
            name: sha256(output_dir / filename)
            for name, (filename, _) in ARTIFACTS.items()
            if name != "performance"
        }
        performance = {
            "schema_version": 1,
            "run_id": payload["run_id"],
            "source": payload["source"],
            "source_sha256": sha256(video),
            "gpu": gpu,
            "gpu_delegate_requested": True,
            "gpu_delegate_initialized": True,
            "artifacts_sha256": hashes,
            "analysis": analysis,
            "download_seconds": round(download_seconds, 6),
            "processing_seconds": round(processing_seconds, 6),
        }
        (output_dir / "performance.json").write_text(
            json.dumps(performance, indent=2, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        uploaded = {}
        # The performance report is uploaded last as a completion marker.
        for name, (filename, _) in ARTIFACTS.items():
            url, uri, content_type = payload["outputs"][name]
            _upload(url, output_dir / filename, content_type)
            uploaded[name] = uri
        return {
            "run_id": payload["run_id"],
            "status": "complete",
            "frames": analysis["analysis"]["frames"],
            "gpu": gpu,
            "model_api": analysis["model_api"],
            "artifacts": uploaded,
            "worker_wall_seconds": round(time.perf_counter() - started, 6),
        }


if __name__ == "__main__":
    import runpod

    runpod.serverless.start({"handler": handler})
