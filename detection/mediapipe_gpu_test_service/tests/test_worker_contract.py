from __future__ import annotations

import json

import pytest

from contract import ARTIFACTS
from runpod_worker import handler as worker
from runpod_worker.handler import _gpu_info, _s3_url, parse_payload


def _payload() -> dict:
    return {
        "input": {
            "run_id": "mpgpu-20260930T120000Z-abcdef12",
            "video_get_url": "https://bucket.s3.ap-southeast-1.amazonaws.com/input.mov?sig=x",
            "source": {
                "bucket": "bucket", "key": "inputs/input.mov", "version_id": "v1",
                "etag": "etag", "size_bytes": 1024,
            },
            "processing_frame_rate": 30,
            "output_put_urls": {
                name: f"https://bucket.s3.ap-southeast-1.amazonaws.com/{filename}?sig=x"
                for name, (filename, _) in ARTIFACTS.items()
            },
            "output_s3_uris": {
                name: f"s3://bucket/tests/run/{filename}"
                for name, (filename, _) in ARTIFACTS.items()
            },
        }
    }


def test_worker_payload_requires_every_artifact() -> None:
    payload = _payload()
    assert len(parse_payload(payload)["outputs"]) == len(ARTIFACTS)
    del payload["input"]["output_put_urls"]["turns"]
    with pytest.raises(ValueError, match="presigned URL"):
        parse_payload(payload)


def test_worker_rejects_arbitrary_download_hosts() -> None:
    with pytest.raises(ValueError, match="AWS S3"):
        _s3_url("https://attacker.example/video.mov?sig=x")


def test_worker_refuses_missing_nvidia_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    def absent(*args, **kwargs):
        raise FileNotFoundError("nvidia-smi")

    monkeypatch.setattr("runpod_worker.handler.subprocess.run", absent)
    with pytest.raises(RuntimeError, match="refusing CPU fallback"):
        _gpu_info()


def test_worker_uploads_completion_marker_last(monkeypatch: pytest.MonkeyPatch) -> None:
    uploads = []

    def download(_url, destination, _expected_size):
        destination.write_bytes(b"test video")

    def analyze(_video, output_dir, **_kwargs):
        output_dir.mkdir()
        for name, (filename, _) in ARTIFACTS.items():
            if name != "performance":
                (output_dir / filename).write_bytes(b"artifact")
        return {"analysis": {"frames": 1}, "model_api": "PoseLandmarker"}

    def upload(_url, path, _content_type):
        uploads.append(path.name)
        if path.name == "performance.json":
            report = json.loads(path.read_text())
            assert report["gpu_delegate_initialized"] is True
            assert len(report["artifacts_sha256"]) == len(ARTIFACTS) - 1

    monkeypatch.setattr(worker, "_gpu_info", lambda: {"name": "Test GPU"})
    monkeypatch.setattr(worker, "_download", download)
    monkeypatch.setattr(worker, "run_analysis", analyze)
    monkeypatch.setattr(worker, "_upload", upload)
    result = worker.handler(_payload())
    assert result["status"] == "complete"
    assert uploads == [filename for filename, _ in ARTIFACTS.values()]
