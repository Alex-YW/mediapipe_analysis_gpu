"""Reuse ARCS metric/render pipelines with a GPU-only pose adapter."""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import threading
from itertools import zip_longest
from pathlib import Path
from typing import Any
from unittest.mock import patch

import mediapipe as mp

from ski_motion_lambda import pipeline as analysis_pipeline
from ski_motion_render.artifacts import read_overlay_manifest
from ski_motion_render.pipeline import render_video

from .pose_adapter import TaskPoseEstimator


# The existing Service 1 entry point constructs its estimator through a module
# symbol. Keep the temporary replacement confined to one analysis at a time.
_ANALYSIS_LOCK = threading.Lock()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def join_landmark_rows(raw_path: Path, angles_path: Path, output_path: Path) -> int:
    """Add authoritative source-frame IDs from the completed angles artifact."""

    count = 0
    with raw_path.open(encoding="utf-8") as raw_file, angles_path.open(
        newline="", encoding="utf-8"
    ) as angles_file, output_path.open("w", encoding="utf-8") as output:
        for raw_line, angle in zip_longest(
            raw_file, csv.DictReader(angles_file), fillvalue=None
        ):
            if raw_line is None or angle is None:
                raise ValueError("raw landmark count differs from angles rows")
            raw = json.loads(raw_line)
            if (
                raw["frame_index"] != int(angle["frame_index"])
                or raw["timestamp_ms"] != int(angle["timestamp_ms"])
            ):
                raise ValueError("raw landmarks and angles are misaligned")
            raw["source_frame_index"] = int(angle["source_frame_index"])
            output.write(json.dumps(raw, separators=(",", ":"), allow_nan=False) + "\n")
            count += 1
    if count == 0:
        raise ValueError("no landmark rows were produced")
    return count


def check_video(path: Path, expected_frames: int) -> dict[str, Any]:
    completed = subprocess.run(
        [
            "ffprobe", "-v", "error", "-select_streams", "v:0",
            "-show_entries", "stream=codec_name,width,height,r_frame_rate,nb_frames,duration",
            "-of", "json", str(path),
        ],
        capture_output=True, text=True, check=True,
    )
    streams = json.loads(completed.stdout).get("streams", [])
    if len(streams) != 1 or streams[0].get("codec_name") != "h264":
        raise ValueError("annotated output is not a single H.264 video")
    if int(streams[0].get("nb_frames", -1)) != expected_frames:
        raise ValueError("annotated frame count differs from analysis")
    return streams[0]


def run_analysis(
    video: Path,
    output_dir: Path,
    *,
    model_path: Path,
    source: dict[str, object],
    processing_frame_rate: float = 30.0,
) -> dict[str, Any]:
    if not model_path.is_file():
        raise FileNotFoundError(f"Pose model missing: {model_path}")
    output_dir.mkdir(parents=True, exist_ok=False)
    raw_path = output_dir / "raw-landmarks.jsonl"
    with raw_path.open("w", encoding="utf-8") as raw_records:

        def estimator_factory(visibility_threshold: float, smoothing_alpha: float):
            return TaskPoseEstimator(
                visibility_threshold,
                smoothing_alpha,
                model_path=model_path,
                raw_records=raw_records,
            )

        with _ANALYSIS_LOCK:
            with patch.object(analysis_pipeline, "PoseEstimator", estimator_factory):
                analysis = analysis_pipeline.analyze_video(
                    video,
                    output_dir,
                    source=source,
                    processing_frame_rate=processing_frame_rate,
                )

    landmark_count = join_landmark_rows(
        raw_path, analysis["angles_csv"], output_dir / "landmarks.jsonl"
    )
    if landmark_count != analysis["frames"]:
        raise ValueError("landmark count differs from analyzed frames")
    manifest = read_overlay_manifest(analysis["overlay_manifest"])
    if len(manifest.frames) != analysis["frames"]:
        raise ValueError("manifest count differs from analyzed frames")
    render = render_video(video, output_dir / "annotated.mp4", manifest)
    if render["frames"] != analysis["frames"]:
        raise ValueError("render count differs from analyzed frames")
    video_probe = check_video(render["annotated_video"], analysis["frames"])

    return {
        "model_api": "mediapipe.tasks.vision.PoseLandmarker",
        "model_variant": "heavy",
        "delegate_requested": "GPU",
        "model_sha256": sha256(model_path),
        "mediapipe_version": mp.__version__,
        "landmark_rows": landmark_count,
        "analysis": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in analysis.items()
        },
        "render": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in render.items()
        },
        "video_probe": video_probe,
    }
