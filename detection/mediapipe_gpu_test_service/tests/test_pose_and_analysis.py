from __future__ import annotations

import csv
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import mediapipe as mp

from runpod_worker.analysis import join_landmark_rows
from runpod_worker.pose_adapter import TaskPoseEstimator, convert_task_landmarks


def test_task_landmarks_keep_image_and_world_scores() -> None:
    points = convert_task_landmarks(
        ("left_hip", "right_hip"),
        [
            SimpleNamespace(x=0.1, y=0.2, z=-0.3, visibility=0.8, presence=0.9),
            SimpleNamespace(x=0.4, y=0.5, z=0.6, visibility=None, presence=None),
        ],
    )
    assert points["left_hip"].z == -0.3
    assert points["left_hip"].visibility == 0.8
    assert points["right_hip"].visibility == 0.0
    assert points["right_hip"].presence == 1.0


def test_task_landmarks_reject_wrong_count() -> None:
    with pytest.raises(ValueError, match="landmark count"):
        convert_task_landmarks(("left_hip", "right_hip"), [SimpleNamespace()])


def test_pose_task_requests_gpu_video_heavy_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "heavy.task"
    model.write_bytes(b"test model placeholder")
    captured = {}

    def create(options):
        captured["options"] = options
        return SimpleNamespace(close=lambda: None)

    monkeypatch.setattr(mp.tasks.vision.PoseLandmarker, "create_from_options", create)
    estimator = TaskPoseEstimator(
        0.5, 0.3, model_path=model, raw_records=SimpleNamespace(write=lambda _: None)
    )
    options = captured["options"]
    assert options.base_options.delegate == mp.tasks.BaseOptions.Delegate.GPU
    assert options.base_options.model_asset_path == str(model)
    assert options.running_mode == mp.tasks.vision.RunningMode.VIDEO
    assert options.num_poses == 1
    estimator.close()


def test_pose_task_does_not_fallback_after_gpu_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    model = tmp_path / "heavy.task"
    model.write_bytes(b"test model placeholder")

    def fail(_options):
        raise RuntimeError("GPU delegate unavailable")

    monkeypatch.setattr(mp.tasks.vision.PoseLandmarker, "create_from_options", fail)
    with pytest.raises(RuntimeError, match="GPU delegate unavailable"):
        TaskPoseEstimator(
            0.5, 0.3, model_path=model, raw_records=SimpleNamespace(write=lambda _: None)
        )


def test_join_landmark_rows_preserves_source_frame_identity(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text(
        json.dumps({"frame_index": 0, "timestamp_ms": 33, "raw_world": {}}) + "\n"
        + json.dumps({"frame_index": 1, "timestamp_ms": 67, "raw_world": {}}) + "\n",
        encoding="utf-8",
    )
    angles_path = tmp_path / "angles.csv"
    with angles_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["frame_index", "source_frame_index", "timestamp_ms"]
        )
        writer.writeheader()
        writer.writerow({"frame_index": 0, "source_frame_index": 2, "timestamp_ms": 33})
        writer.writerow({"frame_index": 1, "source_frame_index": 4, "timestamp_ms": 67})
    output = tmp_path / "landmarks.jsonl"
    assert join_landmark_rows(raw_path, angles_path, output) == 2
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert [row["source_frame_index"] for row in rows] == [2, 4]


def test_join_landmark_rows_rejects_mismatch(tmp_path: Path) -> None:
    raw_path = tmp_path / "raw.jsonl"
    raw_path.write_text('{"frame_index":0,"timestamp_ms":33}\n', encoding="utf-8")
    angles_path = tmp_path / "angles.csv"
    angles_path.write_text(
        "frame_index,source_frame_index,timestamp_ms\n0,2,34\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="misaligned"):
        join_landmark_rows(raw_path, angles_path, tmp_path / "landmarks.jsonl")
