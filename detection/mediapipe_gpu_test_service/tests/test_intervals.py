import csv
import io
import json
from types import SimpleNamespace

import pytest
import mediapipe as mp

from contract import validate_analysis_plan
from runpod_worker.pose_adapter import TaskPoseEstimator
from ski_motion_lambda.intervals import postprocess_intervals


def test_excluded_frames_skip_inference_and_interval_reset_recreates_gpu(tmp_path, monkeypatch):
    model = tmp_path / "heavy.task"
    model.write_bytes(b"fixture")
    created = []
    def create(options):
        created.append(options)
        return SimpleNamespace(close=lambda: None)
    monkeypatch.setattr(mp.tasks.vision.PoseLandmarker, "create_from_options", create)
    records = io.StringIO()
    pose = TaskPoseEstimator(.45, .45, model_path=model, raw_records=records)
    assert pose.process_excluded(100).invalid_frame
    assert json.loads(records.getvalue())["analysis_status"] == "excluded_view"
    pose.reset_interval()
    assert len(created) == 2
    assert pose._row_index == 1
    pose.close()


def test_postprocessing_never_bridges_excluded_interval(tmp_path):
    fields = ["frame_index", "source_frame_index", "timestamp_ms", "invalid_frame",
              "ankle_to_shoulder_tilt_x_degrees", "shoulder_hip_rotation_difference_degrees",
              "center_of_gravity_left_knee_plane_distance", "center_of_gravity_right_knee_plane_distance",
              "outside_knee_side", "center_of_gravity_outside_knee_plane_distance"]
    rows = []
    for i in range(180):
        excluded = 60 <= i < 120
        row = dict.fromkeys(fields, "")
        row.update(frame_index=i, source_frame_index=i * 2, timestamp_ms=round(i * 1000 / 30),
                   invalid_frame=str(excluded).lower())
        if not excluded:
            row.update(ankle_to_shoulder_tilt_x_degrees=-5 if i % 60 < 30 else 5,
                       shoulder_hip_rotation_difference_degrees=10 if i < 60 else 50,
                       center_of_gravity_left_knee_plane_distance=.1,
                       center_of_gravity_right_knee_plane_distance=.2)
        rows.append(row)
    path = tmp_path / "angles.csv"
    with path.open("w", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    intervals = [{"interval_id": "a", "start_ms": 0, "end_ms": 2000},
                 {"interval_id": "b", "start_ms": 4000, "end_ms": 6000}]
    report = postprocess_intervals(path, tmp_path / "turns.json", intervals, 30)
    assert report["turns"]
    assert all(t["end_time_ms"] <= 2000 or t["begin_time_ms"] >= 4000 for t in report["turns"])
    completed = list(csv.DictReader(path.open()))
    assert all(row["shoulder_hip_rotation_difference_degrees_smoothed"] == "" for row in completed[60:120])
    assert float(completed[59]["shoulder_hip_rotation_difference_degrees_smoothed"]) == pytest.approx(10)
    assert float(completed[120]["shoulder_hip_rotation_difference_degrees_smoothed"]) == pytest.approx(50)


def test_worker_plan_is_bound_to_source_version():
    source = {"bucket": "bucket", "key": "source.mov", "version_id": "v1"}
    plan = {"schema_version": 1, "accepted": True, "source": source, "duration_ms": 5000,
            "intervals": [{"interval_id": "a", "start_ms": 0, "end_ms": 4000}]}
    validate_analysis_plan(plan, source)
    with pytest.raises(ValueError, match="source/version"):
        validate_analysis_plan(plan, {**source, "version_id": "v2"})


def test_pipeline_preserves_original_timeline_and_skips_gap_inference(tmp_path, monkeypatch):
    import numpy as np
    from ski_motion_lambda import pipeline
    from ski_motion_lambda.types import DecodedFrame, PoseResult, VideoMetadata
    from ski_motion_render.artifacts import read_overlay_manifest
    input_path = tmp_path / "source.mov"
    input_path.write_bytes(b"fixture")
    timestamps = [0, 34, 68, 102, 136, 170]
    frame = np.zeros((8, 8, 3), dtype=np.uint8)
    calls = []
    class Estimator:
        landmark_names = ()
        def __init__(self, *args): pass
        def reset_interval(self): calls.append("reset")
        def process(self, image, timestamp):
            calls.append(timestamp)
            return PoseResult()
        def process_excluded(self, timestamp):
            return PoseResult(invalid_frame=True, invalid_frame_reason="excluded_view")
        def close(self): pass
    monkeypatch.setattr(pipeline, "PoseEstimator", Estimator)
    monkeypatch.setattr(pipeline, "probe_video", lambda path: VideoMetadata(8, 8, 60., 12, .2, 0))
    monkeypatch.setattr(pipeline, "iter_video", lambda *args, **kwargs: iter(
        DecodedFrame(frame, i * 2, timestamp) for i, timestamp in enumerate(timestamps)))
    intervals = [{"interval_id": "a", "start_ms": 0, "end_ms": 68},
                 {"interval_id": "b", "start_ms": 136, "end_ms": 200}]
    result = pipeline.analyze_video(input_path, tmp_path / "output", source={"bucket": "bucket", "key": "source.mov"},
                                    analysis_intervals=intervals)
    assert calls == ["reset", 0, 34, "reset", 136, 170]
    rows = list(csv.DictReader(result["angles_csv"].open()))
    assert [int(row["timestamp_ms"]) for row in rows] == timestamps
    assert [int(row["source_frame_index"]) for row in rows] == list(range(0, 12, 2))
    assert rows[2]["invalid_frame"].lower() == rows[3]["invalid_frame"].lower() == "true"
    condensed = list(csv.DictReader(result["condensed_angles_csv"].open()))
    assert len(condensed) == 6
    assert all(condensed[2][key] == "" for key in ("turn_direction", "torso_leg_lateral_angulation"))
    manifest = read_overlay_manifest(result["overlay_manifest"])
    assert [item.analysis_excluded for item in manifest.frames] == [False, False, True, True, False, False]
