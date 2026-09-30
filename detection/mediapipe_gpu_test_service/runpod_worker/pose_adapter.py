"""GPU Pose Landmarker adapter retaining Service 1's post-processing rules.

The task API replaces only raw pose inference. This isolated adapter mirrors
Service 1 PoseEstimator's smoothing and anatomical-gate sequence; production
Service 1 is not modified.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from typing import Any

import cv2
import mediapipe as mp

from ski_motion_lambda.pose_quality import AnatomicalCoherenceDetector
from ski_motion_lambda.smoothing import LandmarkSmoother
from ski_motion_lambda.types import Point3D, PoseResult


def _score(value: object, default: float) -> float:
    if value is None:
        return default
    number = float(value)
    return number if math.isfinite(number) else default


def convert_task_landmarks(
    names: tuple[str, ...], landmarks: list[Any] | None
) -> dict[str, Point3D]:
    if not landmarks:
        return {}
    if len(landmarks) != len(names):
        raise ValueError("GPU task returned an unexpected landmark count")
    converted = {}
    for name, point in zip(names, landmarks, strict=True):
        coords = tuple(float(getattr(point, axis)) for axis in ("x", "y", "z"))
        if not all(math.isfinite(value) for value in coords):
            continue
        converted[name] = Point3D(
            *coords,
            visibility=_score(getattr(point, "visibility", None), 0.0),
            presence=_score(getattr(point, "presence", None), 1.0),
        )
    return converted


def _points(points: dict[str, Point3D]) -> dict[str, dict[str, float]]:
    return {name: asdict(point) for name, point in points.items()}


class TaskPoseEstimator:
    """One GPU task instance per video; no CPU fallback is permitted."""

    def __init__(
        self,
        visibility_threshold: float,
        smoothing_alpha: float,
        *,
        model_path: Path,
        raw_records: Any,
    ) -> None:
        if not model_path.is_file():
            raise FileNotFoundError(f"Pose Landmarker model missing: {model_path}")
        options = mp.tasks.vision.PoseLandmarkerOptions(
            base_options=mp.tasks.BaseOptions(
                model_asset_path=str(model_path),
                delegate=mp.tasks.BaseOptions.Delegate.GPU,
            ),
            running_mode=mp.tasks.vision.RunningMode.VIDEO,
            num_poses=1,
            min_pose_detection_confidence=0.45,
            min_pose_presence_confidence=0.45,
            min_tracking_confidence=0.45,
            output_segmentation_masks=False,
        )
        # A GPU initialization error propagates; this test must never silently
        # become a CPU run. MediaPipe may still use CPU for non-inference work.
        self._landmarker = mp.tasks.vision.PoseLandmarker.create_from_options(options)
        self._visibility_threshold = visibility_threshold
        self._image_smoother = LandmarkSmoother(smoothing_alpha)
        self._world_smoother = LandmarkSmoother(smoothing_alpha)
        self._turn_world_smoother = LandmarkSmoother(smoothing_alpha)
        self._anatomical_coherence = AnatomicalCoherenceDetector()
        self._names = tuple(
            landmark.name.lower() for landmark in mp.solutions.pose.PoseLandmark
        )
        self._raw_records = raw_records
        self._row_index = 0

    @property
    def landmark_names(self) -> tuple[str, ...]:
        return self._names

    def close(self) -> None:
        self._landmarker.close()

    def process(self, bgr_frame: Any, timestamp_ms: int) -> PoseResult:
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        task_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb,
        )
        result = self._landmarker.detect_for_video(task_image, timestamp_ms)
        image_poses = result.pose_landmarks or []
        world_poses = result.pose_world_landmarks or []
        raw_image = convert_task_landmarks(
            self._names, image_poses[0] if image_poses else None
        )
        raw_world = convert_task_landmarks(
            self._names, world_poses[0] if world_poses else None
        )

        # This sequence intentionally matches Service 1 PoseEstimator.process.
        turn_world = self._turn_world_smoother.update(
            raw_world, self._visibility_threshold
        )
        quality = self._anatomical_coherence.evaluate(raw_image, raw_world)
        if quality.invalid_frame:
            self._image_smoother.reset()
            self._world_smoother.reset()
            image, world = raw_image, raw_world
        else:
            image = self._image_smoother.update(
                raw_image, self._visibility_threshold
            )
            world = self._world_smoother.update(
                raw_world, self._visibility_threshold
            )
        pose = PoseResult(
            raw_image=raw_image,
            image=image,
            raw_world=raw_world,
            world=world,
            turn_world=turn_world,
            invalid_frame=quality.invalid_frame,
            invalid_frame_reason=quality.reason,
        )
        self._raw_records.write(
            json.dumps(
                {
                    "frame_index": self._row_index,
                    "timestamp_ms": timestamp_ms,
                    "raw_image": _points(raw_image),
                    "raw_world": _points(raw_world),
                    "image": _points(image),
                    "world": _points(world),
                    "turn_world": _points(turn_world),
                    "invalid_frame": quality.invalid_frame,
                    "invalid_frame_reason": quality.reason,
                },
                separators=(",", ":"),
                allow_nan=False,
            ) + "\n"
        )
        self._row_index += 1
        return pose
