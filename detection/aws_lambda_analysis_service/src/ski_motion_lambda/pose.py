from __future__ import annotations

import cv2
import mediapipe as mp

from .pose_quality import AnatomicalCoherenceDetector
from .smoothing import LandmarkSmoother
from .types import Point3D, PoseResult


class PoseEstimator:
    def __init__(self, visibility_threshold: float, smoothing_alpha: float) -> None:
        self._landmarker = mp.solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=False,
            enable_segmentation=False,
            min_detection_confidence=0.45,
            min_tracking_confidence=0.45,
        )
        self._visibility_threshold = visibility_threshold
        self._image_smoother = LandmarkSmoother(smoothing_alpha)
        self._world_smoother = LandmarkSmoother(smoothing_alpha)
        # Turn detection deliberately retains the pre-quality-filter smoothing
        # path. Its input must remain continuous so an invalid-frame mask never
        # creates, shifts, or removes an otherwise detected turn.
        self._turn_world_smoother = LandmarkSmoother(smoothing_alpha)
        self._anatomical_coherence = AnatomicalCoherenceDetector()
        self._names = tuple(
            landmark.name.lower() for landmark in mp.solutions.pose.PoseLandmark
        )

    @property
    def landmark_names(self) -> tuple[str, ...]:
        return self._names

    def close(self) -> None:
        self._landmarker.close()

    def process(self, bgr_frame: object, timestamp_ms: int) -> PoseResult:
        del timestamp_ms  # Legacy Pose is stateful but does not accept timestamps.
        rgb = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self._landmarker.process(rgb)
        raw_image = (
            self._convert(result.pose_landmarks.landmark)
            if result.pose_landmarks
            else {}
        )
        raw_world = (
            self._convert(result.pose_world_landmarks.landmark)
            if result.pose_world_landmarks
            else {}
        )
        turn_world = self._turn_world_smoother.update(
            raw_world,
            self._visibility_threshold,
        )
        quality = self._anatomical_coherence.evaluate(raw_image, raw_world)
        if quality.invalid_frame:
            # Preserve the current calculation path for this flagged row (and
            # therefore turn detection), but prevent the malformed pose from
            # entering temporal smoothing or influencing later frames.
            self._image_smoother.reset()
            self._world_smoother.reset()
            image = raw_image
            world = raw_world
        else:
            image = self._image_smoother.update(
                raw_image,
                self._visibility_threshold,
            )
            world = self._world_smoother.update(
                raw_world,
                self._visibility_threshold,
            )
        return PoseResult(
            raw_image=raw_image,
            image=image,
            raw_world=raw_world,
            world=world,
            turn_world=turn_world,
            invalid_frame=quality.invalid_frame,
            invalid_frame_reason=quality.reason,
        )

    def _convert(self, landmarks: object) -> dict[str, Point3D]:
        converted: dict[str, Point3D] = {}
        for name, landmark in zip(self._names, landmarks, strict=False):
            # MediaPipe's proto exposes optional scalar attributes even when the
            # field is unset. In that case ``landmark.presence`` reads as 0.0,
            # which must not be mistaken for an explicitly reported absence.
            presence = (
                float(landmark.presence)
                if landmark.HasField("presence")
                else 1.0
            )
            converted[name] = Point3D(
                x=float(landmark.x),
                y=float(landmark.y),
                z=float(landmark.z),
                visibility=float(getattr(landmark, "visibility", 0.0)),
                presence=presence,
            )
        return converted
