from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from statistics import median

from .types import Point3D


INVALID_FRAME_FIELD = "invalid_frame"
INVALID_FRAME_REASON_FIELD = "invalid_frame_reason"
ANATOMICAL_COHERENCE_REASON = "anatomical_coherence"

_CORE_SEGMENTS = (
    ("left_shoulder", "right_shoulder"),
    ("left_hip", "right_hip"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "left_knee"),
    ("right_hip", "right_knee"),
    ("left_knee", "left_ankle"),
    ("right_knee", "right_ankle"),
)
_MINIMUM_HISTORY = 5
_HISTORY_SIZE = 15
_MAXIMUM_RATIO_CHANGE = 0.75
_MINIMUM_CHANGED_SEGMENTS = 3
_MINIMUM_SEGMENT_LENGTH = 1e-4
_MAXIMUM_WORLD_SCALE_CONTRACTION = 0.20
_MAXIMUM_STABLE_SCALE_CHANGE = 0.20
_RECOVERY_STABLE_FRAMES = 5


@dataclass(frozen=True, slots=True)
class FrameQuality:
    invalid_frame: bool = False
    reason: str = ""


class AnatomicalCoherenceDetector:
    """Reject non-rigid, high-confidence landmark collapses before smoothing.

    Segment lengths are normalized by their per-frame median so a camera zoom,
    which changes the image scale uniformly, does not by itself invalidate a
    frame. A separate world-landmark scale check catches a pose-model collapse
    which is uniform in image space but implausibly contracts the inferred
    anatomical model. The detector learns a short, video-local baseline from
    accepted frames and rejects only large, multi-segment discontinuities.
    """

    def __init__(self) -> None:
        self._history: deque[tuple[float, ...]] = deque(maxlen=_HISTORY_SIZE)
        self._world_scale_history: deque[float] = deque(maxlen=_HISTORY_SIZE)
        self._previous_image_scale: float | None = None
        self._previous_world_scale: float | None = None
        self._recovering = False
        self._stable_recovery_frames = 0

    def evaluate(
        self,
        image_points: dict[str, Point3D],
        world_points: dict[str, Point3D],
    ) -> FrameQuality:
        image_measurement = _anatomical_measurement(image_points, include_z=False)
        world_measurement = _anatomical_measurement(world_points, include_z=True)
        if image_measurement is None or world_measurement is None:
            # Existing visibility/presence handling owns incomplete poses. This
            # detector is deliberately narrow: it marks implausible geometry,
            # not ordinary landmark absence or occlusion.
            return FrameQuality()
        signature, image_scale = image_measurement
        _, world_scale = world_measurement

        if len(self._history) >= _MINIMUM_HISTORY:
            baseline = tuple(median(values) for values in zip(*self._history, strict=True))
            changed = _changed_segments(
                signature,
                baseline,
                _MAXIMUM_RATIO_CHANGE,
            )
            world_scale_baseline = median(self._world_scale_history)
            world_contracts = (
                world_scale
                < world_scale_baseline * (1 - _MAXIMUM_WORLD_SCALE_CONTRACTION)
            )
            shape_is_incoherent = changed >= _MINIMUM_CHANGED_SEGMENTS
            collapse_is_incoherent = world_contracts
            if shape_is_incoherent or collapse_is_incoherent:
                self._recovering = True
                self._stable_recovery_frames = 0

        if self._recovering:
            if self._is_stable(image_scale, world_scale):
                self._stable_recovery_frames += 1
            else:
                self._stable_recovery_frames = 0
            self._remember_observation(image_scale, world_scale)
            if self._stable_recovery_frames >= _RECOVERY_STABLE_FRAMES:
                self._recovering = False
                self._stable_recovery_frames = 0
            return FrameQuality(True, ANATOMICAL_COHERENCE_REASON)

        self._history.append(signature)
        self._world_scale_history.append(world_scale)
        self._remember_observation(image_scale, world_scale)
        return FrameQuality()

    def _is_stable(self, image_scale: float, world_scale: float) -> bool:
        if self._previous_image_scale is None or self._previous_world_scale is None:
            return False
        return (
            _relative_change(image_scale, self._previous_image_scale)
            <= _MAXIMUM_STABLE_SCALE_CHANGE
            and _relative_change(world_scale, self._previous_world_scale)
            <= _MAXIMUM_STABLE_SCALE_CHANGE
        )

    def _remember_observation(
        self,
        image_scale: float,
        world_scale: float,
    ) -> None:
        self._previous_image_scale = image_scale
        self._previous_world_scale = world_scale


def _changed_segments(
    signature: tuple[float, ...],
    baseline: tuple[float, ...],
    threshold: float,
) -> int:
    return sum(
        _relative_change(current, expected) > threshold
        for current, expected in zip(signature, baseline, strict=True)
    )


def _anatomical_signature(points: dict[str, Point3D]) -> tuple[float, ...] | None:
    measurement = _anatomical_measurement(points, include_z=False)
    return measurement[0] if measurement is not None else None


def _anatomical_measurement(
    points: dict[str, Point3D],
    *,
    include_z: bool,
) -> tuple[tuple[float, ...], float] | None:
    lengths: list[float] = []
    for start_name, end_name in _CORE_SEGMENTS:
        start = points.get(start_name)
        end = points.get(end_name)
        if start is None or end is None:
            return None
        if not _finite_point(start) or not _finite_point(end):
            return None
        length = (
            math.dist(start.array(), end.array())
            if include_z
            else math.hypot(start.x - end.x, start.y - end.y)
        )
        if length <= _MINIMUM_SEGMENT_LENGTH:
            return None
        lengths.append(length)

    scale = median(lengths)
    if scale <= _MINIMUM_SEGMENT_LENGTH:
        return None
    return tuple(length / scale for length in lengths), scale


def _finite_point(point: Point3D) -> bool:
    return all(math.isfinite(value) for value in (point.x, point.y, point.z))


def _relative_change(current: float, expected: float) -> float:
    if expected <= 0:
        return math.inf
    return abs(current - expected) / expected
