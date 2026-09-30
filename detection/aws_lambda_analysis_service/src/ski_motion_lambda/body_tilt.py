from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .types import Point3D


BODY_TILT_FIELDS = (
    "ankle_to_shoulder_tilt_x_degrees",
    "ankle_to_shoulder_tilt_y_degrees",
)

_CAMERA_WORLD_UP = np.asarray((0.0, -1.0, 0.0), dtype=np.float64)
_MINIMUM_HORIZONTAL_COMPONENT = 0.35


@dataclass(frozen=True, slots=True)
class SkierRelativeBodyTilt:
    """Signed body-vector tilts relative to camera-world up, in degrees."""

    x: float | None
    y: float | None


def estimate_skier_relative_body_tilt(
    points: Mapping[str, Point3D],
) -> SkierRelativeBodyTilt:
    """Measure body-vector tilt in skier-left/back camera-world axes.

    MediaPipe world Y points down, so camera-world up is (0, -1, 0).
    Positive X follows the horizontally projected pelvis axis toward the
    skier's left. Positive Y is the corresponding camera-horizontal backward
    direction.
    """

    axes = _skier_horizontal_axes(points)
    if axes is None:
        return _empty_tilts()
    positive_x, positive_y = axes

    ankle_center = _pair_midpoint(points, "left_ankle", "right_ankle")
    shoulder_center = _pair_midpoint(points, "left_shoulder", "right_shoulder")

    ankle_to_shoulder = (
        _directional_tilt(shoulder_center - ankle_center, positive_x, positive_y)
        if ankle_center is not None and shoulder_center is not None
        else None
    )
    return SkierRelativeBodyTilt(
        x=ankle_to_shoulder[0] if ankle_to_shoulder is not None else None,
        y=ankle_to_shoulder[1] if ankle_to_shoulder is not None else None,
    )


def body_tilt_csv_values(
    tilt: SkierRelativeBodyTilt,
) -> dict[str, float | None]:
    return {
        "ankle_to_shoulder_tilt_x_degrees": tilt.x,
        "ankle_to_shoulder_tilt_y_degrees": tilt.y,
    }


def _skier_horizontal_axes(
    points: Mapping[str, Point3D],
) -> tuple[np.ndarray, np.ndarray] | None:
    if "left_hip" not in points or "right_hip" not in points:
        return None

    pelvis_left = _unit_vector(
        _array(points["left_hip"]) - _array(points["right_hip"])
    )
    if pelvis_left is None:
        return None
    horizontal_left = (
        pelvis_left - np.dot(pelvis_left, _CAMERA_WORLD_UP) * _CAMERA_WORLD_UP
    )
    if float(np.linalg.norm(horizontal_left)) < _MINIMUM_HORIZONTAL_COMPONENT:
        return None
    positive_x = _unit_vector(horizontal_left)
    if positive_x is None:
        return None

    forward = _unit_vector(np.cross(_CAMERA_WORLD_UP, positive_x))
    if forward is None:
        return None
    positive_y = -forward
    return positive_x, positive_y


def _directional_tilt(
    vector: np.ndarray,
    positive_x: np.ndarray,
    positive_y: np.ndarray,
) -> tuple[float, float] | None:
    body_up = _unit_vector(vector)
    if body_up is None:
        return None
    vertical = float(np.dot(body_up, _CAMERA_WORLD_UP))
    return (
        float(math.degrees(math.atan2(float(np.dot(body_up, positive_x)), vertical))),
        float(math.degrees(math.atan2(float(np.dot(body_up, positive_y)), vertical))),
    )


def _pair_midpoint(
    points: Mapping[str, Point3D],
    left: str,
    right: str,
) -> np.ndarray | None:
    if left not in points or right not in points:
        return None
    return (_array(points[left]) + _array(points[right])) / 2.0


def _array(point: Point3D) -> np.ndarray:
    return np.asarray(point.array(), dtype=np.float64)


def _unit_vector(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-9 else None


def _empty_tilts() -> SkierRelativeBodyTilt:
    return SkierRelativeBodyTilt(None, None)
