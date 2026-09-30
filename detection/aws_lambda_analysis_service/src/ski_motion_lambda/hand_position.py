from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from .types import Point3D


WRIST_TORSO_PLANE_FIELDS = (
    "left_wrist_torso_plane_distance_meters",
    "right_wrist_torso_plane_distance_meters",
)
_SHOULDER_NAMES = ("left_shoulder", "right_shoulder")
_WORLD_UP = np.asarray((0.0, -1.0, 0.0))


def wrist_torso_plane_distances(
    points: Mapping[str, Point3D],
) -> dict[str, float | None]:
    """Signed wrist distances to the plane through both shoulders and world up.

    Uses the existing camera-relative MediaPipe world up (0, -1, 0), not
    calibrated gravity. Positive normal is up crossed with shoulder left-to-right.
    Field names remain unchanged for compatibility. No smoothing or clipping.
    """
    result: dict[str, float | None] = dict.fromkeys(WRIST_TORSO_PLANE_FIELDS)
    if not all(name in points for name in _SHOULDER_NAMES):
        return result
    shoulders = np.asarray([points[name].array() for name in _SHOULDER_NAMES], dtype=float)
    if not np.isfinite(shoulders).all():
        return result
    center = (shoulders[0] + shoulders[1]) / 2
    normal = np.cross(_WORLD_UP, shoulders[1] - shoulders[0])
    length = float(np.linalg.norm(normal))
    if length == 0:
        return result
    normal /= length
    for side, field in zip(("left", "right"), WRIST_TORSO_PLANE_FIELDS):
        wrist = points.get(f"{side}_wrist")
        if wrist is not None:
            position = np.asarray(wrist.array(), dtype=float)
            if np.isfinite(position).all():
                result[field] = float(np.dot(position - center, normal))
    return result
