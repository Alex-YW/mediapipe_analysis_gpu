"""Camera-view lower-leg alignment from normalized image landmarks."""

from __future__ import annotations

import math
from collections.abc import Mapping

from .types import Point3D


SHIN_SYMMETRY_FIELDS = (
    "left_shin_angle_deg",
    "right_shin_angle_deg",
    "lower_leg_asymmetry_deg",
)


def calculate_shin_symmetry(
    points: Mapping[str, Point3D],
    display_width: int,
    display_height: int,
    *,
    minimum_confidence: float = 0.45,
) -> dict[str, float | None]:
    """Measure projected ankle-to-knee angles and their unsigned difference.

    The image coordinates are normalized independently by width and height, so
    both axes must be converted to pixels before comparing line directions.
    Positive shin angle means the knee appears to the image-right of its ankle.
    This is a camera-view posture proxy, not a ski edge-angle measurement.
    """

    if display_width <= 0 or display_height <= 0:
        raise ValueError("display dimensions must be positive")
    if not 0 < minimum_confidence <= 1:
        raise ValueError("minimum_confidence must be in (0, 1]")

    left = _shin_angle(points, "left", display_width, display_height, minimum_confidence)
    right = _shin_angle(points, "right", display_width, display_height, minimum_confidence)
    gap = None
    if left is not None and right is not None:
        gap = abs((left - right + 180.0) % 360.0 - 180.0)
    return dict(zip(SHIN_SYMMETRY_FIELDS, (left, right, gap), strict=True))


def _shin_angle(
    points: Mapping[str, Point3D],
    side: str,
    width: int,
    height: int,
    minimum_confidence: float,
) -> float | None:
    knee = points.get(f"{side}_knee")
    ankle = points.get(f"{side}_ankle")
    if knee is None or ankle is None:
        return None
    if not all(
        math.isfinite(value)
        for point in (knee, ankle)
        for value in (point.x, point.y, point.visibility, point.presence)
    ):
        return None
    if min(knee.visibility, knee.presence, ankle.visibility, ankle.presence) < minimum_confidence:
        return None

    horizontal = (knee.x - ankle.x) * width
    upward = (ankle.y - knee.y) * height
    if math.hypot(horizontal, upward) <= 1e-9:
        return None
    return math.degrees(math.atan2(horizontal, upward))
