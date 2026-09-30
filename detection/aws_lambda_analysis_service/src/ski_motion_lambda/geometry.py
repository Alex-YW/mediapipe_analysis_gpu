from __future__ import annotations

import math
from collections.abc import Mapping

import numpy as np

from .types import Point3D
from .hand_position import WRIST_TORSO_PLANE_FIELDS, wrist_torso_plane_distances


SHOULDER_HIP_ROTATION_FIELD = "shoulder_hip_rotation_difference_degrees"
SHOULDER_HIP_BODY_RELATIVE_ROTATION_FIELD = (
    "shoulder_hip_body_relative_rotation_difference_degrees"
)
_CAMERA_WORLD_UP = np.asarray((0.0, -1.0, 0.0), dtype=np.float64)
_MINIMUM_HORIZONTAL_COMPONENT = 0.35


def angle_at(a: Point3D, vertex: Point3D, c: Point3D) -> float | None:
    va = np.asarray(a.array(), dtype=np.float64) - np.asarray(
        vertex.array(), dtype=np.float64
    )
    vc = np.asarray(c.array(), dtype=np.float64) - np.asarray(
        vertex.array(), dtype=np.float64
    )
    return angle_between(va, vc)


def angle_between(a: np.ndarray, b: np.ndarray) -> float | None:
    denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denominator <= 1e-9:
        return None
    cosine = float(np.clip(np.dot(a, b) / denominator, -1.0, 1.0))
    return float(math.degrees(math.acos(cosine)))


def midpoint(a: Point3D, b: Point3D) -> Point3D:
    return Point3D(
        x=(a.x + b.x) / 2,
        y=(a.y + b.y) / 2,
        z=(a.z + b.z) / 2,
        visibility=min(a.visibility, b.visibility),
        presence=min(a.presence, b.presence),
    )


JOINT_TRIPLETS: dict[str, tuple[str, str, str]] = {
    "left_hip": ("left_shoulder", "left_hip", "left_knee"),
    "right_hip": ("right_shoulder", "right_hip", "right_knee"),
    "left_knee": ("left_hip", "left_knee", "left_ankle"),
    "right_knee": ("right_hip", "right_knee", "right_ankle"),
    "left_ankle": ("left_knee", "left_ankle", "left_foot_index"),
    "right_ankle": ("right_knee", "right_ankle", "right_foot_index"),
}

ANGLE_NAMES = tuple(JOINT_TRIPLETS) + (
    "torso_left_thigh",
    "torso_right_thigh",
    "shoulder_alignment",
    "hip_alignment",
    "torso_lean",
    "torso_lateral_inclination",
    "leg_lateral_inclination",
    "torso_leg_lateral_angulation",
    SHOULDER_HIP_ROTATION_FIELD,
    SHOULDER_HIP_BODY_RELATIVE_ROTATION_FIELD,
    *WRIST_TORSO_PLANE_FIELDS,
)


def calculate_angles(points: Mapping[str, Point3D]) -> dict[str, float | None]:
    angles: dict[str, float | None] = {}
    for name, (a_name, b_name, c_name) in JOINT_TRIPLETS.items():
        if not all(key in points for key in (a_name, b_name, c_name)):
            angles[name] = None
        else:
            angles[name] = angle_at(points[a_name], points[b_name], points[c_name])

    shoulders = _pair_midpoint(points, "left_shoulder", "right_shoulder")
    hips = _pair_midpoint(points, "left_hip", "right_hip")
    if shoulders and hips:
        torso = np.asarray(shoulders.array()) - np.asarray(hips.array())
        for side in ("left", "right"):
            hip = points.get(f"{side}_hip")
            knee = points.get(f"{side}_knee")
            angles[f"torso_{side}_thigh"] = (
                angle_between(
                    torso,
                    np.asarray(knee.array()) - np.asarray(hip.array()),
                )
                if hip and knee
                else None
            )
        angles["torso_lean"] = angle_between(
            torso,
            np.asarray((0.0, -1.0, 0.0)),
        )
    else:
        angles["torso_left_thigh"] = None
        angles["torso_right_thigh"] = None
        angles["torso_lean"] = None

    angles["shoulder_alignment"] = _line_to_horizontal(
        points.get("left_shoulder"),
        points.get("right_shoulder"),
    )
    angles["hip_alignment"] = _line_to_horizontal(
        points.get("left_hip"),
        points.get("right_hip"),
    )
    angles.update(_lateral_angulation(points))
    angles[SHOULDER_HIP_ROTATION_FIELD] = shoulder_hip_rotation_difference(points)
    angles[SHOULDER_HIP_BODY_RELATIVE_ROTATION_FIELD] = (
        shoulder_hip_body_relative_rotation_difference(points)
    )
    angles.update(wrist_torso_plane_distances(points))
    return {name: angles.get(name) for name in ANGLE_NAMES}


def shoulder_hip_body_relative_rotation_difference(
    points: Mapping[str, Point3D],
) -> float | None:
    """Return the raw signed atan2 angle around the hip-to-shoulder axis.

    Positive is clockwise viewed from the head toward the pelvis. No axial
    folding, turn normalization, confidence threshold, or smoothing is applied.
    Only missing, non-finite, or geometrically undefined inputs return None.
    """
    names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(name in points for name in names):
        return None
    left_shoulder, right_shoulder, left_hip, right_hip = (
        np.asarray(points[name].array(), dtype=np.float64) for name in names
    )
    if not all(np.isfinite(vector).all() for vector in (
        left_shoulder, right_shoulder, left_hip, right_hip
    )):
        return None
    up = (left_shoulder + right_shoulder - left_hip - right_hip) / 2
    up_length = float(np.linalg.norm(up))
    if up_length == 0:
        return None
    up /= up_length
    shoulder = right_shoulder - left_shoulder
    hip = right_hip - left_hip
    shoulder -= np.dot(shoulder, up) * up
    hip -= np.dot(hip, up) * up
    if float(np.linalg.norm(shoulder)) == 0 or float(np.linalg.norm(hip)) == 0:
        return None
    return -math.degrees(math.atan2(
        float(np.dot(up, np.cross(hip, shoulder))),
        float(np.dot(hip, shoulder)),
    ))


def shoulder_hip_rotation_difference(
    points: Mapping[str, Point3D],
) -> float | None:
    """Return clockwise shoulder rotation relative to hips, viewed from above.

    Left-to-right shoulder and hip axes are projected into the camera-relative
    horizontal plane. Their signed axial difference is wrapped to [-90, 90].
    """

    names = ("left_shoulder", "right_shoulder", "left_hip", "right_hip")
    if not all(name in points for name in names):
        return None
    shoulder_axis = _horizontal_axis(
        np.asarray(points["right_shoulder"].array(), dtype=np.float64)
        - np.asarray(points["left_shoulder"].array(), dtype=np.float64)
    )
    hip_axis = _horizontal_axis(
        np.asarray(points["right_hip"].array(), dtype=np.float64)
        - np.asarray(points["left_hip"].array(), dtype=np.float64)
    )
    if shoulder_axis is None or hip_axis is None:
        return None

    counterclockwise = math.degrees(
        math.atan2(
            float(np.dot(_CAMERA_WORLD_UP, np.cross(hip_axis, shoulder_axis))),
            float(np.clip(np.dot(hip_axis, shoulder_axis), -1.0, 1.0)),
        )
    )
    clockwise = -counterclockwise
    return _wrap_axial_angle(clockwise)


def _horizontal_axis(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    if length <= 1e-9:
        return None
    horizontal = vector - np.dot(vector, _CAMERA_WORLD_UP) * _CAMERA_WORLD_UP
    horizontal_length = float(np.linalg.norm(horizontal))
    if horizontal_length / length < _MINIMUM_HORIZONTAL_COMPONENT:
        return None
    return horizontal / horizontal_length


def _wrap_axial_angle(value: float) -> float:
    wrapped = (value + 90.0) % 180.0 - 90.0
    return 90.0 if math.isclose(wrapped, -90.0) and value > 0 else wrapped


def _lateral_angulation(
    points: Mapping[str, Point3D],
) -> dict[str, float | None]:
    names = (
        "left_shoulder",
        "right_shoulder",
        "left_hip",
        "right_hip",
        "left_ankle",
        "right_ankle",
    )
    if not all(name in points for name in names):
        return _empty_lateral_angulation()

    shoulder_center = midpoint(points["left_shoulder"], points["right_shoulder"])
    hip_center = midpoint(points["left_hip"], points["right_hip"])
    ankle_center = midpoint(points["left_ankle"], points["right_ankle"])
    lateral_axis = _body_lateral_axis(points)
    if lateral_axis is None:
        return _empty_lateral_angulation()

    torso = np.asarray(shoulder_center.array()) - np.asarray(hip_center.array())
    legs = np.asarray(hip_center.array()) - np.asarray(ankle_center.array())
    torso_inclination = _signed_lateral_inclination(torso, lateral_axis)
    leg_inclination = _signed_lateral_inclination(legs, lateral_axis)
    angulation = (
        _wrap_signed_angle(torso_inclination - leg_inclination)
        if torso_inclination is not None and leg_inclination is not None
        else None
    )
    return {
        "torso_lateral_inclination": torso_inclination,
        "leg_lateral_inclination": leg_inclination,
        "torso_leg_lateral_angulation": angulation,
    }


def _body_lateral_axis(points: Mapping[str, Point3D]) -> np.ndarray | None:
    up = np.asarray((0.0, -1.0, 0.0))
    axes: list[np.ndarray] = []
    for left, right in (
        ("left_hip", "right_hip"),
        ("left_shoulder", "right_shoulder"),
    ):
        vector = np.asarray(points[right].array()) - np.asarray(points[left].array())
        horizontal = vector - np.dot(vector, up) * up
        length = float(np.linalg.norm(horizontal))
        if length <= 1e-9:
            continue
        horizontal /= length
        if axes and np.dot(horizontal, axes[0]) < 0:
            horizontal *= -1
        axes.append(horizontal)
    if not axes:
        return None
    combined = np.sum(axes, axis=0)
    length = float(np.linalg.norm(combined))
    return combined / length if length > 1e-9 else None


def _signed_lateral_inclination(
    vector: np.ndarray,
    lateral_axis: np.ndarray,
) -> float | None:
    up = np.asarray((0.0, -1.0, 0.0))
    vertical_component = float(np.dot(vector, up))
    lateral_component = float(np.dot(vector, lateral_axis))
    if math.hypot(vertical_component, lateral_component) <= 1e-9:
        return None
    return float(math.degrees(math.atan2(lateral_component, vertical_component)))


def _wrap_signed_angle(value: float) -> float:
    return (value + 180.0) % 360.0 - 180.0


def _empty_lateral_angulation() -> dict[str, None]:
    return {
        "torso_lateral_inclination": None,
        "leg_lateral_inclination": None,
        "torso_leg_lateral_angulation": None,
    }


def _pair_midpoint(
    points: Mapping[str, Point3D],
    left: str,
    right: str,
) -> Point3D | None:
    if left not in points or right not in points:
        return None
    return midpoint(points[left], points[right])


def _line_to_horizontal(a: Point3D | None, b: Point3D | None) -> float | None:
    if a is None or b is None:
        return None
    vector = np.asarray(b.array()) - np.asarray(a.array())
    value = angle_between(vector, np.asarray((1.0, 0.0, 0.0)))
    if value is None:
        return None
    return min(value, 180.0 - value)
