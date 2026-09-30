from __future__ import annotations

import cv2
import numpy as np

from .types import Point3D


POSE_CONNECTIONS = (
    ("left_shoulder", "right_shoulder"),
    ("left_shoulder", "left_elbow"),
    ("left_elbow", "left_wrist"),
    ("right_shoulder", "right_elbow"),
    ("right_elbow", "right_wrist"),
    ("left_shoulder", "left_hip"),
    ("right_shoulder", "right_hip"),
    ("left_hip", "right_hip"),
    ("left_hip", "left_knee"),
    ("left_knee", "left_ankle"),
    ("left_ankle", "left_heel"),
    ("left_heel", "left_foot_index"),
    ("left_ankle", "left_foot_index"),
    ("right_hip", "right_knee"),
    ("right_knee", "right_ankle"),
    ("right_ankle", "right_heel"),
    ("right_heel", "right_foot_index"),
    ("right_ankle", "right_foot_index"),
)

ANGLE_LABEL_VERTICES = {
    "left_hip": "left_hip",
    "right_hip": "right_hip",
    "left_knee": "left_knee",
    "right_knee": "right_knee",
    "left_ankle": "left_ankle",
    "right_ankle": "right_ankle",
}

LEFT = (80, 220, 255)
RIGHT = (255, 120, 80)
CORE = (210, 210, 210)
COG_COLOR = (0, 165, 255)
COG_RADIUS = 9


def draw_overlay(
    frame: np.ndarray,
    pose: dict[str, Point3D],
    angles: dict[str, float | None],
    timestamp_ms: int,
) -> np.ndarray:
    """Reproduce the overlay from the original combined processing service."""

    canvas = frame.copy()
    height, width = canvas.shape[:2]
    for first, second in POSE_CONNECTIONS:
        if first not in pose or second not in pose:
            continue
        a, b = pose[first], pose[second]
        if min(a.visibility, b.visibility) < 0.18:
            continue
        color = LEFT if first.startswith("left") and second.startswith("left") else RIGHT
        if first.split("_")[0] != second.split("_")[0]:
            color = CORE
        cv2.line(
            canvas,
            _pixel(a, width, height),
            _pixel(b, width, height),
            color,
            3,
            cv2.LINE_AA,
        )

    for name, point in pose.items():
        if point.visibility < 0.18:
            continue
        if name == "center_of_gravity":
            continue
        color = (
            LEFT
            if name.startswith("left")
            else RIGHT
            if name.startswith("right")
            else CORE
        )
        cv2.circle(canvas, _pixel(point, width, height), 4, color, -1, cv2.LINE_AA)

    _draw_angle_labels(canvas, pose, angles, width, height)
    _draw_panel(canvas, angles, timestamp_ms)
    center_of_gravity = pose.get("center_of_gravity")
    if center_of_gravity is not None and center_of_gravity.visibility >= 0.18:
        cv2.circle(
            canvas,
            _pixel(center_of_gravity, width, height),
            COG_RADIUS,
            COG_COLOR,
            -1,
            cv2.LINE_AA,
        )
    return canvas


def _draw_angle_labels(
    canvas: np.ndarray,
    pose: dict[str, Point3D],
    angles: dict[str, float | None],
    width: int,
    height: int,
) -> None:
    for angle_name, vertex in ANGLE_LABEL_VERTICES.items():
        value = angles.get(angle_name)
        point = pose.get(vertex)
        if value is None or point is None or point.visibility < 0.35:
            continue
        x, y = _pixel(point, width, height)
        cv2.putText(
            canvas,
            f"{value:.0f}",
            (x + 7, y - 7),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _draw_panel(
    canvas: np.ndarray,
    angles: dict[str, float | None],
    timestamp_ms: int,
) -> None:
    entries = [
        ("L knee", angles.get("left_knee"), False),
        ("R knee", angles.get("right_knee"), False),
        ("L hip", angles.get("left_hip"), False),
        ("R hip", angles.get("right_hip"), False),
        ("Torso incl", angles.get("torso_lateral_inclination"), True),
        ("Leg incl", angles.get("leg_lateral_inclination"), True),
        ("Torso ang", angles.get("torso_leg_lateral_angulation"), True),
    ]
    overlay = canvas.copy()
    panel_bottom = 37 + len(entries) * 19
    cv2.rectangle(overlay, (8, 8), (205, panel_bottom), (10, 10, 10), -1)
    cv2.addWeighted(overlay, 0.62, canvas, 0.38, 0, canvas)
    cv2.putText(
        canvas,
        f"t={timestamp_ms / 1000:.2f}s",
        (16, 27),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        CORE,
        1,
    )
    for row, (label, value, signed) in enumerate(entries, start=1):
        shown = (
            "--"
            if value is None
            else f"{value:+.1f} deg"
            if signed
            else f"{value:.1f} deg"
        )
        cv2.putText(
            canvas,
            f"{label}: {shown}",
            (16, 27 + row * 19),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.43,
            CORE,
            1,
        )


def _pixel(point: Point3D, width: int, height: int) -> tuple[int, int]:
    return int(round(point.x * width)), int(round(point.y * height))
