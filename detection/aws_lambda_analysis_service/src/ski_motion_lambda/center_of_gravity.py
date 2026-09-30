from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import numpy as np

from .types import Point3D


CENTER_OF_GRAVITY_LANDMARK = "center_of_gravity"
CENTER_OF_GRAVITY_FIELDS = (
    "center_of_gravity_x",
    "center_of_gravity_y",
    "center_of_gravity_z",
    "center_of_gravity_mass_coverage",
    "center_of_gravity_feet_distance_x",
    "center_of_gravity_feet_distance_y",
    "center_of_gravity_left_knee_plane_distance",
    "center_of_gravity_right_knee_plane_distance",
    "outside_knee_side",
    "center_of_gravity_outside_knee_plane_distance",
)


@dataclass(frozen=True, slots=True)
class CenterOfGravityEstimate:
    """Mass-weighted whole-body center estimate for one pose frame."""

    point: Point3D
    mass_coverage: float


@dataclass(frozen=True, slots=True)
class BodyUpPlaneDisplacement:
    """CoG displacement in the plane perpendicular to estimated body-up."""

    x: float
    y: float


@dataclass(frozen=True, slots=True)
class _Segment:
    mass_fraction: float
    proximal: str | tuple[str, str]
    distal: str | tuple[str, str]
    center_fraction: float


# Generic adult segment parameters adapted to MediaPipe's available joint
# centers. Fractions are measured from the proximal endpoint and the masses sum
# to 1.0. This is an estimate, not a subject-specific biomechanical measurement.
_SEGMENTS = (
    _Segment(0.497, ("left_shoulder", "right_shoulder"), ("left_hip", "right_hip"), 0.500),
    _Segment(0.028, "left_shoulder", "left_elbow", 0.436),
    _Segment(0.028, "right_shoulder", "right_elbow", 0.436),
    _Segment(0.016, "left_elbow", "left_wrist", 0.430),
    _Segment(0.016, "right_elbow", "right_wrist", 0.430),
    _Segment(0.006, "left_wrist", "left_index", 0.506),
    _Segment(0.006, "right_wrist", "right_index", 0.506),
    _Segment(0.100, "left_hip", "left_knee", 0.433),
    _Segment(0.100, "right_hip", "right_knee", 0.433),
    _Segment(0.0465, "left_knee", "left_ankle", 0.433),
    _Segment(0.0465, "right_knee", "right_ankle", 0.433),
    _Segment(0.0145, "left_ankle", "left_foot_index", 0.500),
    _Segment(0.0145, "right_ankle", "right_foot_index", 0.500),
)
_HEAD_MASS_FRACTION = 0.081
_MINIMUM_MASS_COVERAGE = 0.75
_MINIMUM_FOOT_DIRECTION_DOT = 0.0
_MINIMUM_PLANAR_FOOT_COMPONENT = 0.35
_CAMERA_WORLD_UP = np.asarray((0.0, -1.0, 0.0), dtype=np.float64)
_MINIMUM_HORIZONTAL_TRANSVERSE_COMPONENT = 0.35
_MAXIMUM_REFERENCE_DISAGREEMENT_DOT = -0.25


@dataclass(frozen=True, slots=True)
class HipParallelKneePlaneDistances:
    """Signed CoG distances from hip-parallel planes through each knee."""

    left: float | None
    right: float | None


class KneePlaneDistanceTracker:
    """Measure CoG from vertical hip-parallel planes through each knee.

    Positive distance is anatomically forward. Each plane contains camera-world
    up, is parallel to the plane through both hips, and passes through one knee.
    The preceding valid normal prevents isolated left/right landmark reversals
    from flipping the sign.
    """

    def __init__(self) -> None:
        self._previous_forward: np.ndarray | None = None

    def update(
        self,
        center_of_gravity: CenterOfGravityEstimate | None,
        points: Mapping[str, Point3D],
    ) -> HipParallelKneePlaneDistances:
        if center_of_gravity is None:
            return HipParallelKneePlaneDistances(None, None)
        left_hip = points.get("left_hip")
        right_hip = points.get("right_hip")
        if left_hip is None or right_hip is None:
            return HipParallelKneePlaneDistances(None, None)

        hip_left = _horizontal_left_axis(
            np.asarray(left_hip.array(), dtype=np.float64)
            - np.asarray(right_hip.array(), dtype=np.float64)
        )
        if hip_left is None:
            return HipParallelKneePlaneDistances(None, None)

        forward = _unit_vector(np.cross(_CAMERA_WORLD_UP, hip_left))
        if forward is None:
            return HipParallelKneePlaneDistances(None, None)
        reference_forward = _shoulder_forward_reference(points)

        if self._previous_forward is not None:
            if float(np.dot(forward, self._previous_forward)) < 0.0:
                forward = -forward
            if (
                reference_forward is not None
                and float(np.dot(forward, reference_forward))
                < _MAXIMUM_REFERENCE_DISAGREEMENT_DOT
            ):
                return HipParallelKneePlaneDistances(None, None)
        elif (
            reference_forward is not None
            and float(np.dot(forward, reference_forward)) < 0.0
        ):
            forward = -forward

        self._previous_forward = forward
        center = np.asarray(center_of_gravity.point.array(), dtype=np.float64)
        distances: list[float | None] = []
        for knee_name in ("left_knee", "right_knee"):
            knee = points.get(knee_name)
            distances.append(
                None
                if knee is None
                else float(
                    np.dot(
                        center - np.asarray(knee.array(), dtype=np.float64),
                        forward,
                    )
                )
            )
        return HipParallelKneePlaneDistances(*distances)


def estimate_center_of_gravity(
    points: Mapping[str, Point3D],
    *,
    minimum_mass_coverage: float = _MINIMUM_MASS_COVERAGE,
) -> CenterOfGravityEstimate | None:
    """Estimate whole-body CoG/CoM from available smoothed pose landmarks.

    Missing segments are omitted and the remaining mass is renormalized. An
    estimate is returned only when enough nominal body mass is represented.
    """

    if not 0 < minimum_mass_coverage <= 1:
        raise ValueError("minimum_mass_coverage must be in (0, 1]")

    weighted_points: list[tuple[float, Point3D]] = []
    head = _head_center(points)
    if head is not None:
        weighted_points.append((_HEAD_MASS_FRACTION, head))

    for segment in _SEGMENTS:
        proximal = _endpoint(points, segment.proximal)
        distal = _endpoint(points, segment.distal)
        if proximal is None or distal is None:
            continue
        weighted_points.append(
            (
                segment.mass_fraction,
                _interpolate(proximal, distal, segment.center_fraction),
            )
        )

    mass_coverage = sum(mass for mass, _ in weighted_points)
    if mass_coverage + 1e-9 < minimum_mass_coverage:
        return None

    return CenterOfGravityEstimate(
        point=Point3D(
            x=sum(mass * point.x for mass, point in weighted_points) / mass_coverage,
            y=sum(mass * point.y for mass, point in weighted_points) / mass_coverage,
            z=sum(mass * point.z for mass, point in weighted_points) / mass_coverage,
            visibility=sum(
                mass * point.visibility for mass, point in weighted_points
            )
            / mass_coverage,
            presence=sum(mass * point.presence for mass, point in weighted_points)
            / mass_coverage,
        ),
        mass_coverage=min(1.0, mass_coverage),
    )


def center_of_gravity_csv_values(
    estimate: CenterOfGravityEstimate | None,
    body_up_plane: BodyUpPlaneDisplacement | None = None,
    knee_plane_distances: HipParallelKneePlaneDistances | None = None,
) -> dict[str, float | str | None]:
    if estimate is None:
        return {name: None for name in CENTER_OF_GRAVITY_FIELDS}
    return {
        "center_of_gravity_x": estimate.point.x,
        "center_of_gravity_y": estimate.point.y,
        "center_of_gravity_z": estimate.point.z,
        "center_of_gravity_mass_coverage": estimate.mass_coverage,
        "center_of_gravity_feet_distance_x": (
            body_up_plane.x if body_up_plane is not None else None
        ),
        "center_of_gravity_feet_distance_y": (
            body_up_plane.y if body_up_plane is not None else None
        ),
        "center_of_gravity_left_knee_plane_distance": (
            knee_plane_distances.left if knee_plane_distances is not None else None
        ),
        "center_of_gravity_right_knee_plane_distance": (
            knee_plane_distances.right if knee_plane_distances is not None else None
        ),
        "outside_knee_side": None,
        "center_of_gravity_outside_knee_plane_distance": None,
    }


def estimate_body_up_plane_displacement(
    center_of_gravity: CenterOfGravityEstimate | None,
    points: Mapping[str, Point3D],
) -> BodyUpPlaneDisplacement | None:
    """Project CoG into a skier-relative plane normal to estimated body-up.

    Body-up runs from the ankle midpoint to the hip midpoint. The averaged
    heel-to-toe direction is projected perpendicular to body-up before it is
    used as the fore/aft axis. Positive X points toward the skier's left and
    negative Y points in the projected heel-to-toe direction.
    """

    if center_of_gravity is None:
        return None
    required = (
        "left_heel",
        "right_heel",
        "left_foot_index",
        "right_foot_index",
        "left_ankle",
        "right_ankle",
        "left_hip",
        "right_hip",
    )
    if not all(name in points for name in required):
        return None

    left_heel = np.asarray(points["left_heel"].array(), dtype=np.float64)
    right_heel = np.asarray(points["right_heel"].array(), dtype=np.float64)
    left_toe = np.asarray(points["left_foot_index"].array(), dtype=np.float64)
    right_toe = np.asarray(points["right_foot_index"].array(), dtype=np.float64)
    left_ankle = np.asarray(points["left_ankle"].array(), dtype=np.float64)
    right_ankle = np.asarray(points["right_ankle"].array(), dtype=np.float64)
    left_hip = np.asarray(points["left_hip"].array(), dtype=np.float64)
    right_hip = np.asarray(points["right_hip"].array(), dtype=np.float64)

    left_forward = _unit_vector(left_toe - left_heel)
    right_forward = _unit_vector(right_toe - right_heel)
    if left_forward is None or right_forward is None:
        return None
    if float(np.dot(left_forward, right_forward)) < _MINIMUM_FOOT_DIRECTION_DOT:
        return None

    raw_forward = _unit_vector(left_forward + right_forward)
    if raw_forward is None:
        return None
    left_center = (left_heel + left_toe) / 2.0
    right_center = (right_heel + right_toe) / 2.0
    feet_center = (left_center + right_center) / 2.0
    ankle_center = (left_ankle + right_ankle) / 2.0
    hip_center = (left_hip + right_hip) / 2.0

    body_up = _unit_vector(hip_center - ankle_center)
    if body_up is None:
        return None

    # Remove the body-up component that previously allowed the feet-to-CoG
    # height to leak into the reported fore/aft distance.
    forward_in_plane = raw_forward - np.dot(raw_forward, body_up) * body_up
    if np.linalg.norm(forward_in_plane) < _MINIMUM_PLANAR_FOOT_COMPONENT:
        return None
    forward = _unit_vector(forward_in_plane)
    if forward is None:
        return None

    # Right-to-left defines positive X. Remove body-up and forward components
    # so X and Y remain within the same orthogonal reference plane.
    positive_x = left_center - right_center
    positive_x -= np.dot(positive_x, body_up) * body_up
    positive_x -= np.dot(positive_x, forward) * forward
    positive_x = _unit_vector(positive_x)
    if positive_x is None:
        return None

    displacement = (
        np.asarray(center_of_gravity.point.array(), dtype=np.float64) - feet_center
    )
    displacement_in_plane = displacement - np.dot(displacement, body_up) * body_up
    return BodyUpPlaneDisplacement(
        x=float(np.dot(displacement_in_plane, positive_x)),
        y=float(np.dot(displacement_in_plane, -forward)),
    )


def _head_center(points: Mapping[str, Point3D]) -> Point3D | None:
    for pair in (
        ("left_ear", "right_ear"),
        ("left_eye", "right_eye"),
    ):
        center = _endpoint(points, pair)
        if center is not None:
            return center
    return points.get("nose")


def _endpoint(
    points: Mapping[str, Point3D],
    landmark: str | tuple[str, str],
) -> Point3D | None:
    if isinstance(landmark, str):
        return points.get(landmark)
    left = points.get(landmark[0])
    right = points.get(landmark[1])
    if left is None or right is None:
        return None
    return _interpolate(left, right, 0.5)


def _interpolate(proximal: Point3D, distal: Point3D, fraction: float) -> Point3D:
    inverse = 1.0 - fraction
    return Point3D(
        x=inverse * proximal.x + fraction * distal.x,
        y=inverse * proximal.y + fraction * distal.y,
        z=inverse * proximal.z + fraction * distal.z,
        visibility=min(proximal.visibility, distal.visibility),
        presence=min(proximal.presence, distal.presence),
    )


def _unit_vector(vector: np.ndarray) -> np.ndarray | None:
    length = float(np.linalg.norm(vector))
    return vector / length if length > 1e-9 else None


def _horizontal_left_axis(vector: np.ndarray) -> np.ndarray | None:
    left = _unit_vector(vector)
    if left is None:
        return None
    horizontal = left - np.dot(left, _CAMERA_WORLD_UP) * _CAMERA_WORLD_UP
    if float(np.linalg.norm(horizontal)) < _MINIMUM_HORIZONTAL_TRANSVERSE_COMPONENT:
        return None
    return _unit_vector(horizontal)


def _shoulder_forward_reference(
    points: Mapping[str, Point3D],
) -> np.ndarray | None:
    left = points.get("left_shoulder")
    right = points.get("right_shoulder")
    if left is None or right is None:
        return None
    left_axis = _horizontal_left_axis(
        np.asarray(left.array(), dtype=np.float64)
        - np.asarray(right.array(), dtype=np.float64)
    )
    if left_axis is None:
        return None
    return _unit_vector(np.cross(_CAMERA_WORLD_UP, left_axis))
