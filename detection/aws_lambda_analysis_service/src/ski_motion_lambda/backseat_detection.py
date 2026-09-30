from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


BACKSEAT_REPORT_SCHEMA_VERSION = 3

# Continuous least-squares surface fitted to the tested 18, 22, 26, and
# 30-degree torso-lean mappings supplied in the corrected table. The fitted
# surface is intentionally extrapolated over the requested 1..80 degree torso
# range. Centering keeps the coefficients readable and numerically stable.
_HIP_LIMIT_INTERCEPT = 113.5515625
_HIP_LIMIT_KNEE = 6.389393939393938
_HIP_LIMIT_KNEE_SQUARED = -0.14867424242423866
_HIP_LIMIT_TORSO = -3.6343750000000354
_HIP_LIMIT_KNEE_TORSO = 0.2854545454545433
_HIP_LIMIT_KNEE_SQUARED_TORSO = -0.05037878787878636


@dataclass(frozen=True, slots=True)
class BackseatFilterConfig:
    torso_lean_min_degrees: float = 1.0
    torso_lean_max_degrees: float = 80.0
    average_hip_min_degrees: float = 0.0
    average_hip_max_degrees: float = 180.0
    average_knee_min_degrees: float = 0.0
    average_knee_max_degrees: float = 180.0


@dataclass(frozen=True, slots=True)
class PostureSample:
    frame_index: int
    source_frame_index: int
    timestamp_ms: float
    torso_lean: float | None
    left_hip: float | None
    right_hip: float | None
    left_knee: float | None
    right_knee: float | None


@dataclass(frozen=True, slots=True)
class BackseatFrameRange:
    range_number: int
    begin_frame_index: int
    end_frame_index: int
    begin_source_frame_index: int
    end_source_frame_index: int
    begin_time_ms: float
    end_time_ms: float
    frame_count: int


def detect_backseat_ranges(
    samples: list[PostureSample],
    config: BackseatFilterConfig = BackseatFilterConfig(),
) -> list[BackseatFrameRange]:
    """Group consecutive rows satisfying the continuous posture boundary."""

    _validate_config(config)
    matching = [sample for sample in samples if is_backseat_candidate(sample, config)]
    if not matching:
        return []

    groups: list[list[PostureSample]] = []
    for sample in matching:
        if not groups or sample.frame_index != groups[-1][-1].frame_index + 1:
            groups.append([sample])
        else:
            groups[-1].append(sample)

    return [
        BackseatFrameRange(
            range_number=index + 1,
            begin_frame_index=group[0].frame_index,
            end_frame_index=group[-1].frame_index,
            begin_source_frame_index=group[0].source_frame_index,
            end_source_frame_index=group[-1].source_frame_index,
            begin_time_ms=group[0].timestamp_ms,
            end_time_ms=group[-1].timestamp_ms,
            frame_count=len(group),
        )
        for index, group in enumerate(groups)
    ]


def is_backseat_candidate(
    sample: PostureSample,
    config: BackseatFilterConfig = BackseatFilterConfig(),
) -> bool:
    required = (
        sample.torso_lean,
        sample.left_hip,
        sample.right_hip,
        sample.left_knee,
        sample.right_knee,
    )
    if any(value is None for value in required):
        return False
    assert sample.torso_lean is not None
    assert sample.left_hip is not None and sample.right_hip is not None
    assert sample.left_knee is not None and sample.right_knee is not None
    average_hip = (sample.left_hip + sample.right_hip) / 2.0
    average_knee = (sample.left_knee + sample.right_knee) / 2.0
    return is_backseat_position(
        sample.torso_lean,
        average_hip,
        average_knee,
        config,
    )


def backseat_hip_limit_degrees(
    torso_lean_degrees: float,
    average_knee_degrees: float,
    config: BackseatFilterConfig = BackseatFilterConfig(),
) -> float | None:
    """Return the continuous hip-angle boundary, or None outside its domain."""

    _validate_config(config)
    if not _angles_are_finite(torso_lean_degrees, average_knee_degrees):
        return None
    if not (
        config.torso_lean_min_degrees
        <= torso_lean_degrees
        <= config.torso_lean_max_degrees
        and config.average_knee_min_degrees
        <= average_knee_degrees
        <= config.average_knee_max_degrees
    ):
        return None

    knee = (average_knee_degrees - 115.0) / 10.0
    torso = (torso_lean_degrees - 24.0) / 4.0
    return (
        _HIP_LIMIT_INTERCEPT
        + _HIP_LIMIT_KNEE * knee
        + _HIP_LIMIT_KNEE_SQUARED * knee * knee
        + _HIP_LIMIT_TORSO * torso
        + _HIP_LIMIT_KNEE_TORSO * knee * torso
        + _HIP_LIMIT_KNEE_SQUARED_TORSO * knee * knee * torso
    )


def is_backseat_position(
    torso_lean_degrees: float,
    average_hip_degrees: float,
    average_knee_degrees: float,
    config: BackseatFilterConfig = BackseatFilterConfig(),
) -> bool:
    """Return whether one frame falls below the continuous back-seat boundary."""

    if not _angles_are_finite(
        torso_lean_degrees,
        average_hip_degrees,
        average_knee_degrees,
    ):
        return False
    if not (
        config.average_hip_min_degrees
        <= average_hip_degrees
        <= config.average_hip_max_degrees
    ):
        return False
    hip_limit = backseat_hip_limit_degrees(
        torso_lean_degrees,
        average_knee_degrees,
        config,
    )
    return hip_limit is not None and average_hip_degrees < hip_limit


def read_posture_samples_csv(path: Path) -> list[PostureSample]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {
            "frame_index",
            "source_frame_index",
            "timestamp_ms",
            "torso_lean",
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
        }
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        return [
            PostureSample(
                frame_index=int(row["frame_index"]),
                source_frame_index=int(row["source_frame_index"]),
                timestamp_ms=float(row["timestamp_ms"]),
                torso_lean=_optional_finite_float(row["torso_lean"]),
                left_hip=_optional_finite_float(row["left_hip"]),
                right_hip=_optional_finite_float(row["right_hip"]),
                left_knee=_optional_finite_float(row["left_knee"]),
                right_knee=_optional_finite_float(row["right_knee"]),
            )
            for row in reader
        ]


def build_backseat_report(
    samples: list[PostureSample],
    config: BackseatFilterConfig = BackseatFilterConfig(),
) -> dict[str, object]:
    ranges = detect_backseat_ranges(samples, config)
    return {
        "schema_version": BACKSEAT_REPORT_SCHEMA_VERSION,
        "classification": "backseat_angle_candidate",
        "time_unit": "milliseconds",
        "conditions_degrees": {
            "torso_lean": [
                config.torso_lean_min_degrees,
                config.torso_lean_max_degrees,
            ],
            "average_hip": [
                config.average_hip_min_degrees,
                config.average_hip_max_degrees,
            ],
            "average_knee": [
                config.average_knee_min_degrees,
                config.average_knee_max_degrees,
            ],
            "qualification": "average_hip < continuous_hip_limit",
            "hip_limit_model": "quadratic_knee_with_knee_dependent_linear_torso",
        },
        "ranges": [asdict(frame_range) for frame_range in ranges],
    }


def write_backseat_report(
    angles_csv: Path,
    output_json: Path,
    config: BackseatFilterConfig = BackseatFilterConfig(),
) -> dict[str, object]:
    report = build_backseat_report(read_posture_samples_csv(angles_csv), config)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def _optional_finite_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"posture angle must be finite, got {value!r}")
    return parsed


def _angles_are_finite(*values: float) -> bool:
    return all(isinstance(value, (int, float)) and math.isfinite(value) for value in values)


def _validate_config(config: BackseatFilterConfig) -> None:
    for name in ("torso_lean", "average_hip", "average_knee"):
        minimum = getattr(config, f"{name}_min_degrees")
        maximum = getattr(config, f"{name}_max_degrees")
        if not math.isfinite(minimum) or not math.isfinite(maximum) or minimum > maximum:
            raise ValueError(f"invalid {name} range")
