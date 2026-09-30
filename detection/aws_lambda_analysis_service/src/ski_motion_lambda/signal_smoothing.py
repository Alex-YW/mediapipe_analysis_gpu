from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path
from statistics import median

from .pose_quality import INVALID_FRAME_FIELD


SHOULDER_HIP_ROTATION_SMOOTHED_FIELD = (
    "shoulder_hip_rotation_difference_degrees_smoothed"
)
_SOURCE_FIELD = "shoulder_hip_rotation_difference_degrees"


@dataclass(frozen=True, slots=True)
class RotationSmoothingConfig:
    maximum_gap_frames: int = 3
    median_window_frames: int = 3
    butterworth_order: int = 2
    cutoff_hz: float = 2.5


def smooth_axial_angles(
    values: list[float | None],
    sampling_rate_hz: float,
    config: RotationSmoothingConfig = RotationSmoothingConfig(),
    *,
    hard_gap_indices: frozenset[int] = frozenset(),
) -> list[float | None]:
    """Smooth an axial angle series while retaining long and edge gaps."""

    _validate_config(config, sampling_rate_hz)
    filled = _interpolate_short_gaps(
        values,
        config.maximum_gap_frames,
        hard_gap_indices,
    )
    result: list[float | None] = [None] * len(values)
    for start, end in _continuous_ranges(filled):
        unwrapped = _unwrap_axial([float(value) for value in filled[start:end]])
        filtered = _median_filter(unwrapped, config.median_window_frames)
        if len(filtered) >= 10:
            filtered = _zero_phase_second_order_lowpass(
                filtered,
                sampling_rate_hz,
                config.cutoff_hz,
            )
        result[start:end] = [_wrap_axial(value) for value in filtered]
    return result


def write_smoothed_rotation_column(
    path: Path,
    sampling_rate_hz: float | None = None,
    config: RotationSmoothingConfig = RotationSmoothingConfig(),
) -> None:
    """Add or replace the smoothed shoulder-hip rotation CSV column."""

    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    required = {"timestamp_ms", _SOURCE_FIELD}
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(sorted(missing))}"
        )
    if sampling_rate_hz is None:
        sampling_rate_hz = _sampling_rate_from_rows(rows)

    invalid_indices = frozenset(
        index for index, row in enumerate(rows) if _is_invalid_frame(row)
    )
    values = [
        None
        if _is_invalid_frame(row)
        else _optional_finite_float(row[_SOURCE_FIELD])
        for row in rows
    ]
    smoothed = smooth_axial_angles(
        values,
        sampling_rate_hz,
        config,
        hard_gap_indices=invalid_indices,
    )
    if SHOULDER_HIP_ROTATION_SMOOTHED_FIELD in fieldnames:
        fieldnames.remove(SHOULDER_HIP_ROTATION_SMOOTHED_FIELD)
    source_index = fieldnames.index(_SOURCE_FIELD)
    fieldnames.insert(source_index + 1, SHOULDER_HIP_ROTATION_SMOOTHED_FIELD)
    for row, value in zip(rows, smoothed, strict=True):
        row[SHOULDER_HIP_ROTATION_SMOOTHED_FIELD] = (
            "" if value is None else str(value)
        )

    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _interpolate_short_gaps(
    values: list[float | None],
    maximum_gap_frames: int,
    hard_gap_indices: frozenset[int] = frozenset(),
) -> list[float | None]:
    filled = list(values)
    index = 0
    while index < len(values):
        if values[index] is not None:
            index += 1
            continue
        start = index
        while index < len(values) and values[index] is None:
            index += 1
        end = index
        gap_length = end - start
        if (
            start == 0
            or end == len(values)
            or gap_length > maximum_gap_frames
            or any(index in hard_gap_indices for index in range(start, end))
        ):
            continue
        left = float(values[start - 1])
        right = _nearest_axial_equivalent(float(values[end]), left)
        for offset in range(1, gap_length + 1):
            fraction = offset / (gap_length + 1)
            filled[start + offset - 1] = left + fraction * (right - left)
    return filled


def _continuous_ranges(values: list[float | None]):
    index = 0
    while index < len(values):
        while index < len(values) and values[index] is None:
            index += 1
        start = index
        while index < len(values) and values[index] is not None:
            index += 1
        if start < index:
            yield start, index


def _unwrap_axial(values: list[float]) -> list[float]:
    if not values:
        return []
    unwrapped = [values[0]]
    for value in values[1:]:
        unwrapped.append(_nearest_axial_equivalent(value, unwrapped[-1]))
    return unwrapped


def _nearest_axial_equivalent(value: float, reference: float) -> float:
    return value + 180.0 * round((reference - value) / 180.0)


def _median_filter(values: list[float], window: int) -> list[float]:
    radius = window // 2
    padded = [values[0]] * radius + values + [values[-1]] * radius
    return [median(padded[index : index + window]) for index in range(len(values))]


def _zero_phase_second_order_lowpass(
    values: list[float], sampling_rate_hz: float, cutoff_hz: float
) -> list[float]:
    tangent = math.tan(math.pi * cutoff_hz / sampling_rate_hz)
    normalizer = 1.0 / (1.0 + math.sqrt(2.0) * tangent + tangent * tangent)
    coefficients = (
        tangent * tangent * normalizer,
        2.0 * tangent * tangent * normalizer,
        tangent * tangent * normalizer,
        2.0 * (tangent * tangent - 1.0) * normalizer,
        (1.0 - math.sqrt(2.0) * tangent + tangent * tangent) * normalizer,
    )
    padding = min(9, len(values) - 1)
    extended = (
        [2.0 * values[0] - value for value in values[1 : padding + 1][::-1]]
        + values
        + [
            2.0 * values[-1] - value
            for value in values[-padding - 1 : -1][::-1]
        ]
    )
    forward = _apply_second_order_filter(extended, coefficients)
    backward = _apply_second_order_filter(forward[::-1], coefficients)[::-1]
    return backward[padding : padding + len(values)]


def _apply_second_order_filter(
    values: list[float], coefficients: tuple[float, float, float, float, float]
) -> list[float]:
    b0, b1, b2, a1, a2 = coefficients
    x1 = x2 = y1 = y2 = values[0]
    output = []
    for value in values:
        filtered = b0 * value + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
        output.append(filtered)
        x2, x1 = x1, value
        y2, y1 = y1, filtered
    return output


def _wrap_axial(value: float) -> float:
    wrapped = (value + 90.0) % 180.0 - 90.0
    return 90.0 if math.isclose(wrapped, -90.0) and value > 0 else wrapped


def _optional_finite_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"rotation metric must be finite, got {value!r}")
    return parsed


def _is_invalid_frame(row: dict[str, str]) -> bool:
    value = row.get(INVALID_FRAME_FIELD, "").strip().lower()
    return value in {"1", "true"}


def _sampling_rate_from_rows(rows: list[dict[str, str]]) -> float:
    timestamps = [float(row["timestamp_ms"]) for row in rows]
    if len(timestamps) < 2 or timestamps[-1] <= timestamps[0]:
        raise ValueError("At least two increasing timestamps are required")
    if any(
        right <= left for left, right in zip(timestamps, timestamps[1:])
    ):
        raise ValueError("CSV timestamps must be strictly increasing")
    return (len(timestamps) - 1) * 1000.0 / (timestamps[-1] - timestamps[0])


def _validate_config(
    config: RotationSmoothingConfig, sampling_rate_hz: float
) -> None:
    if config.maximum_gap_frames < 0:
        raise ValueError("maximum_gap_frames must be non-negative")
    if config.median_window_frames < 1 or config.median_window_frames % 2 == 0:
        raise ValueError("median_window_frames must be a positive odd integer")
    if config.butterworth_order != 2:
        raise ValueError("only a second-order Butterworth filter is supported")
    if not math.isfinite(sampling_rate_hz) or sampling_rate_hz <= 0:
        raise ValueError("sampling_rate_hz must be positive and finite")
    if not 0 < config.cutoff_hz < sampling_rate_hz / 2:
        raise ValueError("cutoff_hz must be between zero and the Nyquist frequency")
