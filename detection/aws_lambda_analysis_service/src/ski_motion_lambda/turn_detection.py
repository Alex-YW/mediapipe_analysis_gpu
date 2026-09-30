from __future__ import annotations

import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path


TURN_METRIC = "ankle_to_shoulder_tilt_x_degrees"
TURN_REPORT_SCHEMA_VERSION = 2
LEFT_KNEE_DISTANCE_FIELD = "center_of_gravity_left_knee_plane_distance"
RIGHT_KNEE_DISTANCE_FIELD = "center_of_gravity_right_knee_plane_distance"
OUTSIDE_KNEE_SIDE_FIELD = "outside_knee_side"
OUTSIDE_KNEE_DISTANCE_FIELD = "center_of_gravity_outside_knee_plane_distance"
OUTSIDE_KNEE_TRANSITION_EXCLUSION_FRAMES = 0


@dataclass(frozen=True, slots=True)
class TurnDetectionConfig:
    edge_exclusion_fraction: float = 0.05
    confirmation_frames: int = 10
    minimum_excursion_degrees: float = 2.0
    apex_range_fraction: float = 0.30


@dataclass(frozen=True, slots=True)
class TiltSample:
    frame_index: int
    source_frame_index: int
    timestamp_ms: float
    value: float | None


@dataclass(frozen=True, slots=True)
class TurnRange:
    turn_number: int
    direction: str
    begin_time_ms: float
    end_time_ms: float
    apex_start_time_ms: float | None
    apex_time_ms: float | None
    apex_end_time_ms: float | None
    begin_observed: bool = True
    end_observed: bool = True
    apex_observed: bool = True


@dataclass(frozen=True, slots=True)
class _Crossing:
    sample_index: int
    timestamp_ms: float
    direction: str
    sign: int


def detect_turns(
    samples: list[TiltSample],
    config: TurnDetectionConfig = TurnDetectionConfig(),
) -> list[TurnRange]:
    """Detect skier-left/right turns from signed lateral body tilt.

    Negative-to-positive crossings begin left turns and positive-to-negative
    crossings begin right turns. A crossing is accepted only when the next
    configured number of frames remain strictly on the new side of zero and
    reach the minimum excursion.
    """

    _validate_config(config)
    if len(samples) < 2:
        return []
    _validate_samples(samples)

    filled = interpolate_middle_gaps(samples, config.edge_exclusion_fraction)
    crossings = _confirmed_crossings(filled, config)
    turns: list[TurnRange] = []
    # Keep complete turns exactly as before: two accepted starts bound each one.
    for crossing_index, crossing in enumerate(crossings[:-1]):
        end_time_ms = crossings[crossing_index + 1].timestamp_ms
        if end_time_ms <= crossing.timestamp_ms:
            continue
        apex = _apex_times(
            filled,
            crossing,
            end_time_ms,
            config.apex_range_fraction,
        )
        if apex is None:
            continue
        turns.append(
            TurnRange(
                turn_number=len(turns) + 1,
                direction=crossing.direction,
                begin_time_ms=_rounded_time(crossing.timestamp_ms),
                end_time_ms=_rounded_time(end_time_ms),
                apex_start_time_ms=_rounded_time(apex[0]),
                apex_time_ms=_rounded_time(apex[1]),
                apex_end_time_ms=_rounded_time(apex[2]),
            )
        )

    partials: list[TurnRange] = []
    if crossings:
        first = crossings[0]
        initial_sign = -first.sign
        initial_start = _preceding_qualified_run_start(
            filled, first.sample_index, initial_sign, config
        )
        if initial_start is not None:
            partials.append(
                _partial_turn(
                    filled,
                    start_index=initial_start,
                    end_time_ms=first.timestamp_ms,
                    sign=initial_sign,
                    begin_observed=False,
                    end_observed=True,
                    apex_range_fraction=config.apex_range_fraction,
                )
            )

        last = crossings[-1]
        last_matching = max(
            (
                index for index in range(last.sample_index, len(filled))
                if filled[index].value is not None
                and float(filled[index].value) * last.sign > 0
            ),
            default=None,
        )
        if last_matching is not None:
            partials.append(
                _partial_turn(
                    filled,
                    start_index=last.sample_index,
                    end_time_ms=_exclusive_end_time(filled, last_matching),
                    sign=last.sign,
                    begin_observed=True,
                    end_observed=False,
                    apex_range_fraction=config.apex_range_fraction,
                    begin_time_ms=last.timestamp_ms,
                )
            )
    else:
        # Without a crossing, a stable lean is insufficient evidence on its
        # own. Accept only one signed lobe whose apex is visible on both sides.
        signed = [
            (index, 1 if sample.value > 0 else -1)
            for index, sample in enumerate(filled)
            if sample.value is not None and sample.value != 0
        ]
        if signed and len({sign for _, sign in signed}) == 1:
            sign = signed[0][1]
            start = _preceding_qualified_run_start(
                filled, len(filled), sign, config
            )
            if start is not None:
                candidate = _partial_turn(
                    filled,
                    start_index=start,
                    end_time_ms=_exclusive_end_time(filled, signed[-1][0]),
                    sign=sign,
                    begin_observed=False,
                    end_observed=False,
                    apex_range_fraction=config.apex_range_fraction,
                )
                if candidate.apex_observed:
                    partials.append(candidate)

    ordered = sorted((*turns, *partials), key=lambda turn: turn.begin_time_ms)
    return [
        TurnRange(
            turn_number=index,
            direction=turn.direction,
            begin_time_ms=turn.begin_time_ms,
            end_time_ms=turn.end_time_ms,
            apex_start_time_ms=turn.apex_start_time_ms,
            apex_time_ms=turn.apex_time_ms,
            apex_end_time_ms=turn.apex_end_time_ms,
            begin_observed=turn.begin_observed,
            end_observed=turn.end_observed,
            apex_observed=turn.apex_observed,
        )
        for index, turn in enumerate(ordered, start=1)
    ]


def _preceding_qualified_run_start(
    samples: list[TiltSample],
    stop_index: int,
    sign: int,
    config: TurnDetectionConfig,
) -> int | None:
    """Find the nearest sustained signed run before a known turn boundary."""
    end = stop_index
    while end > 0:
        while end > 0 and (
            samples[end - 1].value is None
            or float(samples[end - 1].value) * sign <= 0
        ):
            end -= 1
        if end == 0:
            return None
        start = end - 1
        while start > 0 and (
            samples[start - 1].value is not None
            and float(samples[start - 1].value) * sign > 0
        ):
            start -= 1
        run = samples[start:end]
        if len(run) >= config.confirmation_frames and max(
            float(sample.value) * sign for sample in run if sample.value is not None
        ) >= config.minimum_excursion_degrees:
            return start
        end = start
    return None


def _exclusive_end_time(samples: list[TiltSample], last_index: int) -> float:
    if last_index + 1 < len(samples):
        return samples[last_index + 1].timestamp_ms
    return samples[-1].timestamp_ms + (
        samples[-1].timestamp_ms - samples[-2].timestamp_ms
    )


def _partial_turn(
    samples: list[TiltSample],
    *,
    start_index: int,
    end_time_ms: float,
    sign: int,
    begin_observed: bool,
    end_observed: bool,
    apex_range_fraction: float,
    begin_time_ms: float | None = None,
) -> TurnRange:
    begin = samples[start_index].timestamp_ms if begin_time_ms is None else begin_time_ms
    apex = _observed_partial_apex(
        [
            sample for sample in samples[start_index:]
            if begin <= sample.timestamp_ms < end_time_ms
        ],
        sign,
        apex_range_fraction,
    )
    return TurnRange(
        turn_number=0,
        direction="left" if sign > 0 else "right",
        begin_time_ms=_rounded_time(begin),
        end_time_ms=_rounded_time(end_time_ms),
        apex_start_time_ms=_rounded_time(apex[0]) if apex else None,
        apex_time_ms=_rounded_time(apex[1]) if apex else None,
        apex_end_time_ms=_rounded_time(apex[2]) if apex else None,
        begin_observed=begin_observed,
        end_observed=end_observed,
        apex_observed=apex is not None,
    )


def _observed_partial_apex(
    samples: list[TiltSample],
    sign: int,
    apex_range_fraction: float,
) -> tuple[float, float, float] | None:
    if len(samples) < 3:
        return None
    aligned = [
        float(sample.value) * sign if sample.value is not None else None
        for sample in samples
    ]
    valid_indices = [index for index, value in enumerate(aligned) if value is not None]
    if not valid_indices:
        return None
    peak_index = max(valid_indices, key=lambda index: float(aligned[index]))
    peak = float(aligned[peak_index])
    if peak <= 0:
        return None
    threshold = peak * (1.0 - apex_range_fraction)
    start = peak_index
    while start > 0 and aligned[start - 1] is not None and float(aligned[start - 1]) >= threshold:
        start -= 1
    end = peak_index
    while end + 1 < len(aligned) and aligned[end + 1] is not None and float(aligned[end + 1]) >= threshold:
        end += 1
    if (start == 0 or end + 1 == len(samples)
            or aligned[start - 1] is None or aligned[end + 1] is None):
        return None
    return (
        _threshold_crossing_time(
            samples[start - 1], float(aligned[start - 1]),
            samples[start], float(aligned[start]), threshold,
        ),
        samples[peak_index].timestamp_ms,
        _threshold_crossing_time(
            samples[end], float(aligned[end]),
            samples[end + 1], float(aligned[end + 1]), threshold,
        ),
    )


def interpolate_middle_gaps(
    samples: list[TiltSample],
    edge_exclusion_fraction: float = 0.05,
) -> list[TiltSample]:
    """Fill only middle-region gaps with shape-preserving cubic interpolation."""

    if not samples:
        return []
    if not 0 <= edge_exclusion_fraction < 0.5:
        raise ValueError("edge_exclusion_fraction must be in [0, 0.5)")
    _validate_samples(samples)

    valid_indices = [index for index, sample in enumerate(samples) if sample.value is not None]
    if len(valid_indices) < 2:
        return list(samples)
    times = [samples[index].timestamp_ms for index in valid_indices]
    values = [float(samples[index].value) for index in valid_indices]
    derivatives = _pchip_derivatives(times, values)

    middle_start = math.ceil(len(samples) * edge_exclusion_fraction)
    middle_end = math.floor(len(samples) * (1.0 - edge_exclusion_fraction))
    filled = list(samples)
    interval_index = 0
    for index in range(middle_start, middle_end):
        if samples[index].value is not None:
            continue
        timestamp = samples[index].timestamp_ms
        while (
            interval_index + 1 < len(times)
            and times[interval_index + 1] < timestamp
        ):
            interval_index += 1
        if interval_index + 1 >= len(times) or timestamp <= times[interval_index]:
            continue
        value = _pchip_value(
            timestamp,
            times[interval_index],
            times[interval_index + 1],
            values[interval_index],
            values[interval_index + 1],
            derivatives[interval_index],
            derivatives[interval_index + 1],
        )
        filled[index] = TiltSample(
            frame_index=samples[index].frame_index,
            source_frame_index=samples[index].source_frame_index,
            timestamp_ms=timestamp,
            value=value,
        )
    return filled


def read_tilt_samples_csv(path: Path) -> list[TiltSample]:
    with path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        required = {"frame_index", "source_frame_index", "timestamp_ms", TURN_METRIC}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"CSV is missing required columns: {', '.join(sorted(missing))}")
        samples = []
        for row in reader:
            samples.append(
                TiltSample(
                    frame_index=int(row["frame_index"]),
                    source_frame_index=int(row["source_frame_index"]),
                    timestamp_ms=float(row["timestamp_ms"]),
                    value=_optional_finite_float(row[TURN_METRIC]),
                )
            )
    return samples


def build_turn_report(
    samples: list[TiltSample],
    config: TurnDetectionConfig = TurnDetectionConfig(),
) -> dict[str, object]:
    turns = detect_turns(samples, config)
    return {
        "schema_version": TURN_REPORT_SCHEMA_VERSION,
        "source_metric": TURN_METRIC,
        "time_unit": "milliseconds",
        "direction_convention": {
            "negative_to_positive": "left",
            "positive_to_negative": "right",
        },
        "parameters": {
            "edge_exclusion_fraction": config.edge_exclusion_fraction,
            "missing_data_interpolation": "pchip",
            "confirmation_frames": config.confirmation_frames,
            "minimum_excursion_degrees": config.minimum_excursion_degrees,
            "apex_range_fraction": config.apex_range_fraction,
        },
        "turns": [asdict(turn) for turn in turns],
    }


def write_turn_report(
    angles_csv: Path,
    output_json: Path,
    config: TurnDetectionConfig = TurnDetectionConfig(),
) -> dict[str, object]:
    report = build_turn_report(read_tilt_samples_csv(angles_csv), config)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return report


def write_outside_knee_plane_columns(
    angles_csv: Path,
    turn_report: dict[str, object],
    *,
    transition_exclusion_frames: int = OUTSIDE_KNEE_TRANSITION_EXCLUSION_FRAMES,
) -> None:
    """Select the outside-knee plane distance for detected turn frames.

    A skier-left turn uses the right knee as the outside knee, while a
    skier-right turn uses the left. Frames outside detected turns remain
    empty. Transition frames are included by default.
    """

    if (
        isinstance(transition_exclusion_frames, bool)
        or not isinstance(transition_exclusion_frames, int)
        or transition_exclusion_frames < 0
    ):
        raise ValueError("transition_exclusion_frames must be a non-negative integer")
    with angles_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = list(reader.fieldnames or ())
        rows = list(reader)
    required = {
        "timestamp_ms",
        LEFT_KNEE_DISTANCE_FIELD,
        RIGHT_KNEE_DISTANCE_FIELD,
        OUTSIDE_KNEE_SIDE_FIELD,
        OUTSIDE_KNEE_DISTANCE_FIELD,
    }
    missing = required.difference(fieldnames)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(sorted(missing))}"
        )

    for row in rows:
        row[OUTSIDE_KNEE_SIDE_FIELD] = ""
        row[OUTSIDE_KNEE_DISTANCE_FIELD] = ""

    turns = turn_report.get("turns")
    if not isinstance(turns, list):
        raise ValueError("turn report must contain a turns list")
    for turn in turns:
        if not isinstance(turn, dict):
            raise ValueError("turn entries must be objects")
        direction = turn.get("direction")
        if direction == "left":
            outside_side = "right"
            source_field = RIGHT_KNEE_DISTANCE_FIELD
        elif direction == "right":
            outside_side = "left"
            source_field = LEFT_KNEE_DISTANCE_FIELD
        else:
            raise ValueError(f"unsupported turn direction: {direction!r}")
        begin = float(turn["begin_time_ms"])
        end = float(turn["end_time_ms"])
        indices = [
            index
            for index, row in enumerate(rows)
            if begin <= float(row["timestamp_ms"]) < end
        ]
        if transition_exclusion_frames:
            indices = indices[
                transition_exclusion_frames:-transition_exclusion_frames
            ]
        for index in indices:
            value = rows[index][source_field]
            if not value.strip():
                continue
            rows[index][OUTSIDE_KNEE_SIDE_FIELD] = outside_side
            rows[index][OUTSIDE_KNEE_DISTANCE_FIELD] = value

    temporary = angles_csv.with_suffix(angles_csv.suffix + ".tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as destination:
            writer = csv.DictWriter(destination, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(angles_csv)
    finally:
        if temporary.exists():
            temporary.unlink()


def _confirmed_crossings(
    samples: list[TiltSample],
    config: TurnDetectionConfig,
) -> list[_Crossing]:
    crossings: list[_Crossing] = []
    for index in range(1, len(samples)):
        previous = samples[index - 1]
        current = samples[index]
        if previous.value is None or current.value is None:
            continue
        sign = 0
        direction = ""
        if previous.value <= 0 < current.value:
            sign, direction = 1, "left"
        elif previous.value >= 0 > current.value:
            sign, direction = -1, "right"
        else:
            continue
        if crossings and crossings[-1].sign == sign:
            continue
        confirmation = samples[index : index + config.confirmation_frames]
        if len(confirmation) < config.confirmation_frames:
            continue
        aligned = []
        for sample in confirmation:
            if sample.value is None or sample.value * sign <= 0:
                break
            aligned.append(sample.value * sign)
        if len(aligned) < config.confirmation_frames:
            continue
        if max(aligned) < config.minimum_excursion_degrees:
            continue
        crossings.append(
            _Crossing(
                sample_index=index,
                timestamp_ms=_zero_crossing_time(previous, current),
                direction=direction,
                sign=sign,
            )
        )
    return crossings


def _apex_times(
    samples: list[TiltSample],
    crossing: _Crossing,
    end_time_ms: float,
    apex_range_fraction: float,
) -> tuple[float, float, float] | None:
    turn_samples = [
        sample
        for sample in samples[crossing.sample_index :]
        if sample.timestamp_ms <= end_time_ms and sample.value is not None
    ]
    if not turn_samples:
        return None
    aligned = [float(sample.value) * crossing.sign for sample in turn_samples]
    peak_index = max(range(len(aligned)), key=aligned.__getitem__)
    peak_value = aligned[peak_index]
    if peak_value <= 0:
        return None
    threshold = peak_value * (1.0 - apex_range_fraction)

    start_index = peak_index
    while start_index > 0 and aligned[start_index - 1] >= threshold:
        start_index -= 1
    end_index = peak_index
    while end_index + 1 < len(aligned) and aligned[end_index + 1] >= threshold:
        end_index += 1

    apex_start = turn_samples[start_index].timestamp_ms
    if start_index > 0:
        apex_start = _threshold_crossing_time(
            turn_samples[start_index - 1],
            aligned[start_index - 1],
            turn_samples[start_index],
            aligned[start_index],
            threshold,
        )
    apex_end = turn_samples[end_index].timestamp_ms
    if end_index + 1 < len(turn_samples):
        apex_end = _threshold_crossing_time(
            turn_samples[end_index],
            aligned[end_index],
            turn_samples[end_index + 1],
            aligned[end_index + 1],
            threshold,
        )
    return (
        max(crossing.timestamp_ms, apex_start),
        turn_samples[peak_index].timestamp_ms,
        min(end_time_ms, apex_end),
    )


def _zero_crossing_time(left: TiltSample, right: TiltSample) -> float:
    assert left.value is not None and right.value is not None
    magnitude = abs(left.value) + abs(right.value)
    if magnitude == 0:
        return right.timestamp_ms
    ratio = abs(left.value) / magnitude
    return left.timestamp_ms + ratio * (right.timestamp_ms - left.timestamp_ms)


def _threshold_crossing_time(
    left: TiltSample,
    left_value: float,
    right: TiltSample,
    right_value: float,
    threshold: float,
) -> float:
    difference = right_value - left_value
    if abs(difference) < 1e-12:
        return right.timestamp_ms
    ratio = (threshold - left_value) / difference
    ratio = min(1.0, max(0.0, ratio))
    return left.timestamp_ms + ratio * (right.timestamp_ms - left.timestamp_ms)


def _pchip_derivatives(x: list[float], y: list[float]) -> list[float]:
    if len(x) == 2:
        slope = (y[1] - y[0]) / (x[1] - x[0])
        return [slope, slope]
    h = [x[index + 1] - x[index] for index in range(len(x) - 1)]
    delta = [(y[index + 1] - y[index]) / h[index] for index in range(len(h))]
    derivatives = [0.0] * len(x)
    for index in range(1, len(x) - 1):
        if delta[index - 1] * delta[index] <= 0:
            derivatives[index] = 0.0
            continue
        weight_left = 2.0 * h[index] + h[index - 1]
        weight_right = h[index] + 2.0 * h[index - 1]
        derivatives[index] = (weight_left + weight_right) / (
            weight_left / delta[index - 1] + weight_right / delta[index]
        )
    derivatives[0] = _pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
    derivatives[-1] = _pchip_endpoint_slope(
        h[-1], h[-2], delta[-1], delta[-2]
    )
    return derivatives


def _pchip_endpoint_slope(
    adjacent_interval: float,
    next_interval: float,
    adjacent_slope: float,
    next_slope: float,
) -> float:
    slope = (
        (2.0 * adjacent_interval + next_interval) * adjacent_slope
        - adjacent_interval * next_slope
    ) / (adjacent_interval + next_interval)
    if slope * adjacent_slope <= 0:
        return 0.0
    if adjacent_slope * next_slope < 0 and abs(slope) > 3.0 * abs(adjacent_slope):
        return 3.0 * adjacent_slope
    return slope


def _pchip_value(
    timestamp: float,
    left_time: float,
    right_time: float,
    left_value: float,
    right_value: float,
    left_derivative: float,
    right_derivative: float,
) -> float:
    interval = right_time - left_time
    fraction = (timestamp - left_time) / interval
    fraction_squared = fraction * fraction
    fraction_cubed = fraction_squared * fraction
    return (
        (2.0 * fraction_cubed - 3.0 * fraction_squared + 1.0) * left_value
        + (fraction_cubed - 2.0 * fraction_squared + fraction)
        * interval
        * left_derivative
        + (-2.0 * fraction_cubed + 3.0 * fraction_squared) * right_value
        + (fraction_cubed - fraction_squared) * interval * right_derivative
    )


def _optional_finite_float(value: str | None) -> float | None:
    if value is None or not value.strip():
        return None
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"turn metric must be finite, got {value!r}")
    return parsed


def _validate_samples(samples: list[TiltSample]) -> None:
    for index, sample in enumerate(samples):
        if index and sample.timestamp_ms <= samples[index - 1].timestamp_ms:
            raise ValueError("sample timestamps must be strictly increasing")
        if sample.value is not None and not math.isfinite(sample.value):
            raise ValueError("sample values must be finite or None")


def _validate_config(config: TurnDetectionConfig) -> None:
    if not 0 <= config.edge_exclusion_fraction < 0.5:
        raise ValueError("edge_exclusion_fraction must be in [0, 0.5)")
    if config.confirmation_frames < 1:
        raise ValueError("confirmation_frames must be positive")
    if config.minimum_excursion_degrees < 0:
        raise ValueError("minimum_excursion_degrees must be non-negative")
    if not 0 < config.apex_range_fraction <= 1:
        raise ValueError("apex_range_fraction must be in (0, 1]")


def _rounded_time(value: float) -> float:
    return round(value, 3)
