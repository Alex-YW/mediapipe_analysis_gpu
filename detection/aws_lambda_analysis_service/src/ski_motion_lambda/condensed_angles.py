from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

from .pose_quality import INVALID_FRAME_FIELD


_BODY_RELATIVE_ROTATION_FIELD = "shoulder_hip_body_relative_rotation_difference_degrees"
_MAX_BODY_RELATIVE_ROTATION_DEGREES = 60.0

SOURCE_FIELDS = (
    "frame_index",
    "timestamp_ms",
    INVALID_FRAME_FIELD,
    "outside_knee_side",
    "center_of_gravity_outside_knee_plane_distance",
    _BODY_RELATIVE_ROTATION_FIELD,
    "torso_leg_lateral_angulation",
    "left_wrist_torso_plane_distance_meters",
    "right_wrist_torso_plane_distance_meters",
    "lower_leg_asymmetry_deg",
)
TURN_FIELDS = ("turn_direction", "turn_stage", "turn_completeness")
OUTPUT_FIELDS = (*SOURCE_FIELDS, *TURN_FIELDS)

_SCALED_INTEGER_FIELDS = frozenset(
    {
        "left_wrist_torso_plane_distance_meters",
        "right_wrist_torso_plane_distance_meters",
        "center_of_gravity_outside_knee_plane_distance",
    }
)
_ONE_DECIMAL_FIELDS = frozenset(
    {
        "torso_leg_lateral_angulation",
        "lower_leg_asymmetry_deg",
    }
)


def write_condensed_angles_csv(
    angles_csv: Path,
    turns_json: Path,
    output_csv: Path,
) -> None:
    """Create a compact, turn-labelled view of the final angle data."""
    with angles_csv.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fieldnames = set(reader.fieldnames or ())
        rows = list(reader)
    required_source_fields = set(SOURCE_FIELDS).difference({INVALID_FRAME_FIELD})
    missing = required_source_fields.difference(fieldnames)
    if missing:
        raise ValueError(
            f"CSV is missing required columns: {', '.join(sorted(missing))}"
        )

    turns = _read_turns(turns_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            timestamp_ms = _number(row["timestamp_ms"], "timestamp_ms")
            direction, stage, completeness = _turn_label(timestamp_ms, turns)
            invalid_frame = _is_invalid_frame(row)
            source_values = {
                field: _format_value(field, row[field])
                for field in SOURCE_FIELDS
                if field != INVALID_FRAME_FIELD
            }
            if invalid_frame:
                source_values = {field: "" for field in source_values}
            writer.writerow(
                {
                    "frame_index": row["frame_index"],
                    "timestamp_ms": row["timestamp_ms"],
                    INVALID_FRAME_FIELD: "true" if invalid_frame else "false",
                    **{
                        field: value
                        for field, value in source_values.items()
                        if field not in {"frame_index", "timestamp_ms"}
                    },
                }
                | {
                    "turn_direction": direction,
                    "turn_stage": stage,
                    "turn_completeness": completeness,
                }
            )


def _read_turns(turns_json: Path) -> list[dict[str, Any]]:
    report = json.loads(turns_json.read_text(encoding="utf-8"))
    if not isinstance(report, dict) or not isinstance(report.get("turns"), list):
        raise ValueError("turns.json must contain a turns list")
    turns = report["turns"]
    if not all(isinstance(turn, dict) for turn in turns):
        raise ValueError("turns.json entries must be objects")
    return turns


def _turn_label(timestamp_ms: float, turns: list[dict[str, Any]]) -> tuple[str, str, str]:
    for turn in turns:
        begin = _number(turn.get("begin_time_ms"), "begin_time_ms")
        end = _number(turn.get("end_time_ms"), "end_time_ms")
        # Turn boundaries are half-open, matching the existing outside-knee
        # export and ensuring a shared boundary belongs to the following turn.
        if not begin <= timestamp_ms < end:
            continue
        direction = turn.get("direction")
        if not isinstance(direction, str) or not direction:
            raise ValueError("turns.json turn direction must be a non-empty string")
        begin_observed = turn.get("begin_observed", True)
        end_observed = turn.get("end_observed", True)
        if not isinstance(begin_observed, bool) or not isinstance(end_observed, bool):
            raise ValueError("turn boundary observation flags must be booleans")
        if begin_observed and end_observed:
            completeness = "complete"
        elif begin_observed:
            completeness = "end_missing"
        elif end_observed:
            completeness = "start_missing"
        else:
            completeness = "both_missing"
        if turn.get("apex_observed", True) is False:
            return direction, "", completeness
        apex_start = _number(turn.get("apex_start_time_ms"), "apex_start_time_ms")
        apex_end = _number(turn.get("apex_end_time_ms"), "apex_end_time_ms")
        if apex_start <= timestamp_ms <= apex_end:
            return direction, "apex", completeness
        return direction, "entry" if timestamp_ms < apex_start else "exit", completeness
    return "", "", ""


def _format_value(field: str, value: str) -> str:
    if value == "":
        return ""
    if field == _BODY_RELATIVE_ROTATION_FIELD:
        angle = _number(value, field)
        return "" if abs(angle) > _MAX_BODY_RELATIVE_ROTATION_DEGREES else f"{angle:.1f}"
    if field in _SCALED_INTEGER_FIELDS:
        return str(round(_number(value, field) * 100))
    if field in _ONE_DECIMAL_FIELDS:
        return f"{_number(value, field):.1f}"
    return value


def _number(value: object, field: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a finite number") from error
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number")
    return number


def _is_invalid_frame(row: dict[str, str]) -> bool:
    return row.get(INVALID_FRAME_FIELD, "").strip().lower() in {"1", "true"}
