"""Interval orchestration around unchanged ARCS smoothing/turn definitions."""
from __future__ import annotations

import csv
import json
import tempfile
from pathlib import Path

from .signal_smoothing import write_smoothed_rotation_column
from .turn_detection import (build_turn_report, write_turn_report,
                             write_outside_knee_plane_columns)


def validate_intervals(intervals):
    if not isinstance(intervals, list) or not intervals or len(intervals) > 1000:
        raise ValueError("approved analysis intervals must be nonempty")
    end = -1
    ids = set()
    for interval in intervals:
        if not isinstance(interval, dict):
            raise ValueError("invalid analysis interval")
        start, stop = interval.get("start_ms"), interval.get("end_ms")
        name = interval.get("interval_id")
        if (type(start) is not int or type(stop) is not int or not 0 <= start < stop
                or start < end or not isinstance(name, str) or not name or name in ids):
            raise ValueError("analysis intervals must be ordered, disjoint, and uniquely identified")
        end = stop
        ids.add(name)
    return intervals


def interval_at(intervals, timestamp_ms):
    return next((i for i in intervals if i["start_ms"] <= timestamp_ms < i["end_ms"]), None)


def postprocess_intervals(angles_path: Path, turns_path: Path, intervals, sample_rate):
    """Never let offline interpolation/smoothing connect separate intervals."""
    with angles_path.open(newline="", encoding="utf-8") as source:
        reader = csv.DictReader(source)
        fields, rows = list(reader.fieldnames), list(reader)
    all_turns = []
    report = build_turn_report([])
    with tempfile.TemporaryDirectory(prefix="arcs-interval-metrics-") as temp:
        for interval in intervals:
            selected = [row for row in rows if interval["start_ms"] <= int(row["timestamp_ms"]) < interval["end_ms"]]
            if not selected:
                continue
            path = Path(temp) / "angles.csv"
            with path.open("w", newline="", encoding="utf-8") as target:
                writer = csv.DictWriter(target, fieldnames=fields)
                writer.writeheader()
                writer.writerows(selected)
            write_smoothed_rotation_column(path, sampling_rate_hz=sample_rate)
            part = write_turn_report(path, Path(temp) / "turns.json")
            for turn in part["turns"]:
                # EOF estimation uses the sample spacing; cap it at the actual
                # classifier boundary, without claiming an observed turn end.
                if turn["end_time_ms"] > interval["end_ms"]:
                    turn["end_time_ms"] = interval["end_ms"]
                    turn["end_observed"] = False
                turn["analysis_interval_id"] = interval["interval_id"]
                turn["turn_number"] = len(all_turns) + 1
                all_turns.append(turn)
            write_outside_knee_plane_columns(path, part)
            with path.open(newline="", encoding="utf-8") as completed:
                updated = list(csv.DictReader(completed))
                if updated:
                    fields = list(updated[0])
            by_index = {row["frame_index"]: row for row in updated}
            for row in rows:
                if row["frame_index"] in by_index:
                    row.update(by_index[row["frame_index"]])
    report["turns"] = all_turns
    report["analysis_intervals"] = intervals
    turns_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    with angles_path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return report
