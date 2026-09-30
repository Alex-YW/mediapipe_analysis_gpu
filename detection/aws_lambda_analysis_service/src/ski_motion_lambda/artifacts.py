from __future__ import annotations

import gzip
import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .geometry import ANGLE_NAMES
from .types import Point3D, VideoMetadata


MANIFEST_FORMAT = "ski-motion-overlay-manifest"
MANIFEST_SCHEMA_VERSION = 2


class OverlayManifestWriter:
    """Write the versioned frame contract consumed by the future renderer."""

    def __init__(
        self,
        path: Path,
        *,
        source: Mapping[str, object],
        metadata: VideoMetadata,
        processing_frame_rate: float,
        landmark_names: Sequence[str],
    ) -> None:
        if not landmark_names or len(set(landmark_names)) != len(landmark_names):
            raise ValueError("landmark_names must be non-empty and unique")
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        self._file = gzip.open(path, "wt", encoding="utf-8", newline="\n")
        self._digest = hashlib.sha256()
        self._landmark_names = tuple(landmark_names)
        self._frame_count = 0
        self._last_source_frame_index = -1
        self._first_timestamp_ms: int | None = None
        self._last_timestamp_ms: int | None = None
        self._closed = False

        display_width = (
            metadata.height if metadata.rotation in (90, 270) else metadata.width
        )
        display_height = (
            metadata.width if metadata.rotation in (90, 270) else metadata.height
        )
        self._write_checked(
            {
                "record_type": "header",
                "format": MANIFEST_FORMAT,
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "source": dict(source),
                "video": {
                    "encoded_width": metadata.width,
                    "encoded_height": metadata.height,
                    "display_width": display_width,
                    "display_height": display_height,
                    "rotation_degrees": metadata.rotation,
                    "source_frame_rate": metadata.fps,
                    "processing_frame_rate": processing_frame_rate,
                    "source_frame_count": metadata.frame_count,
                    "duration_seconds": metadata.duration_seconds,
                },
                "coordinate_system": {
                    "space": "normalized_display_frame",
                    "origin": "top_left",
                    "x_direction": "right",
                    "y_direction": "down",
                    "landmark_values": [
                        "x",
                        "y",
                        "z",
                        "visibility",
                        "presence",
                    ],
                },
                "landmark_order": list(self._landmark_names),
                "angle_order": list(ANGLE_NAMES),
            }
        )

    def write_frame(
        self,
        *,
        output_frame_index: int,
        source_frame_index: int,
        timestamp_ms: int,
        landmarks: Mapping[str, Point3D],
        angles: Mapping[str, float | None],
        invalid_frame: bool = False,
    ) -> None:
        self._ensure_open()
        if output_frame_index != self._frame_count:
            raise ValueError("output_frame_index must be contiguous from zero")
        if source_frame_index <= self._last_source_frame_index:
            raise ValueError("source_frame_index must be strictly increasing")
        if self._last_timestamp_ms is not None and timestamp_ms <= self._last_timestamp_ms:
            raise ValueError("timestamp_ms must be strictly increasing")

        packed_landmarks: list[list[float | None] | None] = []
        for name in self._landmark_names:
            point = landmarks.get(name)
            packed_landmarks.append(
                None
                if point is None
                else [
                    _finite_or_none(point.x),
                    _finite_or_none(point.y),
                    _finite_or_none(point.z),
                    _finite_or_none(point.visibility),
                    _finite_or_none(point.presence),
                ]
            )
        packed_angles = [_finite_or_none(angles.get(name)) for name in ANGLE_NAMES]
        self._write_checked(
            {
                "record_type": "frame",
                "output_frame_index": output_frame_index,
                "source_frame_index": source_frame_index,
                "timestamp_ms": timestamp_ms,
                "invalid_frame": invalid_frame,
                "landmarks": packed_landmarks,
                "angles": packed_angles,
            }
        )
        self._frame_count += 1
        self._last_source_frame_index = source_frame_index
        if self._first_timestamp_ms is None:
            self._first_timestamp_ms = timestamp_ms
        self._last_timestamp_ms = timestamp_ms

    def close(self) -> None:
        if self._closed:
            return
        footer = {
            "record_type": "footer",
            "frame_count": self._frame_count,
            "first_timestamp_ms": self._first_timestamp_ms,
            "last_timestamp_ms": self._last_timestamp_ms,
            "records_sha256": self._digest.hexdigest(),
        }
        self._file.write(_json_line(footer))
        self._file.close()
        self._closed = True

    def abort(self) -> None:
        if self._closed:
            return
        self._file.close()
        self._closed = True

    def _write_checked(self, record: Mapping[str, Any]) -> None:
        line = _json_line(record)
        self._file.write(line)
        self._digest.update(line.encode("utf-8"))

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("overlay manifest is closed")


def read_overlay_manifest(
    path: Path,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, Any]]:
    """Read and validate a complete manifest; intended to be reused by Service 2."""

    digest = hashlib.sha256()
    header: dict[str, Any] | None = None
    frames: list[dict[str, Any]] = []
    footer: dict[str, Any] | None = None
    with gzip.open(path, "rt", encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            record = json.loads(line)
            record_type = record.get("record_type")
            if footer is not None:
                raise ValueError("overlay manifest contains records after its footer")
            if record_type == "header":
                if line_number != 1 or header is not None:
                    raise ValueError("overlay manifest must begin with exactly one header")
                header = record
                digest.update(line.encode("utf-8"))
            elif record_type == "frame":
                if header is None:
                    raise ValueError("overlay manifest frame appears before header")
                frames.append(record)
                digest.update(line.encode("utf-8"))
            elif record_type == "footer":
                footer = record
            else:
                raise ValueError(f"unknown overlay manifest record at line {line_number}")

    if header is None or footer is None:
        raise ValueError("overlay manifest is incomplete")
    if header.get("format") != MANIFEST_FORMAT:
        raise ValueError("unsupported overlay manifest format")
    if header.get("schema_version") != MANIFEST_SCHEMA_VERSION:
        raise ValueError("unsupported overlay manifest schema version")
    if footer.get("frame_count") != len(frames):
        raise ValueError("overlay manifest frame count does not match footer")
    if footer.get("records_sha256") != digest.hexdigest():
        raise ValueError("overlay manifest checksum does not match")

    landmark_count = len(header.get("landmark_order", []))
    angle_count = len(header.get("angle_order", []))
    for expected_index, frame in enumerate(frames):
        if frame.get("output_frame_index") != expected_index:
            raise ValueError("overlay manifest output frame indices are not contiguous")
        if len(frame.get("landmarks", [])) != landmark_count:
            raise ValueError("overlay manifest landmark count does not match header")
        if len(frame.get("angles", [])) != angle_count:
            raise ValueError("overlay manifest angle count does not match header")
        invalid_frame = frame.get("invalid_frame", False)
        if not isinstance(invalid_frame, bool):
            raise ValueError("overlay manifest invalid_frame must be boolean")
    return header, frames, footer


def _json_line(record: Mapping[str, Any]) -> str:
    return json.dumps(
        record,
        ensure_ascii=True,
        allow_nan=False,
        separators=(",", ":"),
    ) + "\n"


def _finite_or_none(value: float | None) -> float | None:
    if value is None:
        return None
    converted = float(value)
    return converted if math.isfinite(converted) else None
