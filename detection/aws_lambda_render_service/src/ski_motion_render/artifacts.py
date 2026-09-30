from __future__ import annotations

import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .types import ManifestFrame, Point3D, RenderManifest


MANIFEST_FORMAT = "ski-motion-overlay-manifest"
MANIFEST_SCHEMA_VERSION = 2
_SUPPORTED_MANIFEST_SCHEMA_VERSIONS = frozenset((1, MANIFEST_SCHEMA_VERSION))


def read_overlay_manifest(path: Path) -> RenderManifest:
    """Read and strictly validate the Service 1 rendering contract."""

    digest = hashlib.sha256()
    header: dict[str, Any] | None = None
    raw_frames: list[dict[str, Any]] = []
    footer: dict[str, Any] | None = None
    try:
        with gzip.open(path, "rt", encoding="utf-8") as manifest:
            for line_number, line in enumerate(manifest, start=1):
                record = json.loads(line)
                if not isinstance(record, dict):
                    raise ValueError(
                        f"overlay manifest record {line_number} must be an object"
                    )
                record_type = record.get("record_type")
                if footer is not None:
                    raise ValueError("overlay manifest contains records after its footer")
                if record_type == "header":
                    if line_number != 1 or header is not None:
                        raise ValueError(
                            "overlay manifest must begin with exactly one header"
                        )
                    header = record
                    digest.update(line.encode("utf-8"))
                elif record_type == "frame":
                    if header is None:
                        raise ValueError("overlay manifest frame appears before header")
                    raw_frames.append(record)
                    digest.update(line.encode("utf-8"))
                elif record_type == "footer":
                    footer = record
                else:
                    raise ValueError(
                        f"unknown overlay manifest record at line {line_number}"
                    )
    except (OSError, EOFError, UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("overlay manifest is not valid gzip JSON Lines") from error

    if header is None or footer is None:
        raise ValueError("overlay manifest is incomplete")
    if header.get("format") != MANIFEST_FORMAT:
        raise ValueError("unsupported overlay manifest format")
    if header.get("schema_version") not in _SUPPORTED_MANIFEST_SCHEMA_VERSIONS:
        raise ValueError("unsupported overlay manifest schema version")
    if footer.get("frame_count") != len(raw_frames):
        raise ValueError("overlay manifest frame count does not match footer")
    if footer.get("records_sha256") != digest.hexdigest():
        raise ValueError("overlay manifest checksum does not match")
    if not raw_frames:
        raise ValueError("overlay manifest contains no frames")

    source = _required_mapping(header, "source")
    _required_string(source, "bucket")
    _required_string(source, "key")
    video = _required_mapping(header, "video")
    for field in ("encoded_width", "encoded_height", "display_width", "display_height"):
        _required_positive_int(video, field)
    _required_positive_number(video, "source_frame_rate")
    _required_positive_number(video, "processing_frame_rate")
    rotation = video.get("rotation_degrees")
    if isinstance(rotation, bool) or rotation not in (0, 90, 180, 270):
        raise ValueError("manifest video.rotation_degrees is unsupported")

    landmark_order = _required_unique_strings(header, "landmark_order")
    angle_order = _required_unique_strings(header, "angle_order")
    coordinate_system = _required_mapping(header, "coordinate_system")
    if coordinate_system.get("space") != "normalized_display_frame":
        raise ValueError("unsupported overlay manifest coordinate system")

    frames: list[ManifestFrame] = []
    previous_source_index = -1
    previous_timestamp = -1
    for expected_index, record in enumerate(raw_frames):
        if record.get("output_frame_index") != expected_index:
            raise ValueError("overlay manifest output frame indices are not contiguous")
        source_index = record.get("source_frame_index")
        timestamp_ms = record.get("timestamp_ms")
        if (
            isinstance(source_index, bool)
            or not isinstance(source_index, int)
            or source_index <= previous_source_index
        ):
            raise ValueError("manifest source frame indices must be strictly increasing")
        if (
            isinstance(timestamp_ms, bool)
            or not isinstance(timestamp_ms, int)
            or timestamp_ms <= previous_timestamp
        ):
            raise ValueError("manifest timestamps must be strictly increasing")
        packed_landmarks = record.get("landmarks")
        packed_angles = record.get("angles")
        invalid_frame = record.get("invalid_frame", False)
        if not isinstance(invalid_frame, bool):
            raise ValueError("manifest invalid_frame must be boolean")
        if not isinstance(packed_landmarks, list) or len(packed_landmarks) != len(
            landmark_order
        ):
            raise ValueError("overlay manifest landmark count does not match header")
        if not isinstance(packed_angles, list) or len(packed_angles) != len(angle_order):
            raise ValueError("overlay manifest angle count does not match header")

        landmarks: dict[str, Point3D] = {}
        for name, packed in zip(landmark_order, packed_landmarks, strict=True):
            if packed is None:
                continue
            if not isinstance(packed, list) or len(packed) != 5:
                raise ValueError(f"manifest landmark `{name}` has an invalid shape")
            values = [_optional_finite_number(value) for value in packed]
            if values[0] is None or values[1] is None or values[2] is None:
                continue
            landmarks[name] = Point3D(
                x=values[0],
                y=values[1],
                z=values[2],
                visibility=values[3] if values[3] is not None else 0.0,
                presence=values[4] if values[4] is not None else 0.0,
            )
        angles = {
            name: _optional_finite_number(value)
            for name, value in zip(angle_order, packed_angles, strict=True)
        }
        frames.append(
            ManifestFrame(
                output_frame_index=expected_index,
                source_frame_index=source_index,
                timestamp_ms=timestamp_ms,
                landmarks=landmarks,
                angles=angles,
                invalid_frame=invalid_frame,
            )
        )
        previous_source_index = source_index
        previous_timestamp = timestamp_ms

    if footer.get("first_timestamp_ms") != frames[0].timestamp_ms:
        raise ValueError("manifest first timestamp does not match footer")
    if footer.get("last_timestamp_ms") != frames[-1].timestamp_ms:
        raise ValueError("manifest last timestamp does not match footer")
    return RenderManifest(header=header, frames=tuple(frames), footer=footer)


def _required_mapping(parent: dict[str, Any], field: str) -> dict[str, Any]:
    value = parent.get(field)
    if not isinstance(value, dict):
        raise ValueError(f"manifest `{field}` must be an object")
    return value


def _required_string(parent: dict[str, Any], field: str) -> str:
    value = parent.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"manifest `{field}` must be a non-empty string")
    return value.strip()


def _required_positive_int(parent: dict[str, Any], field: str) -> int:
    value = parent.get(field)
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"manifest video.{field} must be a positive integer")
    return value


def _required_positive_number(parent: dict[str, Any], field: str) -> float:
    value = parent.get(field)
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        or float(value) <= 0
    ):
        raise ValueError(f"manifest video.{field} must be a positive number")
    return float(value)


def _required_unique_strings(parent: dict[str, Any], field: str) -> tuple[str, ...]:
    value = parent.get(field)
    if (
        not isinstance(value, list)
        or not value
        or any(not isinstance(item, str) or not item for item in value)
        or len(value) != len(set(value))
    ):
        raise ValueError(f"manifest `{field}` must contain unique strings")
    return tuple(value)


def _optional_finite_number(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("manifest numeric value must be finite or null")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError("manifest numeric value must be finite or null")
    return converted
