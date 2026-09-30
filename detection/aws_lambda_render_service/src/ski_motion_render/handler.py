from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import unquote_plus

from .artifacts import MANIFEST_SCHEMA_VERSION, read_overlay_manifest
from .pipeline import (
    render_video,
    validate_decoder_threads,
    validate_encoder_settings,
    validate_pipeline_queue_size,
)


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)
MANIFEST_FILENAME = "overlay-manifest.jsonl.gz"


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Render one Service 1 manifest supplied directly or by an S3 event."""

    invocation_started = time.perf_counter()
    request = _normalize_event(event)
    manifest_bucket = request["manifest_bucket"]
    manifest_key = request["manifest_key"].lstrip("/")
    manifest_version_id = request.get("manifest_version_id")
    encoder_preset = request.get(
        "encoder_preset",
        os.environ.get("ENCODER_PRESET", "veryfast"),
    )
    raw_crf = request.get("crf", _environment_int("CRF", 20))
    raw_decoder_threads = request.get(
        "decoder_threads",
        _environment_int("DECODER_THREADS", 0),
    )
    raw_queue_size = request.get(
        "pipeline_queue_size",
        _environment_int("PIPELINE_QUEUE_SIZE", 4),
    )
    crf, encoder_preset = validate_encoder_settings(raw_crf, encoder_preset)
    decoder_threads = validate_decoder_threads(raw_decoder_threads)
    pipeline_queue_size = validate_pipeline_queue_size(raw_queue_size)

    if not manifest_key.endswith(MANIFEST_FILENAME):
        raise ValueError(f"manifest_key must end with {MANIFEST_FILENAME}")
    request_id = str(getattr(context, "aws_request_id", "local-test"))
    job_dir = Path(tempfile.mkdtemp(prefix="ski-render-", dir="/tmp"))
    manifest_path = job_dir / MANIFEST_FILENAME
    input_path: Path | None = None
    output_path = job_dir / "annotated.mp4"
    s3 = _s3_client()
    timings: dict[str, float] = {}

    try:
        stage = time.perf_counter()
        manifest_head_args: dict[str, Any] = {
            "Bucket": manifest_bucket,
            "Key": manifest_key,
        }
        if manifest_version_id:
            manifest_head_args["VersionId"] = manifest_version_id
        manifest_head = s3.head_object(**manifest_head_args)
        resolved_manifest_version = manifest_version_id or manifest_head.get("VersionId")
        timings["s3_head_manifest"] = time.perf_counter() - stage

        stage = time.perf_counter()
        _download(
            s3,
            manifest_bucket,
            manifest_key,
            manifest_path,
            resolved_manifest_version,
        )
        timings["s3_download_manifest"] = time.perf_counter() - stage

        stage = time.perf_counter()
        manifest = read_overlay_manifest(manifest_path)
        timings["manifest_validation"] = time.perf_counter() - stage
        source = manifest.header["source"]
        assert isinstance(source, dict)
        source_bucket = _mapping_required_string(source, "bucket")
        source_key = _mapping_required_string(source, "key").lstrip("/")
        source_version_id = _mapping_optional_string(source, "version_id")
        suffix = Path(source_key).suffix.lower()
        if suffix not in {".mov", ".mp4"}:
            raise ValueError("manifest source key must end with .mov or .mp4")
        input_path = job_dir / f"input{suffix}"

        stage = time.perf_counter()
        source_head_args: dict[str, Any] = {
            "Bucket": source_bucket,
            "Key": source_key,
        }
        if source_version_id:
            source_head_args["VersionId"] = source_version_id
        source_head = s3.head_object(**source_head_args)
        _validate_source_identity(source, source_head)
        timings["s3_head_source"] = time.perf_counter() - stage

        stage = time.perf_counter()
        _download(
            s3,
            source_bucket,
            source_key,
            input_path,
            source_version_id,
        )
        timings["s3_download_source"] = time.perf_counter() - stage

        output_bucket = request.get("output_bucket") or os.environ.get(
            "OUTPUT_BUCKET"
        ) or manifest_bucket
        if not isinstance(output_bucket, str) or not output_bucket.strip():
            raise ValueError("output_bucket must be a non-empty string")
        output_bucket = output_bucket.strip()
        default_output_key = manifest_key[: -len(MANIFEST_FILENAME)] + "annotated.mp4"
        output_key = request.get("output_key") or default_output_key
        if not isinstance(output_key, str) or not output_key.strip():
            raise ValueError("output_key must be a non-empty string")
        output_key = output_key.strip().lstrip("/")
        if not output_key.lower().endswith(".mp4"):
            raise ValueError("output_key must end with .mp4")

        result = render_video(
            input_path,
            output_path,
            manifest,
            crf=crf,
            encoder_preset=encoder_preset,
            pipeline_queue_size=pipeline_queue_size,
            decoder_threads=decoder_threads,
        )

        stage = time.perf_counter()
        metadata = {
            "manifest-schema-version": str(MANIFEST_SCHEMA_VERSION),
            "manifest-etag": str(manifest_head.get("ETag", "")).strip('"')[:1024],
            "source-etag": str(source.get("etag", ""))[:1024],
        }
        s3.upload_file(
            str(result["annotated_video"]),
            output_bucket,
            output_key,
            ExtraArgs={
                "ContentType": "video/mp4",
                "Metadata": metadata,
            },
        )
        timings["s3_upload_annotated_video"] = time.perf_counter() - stage
        timings["total_invocation"] = time.perf_counter() - invocation_started

        response = {
            "status": "render_complete",
            "frames": result["frames"],
            "processing_fps": round(float(result["processing_fps"]), 3),
            "configuration": {
                "encoder_preset": encoder_preset,
                "crf": crf,
                "pipeline_queue_size": pipeline_queue_size,
                "decoder_threads_requested": result["decoder_threads_requested"],
                "decoder_threads_actual": result["decoder_threads_actual"],
                "output_frame_rate": result["output_frame_rate"],
                "source_frames_traversed": result["source_frames_traversed"],
                "selected_frames_retrieved": result["selected_frames_retrieved"],
            },
            "manifest": {
                "bucket": manifest_bucket,
                "key": manifest_key,
                "version_id": resolved_manifest_version,
                "schema_version": MANIFEST_SCHEMA_VERSION,
            },
            "source": source,
            "outputs": {
                "annotated_video": {
                    "bucket": output_bucket,
                    "key": output_key,
                }
            },
            "timings_seconds": {
                **result["timings"],
                **{name: round(value, 6) for name, value in timings.items()},
            },
        }
        LOGGER.info(json.dumps({"event": "render_complete", **response}))
        return response
    except Exception:
        LOGGER.exception(
            "Rendering failed for s3://%s/%s (request_id=%s)",
            manifest_bucket,
            manifest_key,
            request_id,
        )
        raise
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def _normalize_event(event: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(event, dict):
        raise ValueError("Lambda event must be a JSON object")
    records = event.get("Records")
    if records is not None:
        if not isinstance(records, list) or len(records) != 1:
            raise ValueError("S3 events must contain exactly one record")
        record = records[0]
        try:
            if record.get("eventSource") != "aws:s3":
                raise ValueError("event record is not from S3")
            bucket = record["s3"]["bucket"]["name"]
            object_data = record["s3"]["object"]
            key = unquote_plus(object_data["key"])
        except (KeyError, TypeError, AttributeError) as error:
            raise ValueError("S3 event does not contain a valid object location") from error
        normalized: dict[str, Any] = {
            "manifest_bucket": bucket,
            "manifest_key": key,
        }
        if object_data.get("versionId"):
            normalized["manifest_version_id"] = object_data["versionId"]
        return normalized

    if "manifest_bucket" in event or "manifest_key" in event:
        normalized = dict(event)
        normalized["manifest_bucket"] = _required_string(event, "manifest_bucket")
        normalized["manifest_key"] = _required_string(event, "manifest_key")
        return normalized

    outputs = event.get("outputs")
    overlay_manifest = outputs.get("overlay_manifest") if isinstance(outputs, dict) else None
    if isinstance(overlay_manifest, dict):
        normalized = {
            key: event[key]
            for key in (
                "output_bucket",
                "output_key",
                "encoder_preset",
                "crf",
                "decoder_threads",
                "pipeline_queue_size",
            )
            if key in event
        }
        normalized["manifest_bucket"] = _mapping_required_string(
            overlay_manifest,
            "bucket",
        )
        normalized["manifest_key"] = _mapping_required_string(
            overlay_manifest,
            "key",
        )
        if overlay_manifest.get("version_id"):
            normalized["manifest_version_id"] = overlay_manifest["version_id"]
        return normalized
    raise ValueError(
        "Event must contain manifest_bucket/manifest_key, a Service 1 outputs object, or one S3 record"
    )


def _validate_source_identity(source: dict[str, Any], head: dict[str, Any]) -> None:
    expected_etag = source.get("etag")
    actual_etag = str(head.get("ETag", "")).strip('"')
    if expected_etag and str(expected_etag).strip('"') != actual_etag:
        raise ValueError("source S3 object ETag does not match overlay manifest")
    expected_size = source.get("size_bytes")
    if expected_size is not None:
        if (
            isinstance(expected_size, bool)
            or not isinstance(expected_size, int)
            or expected_size < 0
        ):
            raise ValueError("manifest source size_bytes is invalid")
        if int(head.get("ContentLength", -1)) != expected_size:
            raise ValueError("source S3 object size does not match overlay manifest")
    expected_version = source.get("version_id")
    actual_version = head.get("VersionId")
    if expected_version and actual_version and str(expected_version) != str(actual_version):
        raise ValueError("source S3 object version does not match overlay manifest")


def _download(
    s3: Any,
    bucket: str,
    key: str,
    destination: Path,
    version_id: object | None,
) -> None:
    if version_id:
        s3.download_file(
            bucket,
            key,
            str(destination),
            ExtraArgs={"VersionId": str(version_id)},
        )
    else:
        s3.download_file(bucket, key, str(destination))


def _required_string(event: dict[str, Any], field: str) -> str:
    value = event.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Event field `{field}` must be a non-empty string")
    return value.strip()


def _mapping_required_string(mapping: dict[str, Any], field: str) -> str:
    value = mapping.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field `{field}` must be a non-empty string")
    return value.strip()


def _mapping_optional_string(mapping: dict[str, Any], field: str) -> str | None:
    value = mapping.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Manifest field `{field}` must be a non-empty string")
    return value.strip()


def _environment_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError as error:
        raise ValueError(f"Environment variable {name} must be an integer") from error


def _s3_client() -> Any:
    import boto3

    return boto3.client("s3")
