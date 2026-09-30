from __future__ import annotations

import json
import logging
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any

from .pipeline import (
    analyze_video,
    validate_decoder_threads,
    validate_pipeline_queue_size,
    validate_processing_frame_rate,
)


LOGGER = logging.getLogger(__name__)
LOGGER.setLevel(logging.INFO)


def lambda_handler(event: dict[str, Any], context: Any) -> dict[str, Any]:
    """Publish angle, turn, condensed-data, and renderer artifacts for one video."""

    invocation_started = time.perf_counter()
    input_bucket = _required_string(event, "bucket")
    input_key = _required_string(event, "input_key").lstrip("/")
    output_bucket = _optional_string(event, "output_bucket") or input_bucket
    input_version_id = _optional_string(event, "input_version_id")
    request_id = str(getattr(context, "aws_request_id", "local-test"))
    output_prefix = str(
        event.get("output_prefix") or f"results/{request_id}"
    ).strip("/")
    if not output_prefix:
        raise ValueError("output_prefix must not be empty")
    processing_frame_rate = validate_processing_frame_rate(
        event.get("processing_frame_rate", 30)
    )
    pipeline_queue_size = validate_pipeline_queue_size(
        event.get("pipeline_queue_size", 4)
    )
    decoder_threads = validate_decoder_threads(event.get("decoder_threads", 0))

    suffix = Path(input_key).suffix.lower()
    if suffix not in {".mov", ".mp4"}:
        raise ValueError("input_key must end with .mov or .mp4")

    job_dir = Path(tempfile.mkdtemp(prefix="ski-analysis-", dir="/tmp"))
    input_path = job_dir / f"input{suffix}"
    output_dir = job_dir / "output"
    angles_key = f"{output_prefix}/angles.csv"
    turns_key = f"{output_prefix}/turns.json"
    condensed_angles_key = f"{output_prefix}/condensed_angles.csv"
    manifest_key = f"{output_prefix}/overlay-manifest.jsonl.gz"
    s3 = _s3_client()
    handler_timings: dict[str, float] = {}

    try:
        stage = time.perf_counter()
        head_args: dict[str, Any] = {"Bucket": input_bucket, "Key": input_key}
        if input_version_id:
            head_args["VersionId"] = input_version_id
        head = s3.head_object(**head_args)
        resolved_version_id = input_version_id or head.get("VersionId")
        handler_timings["s3_head_input"] = time.perf_counter() - stage

        stage = time.perf_counter()
        if resolved_version_id:
            s3.download_file(
                input_bucket,
                input_key,
                str(input_path),
                ExtraArgs={"VersionId": resolved_version_id},
            )
        else:
            s3.download_file(input_bucket, input_key, str(input_path))
        handler_timings["s3_download"] = time.perf_counter() - stage

        source_identity: dict[str, object] = {
            "bucket": input_bucket,
            "key": input_key,
            "etag": str(head.get("ETag", "")).strip('"'),
            "size_bytes": int(head.get("ContentLength", input_path.stat().st_size)),
        }
        if resolved_version_id:
            source_identity["version_id"] = str(resolved_version_id)

        result = analyze_video(
            input_path,
            output_dir,
            source=source_identity,
            processing_frame_rate=processing_frame_rate,
            pipeline_queue_size=pipeline_queue_size,
            decoder_threads=decoder_threads,
        )

        # Angles are deliberately published before the lower-priority rendering
        # contract so downstream analysis can begin as early as possible.
        stage = time.perf_counter()
        s3.upload_file(
            str(result["angles_csv"]),
            output_bucket,
            angles_key,
            ExtraArgs={"ContentType": "text/csv"},
        )
        handler_timings["s3_upload_angles_csv"] = time.perf_counter() - stage
        handler_timings["angles_ready"] = time.perf_counter() - invocation_started

        stage = time.perf_counter()
        s3.upload_file(
            str(result["turns_json"]),
            output_bucket,
            turns_key,
            ExtraArgs={"ContentType": "application/json"},
        )
        handler_timings["s3_upload_turns_json"] = time.perf_counter() - stage
        handler_timings["turns_ready"] = time.perf_counter() - invocation_started

        stage = time.perf_counter()
        s3.upload_file(
            str(result["condensed_angles_csv"]),
            output_bucket,
            condensed_angles_key,
            ExtraArgs={"ContentType": "text/csv"},
        )
        handler_timings["s3_upload_condensed_angles_csv"] = (
            time.perf_counter() - stage
        )
        handler_timings["condensed_angles_ready"] = (
            time.perf_counter() - invocation_started
        )

        stage = time.perf_counter()
        s3.upload_file(
            str(result["overlay_manifest"]),
            output_bucket,
            manifest_key,
            ExtraArgs={
                "ContentType": "application/x-ndjson",
                "ContentEncoding": "gzip",
                "Metadata": {"schema-version": "1"},
            },
        )
        handler_timings["s3_upload_overlay_manifest"] = time.perf_counter() - stage
        handler_timings["render_artifacts_ready"] = (
            time.perf_counter() - invocation_started
        )
        handler_timings["total_invocation"] = time.perf_counter() - invocation_started

        response = {
            "status": "analysis_complete",
            "frames": result["frames"],
            "processing_fps": round(float(result["processing_fps"]), 3),
            "configuration": {
                "requested_processing_frame_rate": processing_frame_rate,
                "effective_processing_frame_rate": round(
                    float(result["processing_frame_rate"]),
                    3,
                ),
                "source_frame_rate": round(float(result["source_frame_rate"]), 3),
                "pipeline_queue_size": pipeline_queue_size,
                "decoder_threads_requested": result["decoder_threads_requested"],
                "decoder_threads_actual": result["decoder_threads_actual"],
                "rotated_frames": result["rotated_frames"],
            },
            "source": source_identity,
            "pose_diagnostics": result["pose_diagnostics"],
            "outputs": {
                "angles_csv": {"bucket": output_bucket, "key": angles_key},
                "turns_json": {
                    "bucket": output_bucket,
                    "key": turns_key,
                    "schema_version": 1,
                },
                "condensed_angles_csv": {
                    "bucket": output_bucket,
                    "key": condensed_angles_key,
                },
                "overlay_manifest": {
                    "bucket": output_bucket,
                    "key": manifest_key,
                    "schema_version": 1,
                },
            },
            "timings_seconds": {
                **result["timings"],
                **{
                    name: round(value, 6)
                    for name, value in handler_timings.items()
                },
            },
        }
        LOGGER.info(json.dumps({"event": "analysis_complete", **response}))
        return response
    except Exception:
        LOGGER.exception(
            "Analysis failed for s3://%s/%s (request_id=%s)",
            input_bucket,
            input_key,
            request_id,
        )
        raise
    finally:
        shutil.rmtree(job_dir, ignore_errors=True)


def _required_string(event: dict[str, Any], field: str) -> str:
    value = event.get(field)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Event field `{field}` must be a non-empty string")
    return value.strip()


def _optional_string(event: dict[str, Any], field: str) -> str | None:
    value = event.get(field)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"Event field `{field}` must be a non-empty string")
    return value.strip()


def _s3_client() -> Any:
    import boto3

    return boto3.client("s3")
