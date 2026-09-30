"""Submit/check RunPod GPU tests without passing AWS credentials to workers."""

from __future__ import annotations

import http.client
import json
import os
import re
import uuid
from datetime import datetime, timezone
from typing import Any

from contract import (
    ARTIFACTS,
    EXECUTION_TIMEOUT_MS,
    JOB_TTL_MS,
    MAX_INPUT_BYTES,
    URL_EXPIRY_SECONDS,
)


_ID = re.compile(r"^[A-Za-z0-9_-]{8,100}$")
_RUN_ID = re.compile(r"^mpgpu-[0-9]{8}T[0-9]{6}Z-[a-f0-9]{8}$")
_HEX_SHA256 = re.compile(r"^[a-f0-9]{64}$")
_MAX_RESPONSE_BYTES = 1024 * 1024


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise ValueError(f"Missing Lambda environment variable: {name}")
    return value


def _prefix(name: str) -> str:
    value = _required_env(name).strip("/")
    if any(part in ("", ".", "..") for part in value.split("/")):
        raise ValueError(f"Invalid {name}")
    return value + "/"


def _source_key(value: object, prefix: str) -> str:
    if (
        not isinstance(value, str)
        or not value.startswith(prefix)
        or value.lower().rsplit(".", 1)[-1] not in {"mov", "mp4"}
        or any(part in ("", ".", "..") for part in value.split("/"))
    ):
        raise ValueError("input_key must be a .mov or .mp4 under INPUT_PREFIX")
    return value


def _uri(bucket: str, key: str) -> str:
    return f"s3://{bucket}/{key}"


def _runpod_request(method: str, path: str, api_key: str, body: dict | None = None) -> dict:
    connection = http.client.HTTPSConnection("api.runpod.ai", timeout=20)
    try:
        connection.request(
            method, path,
            body=json.dumps(body).encode("utf-8") if body is not None else None,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
        )
        response = connection.getresponse()
        raw = response.read(_MAX_RESPONSE_BYTES + 1)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise RuntimeError("RunPod response exceeds size limit")
        if not 200 <= response.status < 300:
            raise RuntimeError(f"RunPod API returned HTTP {response.status}")
        value = json.loads(raw)
        if not isinstance(value, dict):
            raise RuntimeError("RunPod API response was not an object")
        return value
    except (OSError, ValueError):
        # Error strings can include request details; signed URLs stay private.
        raise RuntimeError("RunPod API request failed") from None
    finally:
        connection.close()


def _api_key(secrets_client: Any) -> str:
    value = secrets_client.get_secret_value(SecretId=_required_env("RUNPOD_SECRET_ARN"))
    key = value.get("SecretString", "").strip()
    if not key or key.startswith("{"):
        raise ValueError("RunPod secret must be a plaintext API key")
    return key


def _submit(event: dict, *, s3: Any, bucket: str, input_prefix: str,
            output_prefix: str, api_key: str, endpoint_id: str,
            runpod_request: Any) -> dict:
    key = _source_key(event.get("input_key"), input_prefix)
    requested_version = event.get("input_version_id")
    if requested_version is not None and (
        not isinstance(requested_version, str) or not requested_version
    ):
        raise ValueError("input_version_id must be a nonempty string")
    head_request = {"Bucket": bucket, "Key": key}
    if requested_version:
        head_request["VersionId"] = requested_version
    head = s3.head_object(**head_request)
    version_id = head.get("VersionId")
    size = head.get("ContentLength")
    if not isinstance(version_id, str) or not version_id or version_id == "null":
        raise ValueError("S3 input bucket must have versioning enabled")
    if type(size) is not int or not 0 < size <= MAX_INPUT_BYTES:
        raise ValueError("input video is empty or exceeds the 1 GiB test limit")
    frame_rate = event.get("processing_frame_rate", 30)
    if type(frame_rate) not in (int, float) or not 1 <= frame_rate <= 60:
        raise ValueError("processing_frame_rate must be 1 through 60")

    run_id = (
        "mpgpu-" + datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-" + uuid.uuid4().hex[:8]
    )
    run_prefix = output_prefix + run_id + "/"
    put_urls = {}
    output_uris = {}
    for name, (filename, content_type) in ARTIFACTS.items():
        output_key = run_prefix + filename
        put_urls[name] = s3.generate_presigned_url(
            "put_object",
            Params={"Bucket": bucket, "Key": output_key, "ContentType": content_type},
            ExpiresIn=URL_EXPIRY_SECONDS,
            HttpMethod="PUT",
        )
        output_uris[name] = _uri(bucket, output_key)
    get_url = s3.generate_presigned_url(
        "get_object",
        Params={"Bucket": bucket, "Key": key, "VersionId": version_id},
        ExpiresIn=URL_EXPIRY_SECONDS,
        HttpMethod="GET",
    )
    source = {
        "bucket": bucket,
        "key": key,
        "version_id": version_id,
        "etag": str(head.get("ETag", "")).strip('"'),
        "size_bytes": size,
    }
    request = {
        "input": {
            "run_id": run_id,
            "video_get_url": get_url,
            "output_put_urls": put_urls,
            "output_s3_uris": output_uris,
            "source": source,
            "processing_frame_rate": frame_rate,
        },
        "policy": {
            "executionTimeout": EXECUTION_TIMEOUT_MS,
            "ttl": JOB_TTL_MS,
        },
    }
    result = runpod_request("POST", f"/v2/{endpoint_id}/run", api_key, request)
    job_id = result.get("id")
    if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
        raise RuntimeError("RunPod did not return a valid job ID")
    return {
        "run_id": run_id,
        "job_id": job_id,
        "status": result.get("status"),
        "source": _uri(bucket, key),
        "source_version_id": version_id,
        "output_prefix": _uri(bucket, run_prefix),
        "status_event": {"action": "status", "run_id": run_id, "job_id": job_id},
    }


def _status(event: dict, *, s3: Any, bucket: str, input_prefix: str,
            output_prefix: str, api_key: str, endpoint_id: str,
            runpod_request: Any) -> dict:
    run_id = event.get("run_id")
    job_id = event.get("job_id")
    if not isinstance(run_id, str) or not _RUN_ID.fullmatch(run_id):
        raise ValueError("status requires a valid run_id")
    if not isinstance(job_id, str) or not _ID.fullmatch(job_id):
        raise ValueError("status requires a valid job_id")
    result = runpod_request("GET", f"/v2/{endpoint_id}/status/{job_id}", api_key)
    status = result.get("status")
    response = {
        "run_id": run_id, "job_id": job_id, "status": status,
        "delay_time_ms": result.get("delayTime"),
        "execution_time_ms": result.get("executionTime"),
    }
    if status == "COMPLETED":
        output = result.get("output")
        if not isinstance(output, dict) or output.get("run_id") != run_id:
            raise RuntimeError("RunPod output does not match requested run")
        listed = output.get("artifacts")
        if not isinstance(listed, dict):
            raise RuntimeError("RunPod returned no artifact list")
        run_prefix = output_prefix + run_id + "/"
        artifacts = {}
        for name, (filename, _) in ARTIFACTS.items():
            key = run_prefix + filename
            uri = _uri(bucket, key)
            if listed.get(name) != uri:
                raise RuntimeError(f"RunPod artifact location mismatch: {name}")
            head = s3.head_object(Bucket=bucket, Key=key)
            if int(head.get("ContentLength", 0)) <= 0:
                raise RuntimeError(f"empty S3 artifact: {name}")
            artifacts[name] = {
                "s3_uri": uri,
                "bytes": int(head["ContentLength"]),
                "version_id": head.get("VersionId"),
            }
        report = json.loads(
            s3.get_object(Bucket=bucket, Key=run_prefix + "performance.json")["Body"].read()
        )
        if (
            report.get("run_id") != run_id
            or report.get("gpu_delegate_initialized") is not True
            or not report.get("gpu", {}).get("name")
            or not isinstance(report.get("source"), dict)
            or not str(report["source"].get("key", "")).startswith(input_prefix)
        ):
            raise RuntimeError("GPU performance/source provenance is incomplete")
        hashes = report.get("artifacts_sha256")
        if not isinstance(hashes, dict) or any(
            not isinstance(hashes.get(name), str)
            or not _HEX_SHA256.fullmatch(hashes[name])
            for name in ARTIFACTS if name != "performance"
        ):
            raise RuntimeError("artifact checksum report is incomplete")
        response["artifacts"] = artifacts
        response["gpu"] = report["gpu"]
        response["frames"] = output.get("frames")
        response["model_api"] = output.get("model_api")
    elif status in {"FAILED", "TIMED_OUT", "CANCELLED"}:
        response["note"] = "Inspect RunPod job logs; signed URLs are not returned."
    return response


def process(event: dict, *, s3: Any, secrets_client: Any,
            runpod_request: Any = _runpod_request) -> dict:
    if not isinstance(event, dict):
        raise ValueError("event must be a JSON object")
    endpoint_id = _required_env("RUNPOD_ENDPOINT_ID")
    if not _ID.fullmatch(endpoint_id):
        raise ValueError("invalid RUNPOD_ENDPOINT_ID")
    bucket = _required_env("S3_BUCKET")
    input_prefix = _prefix("INPUT_PREFIX")
    output_prefix = _prefix("OUTPUT_PREFIX")
    api_key = _api_key(secrets_client)
    arguments = {
        "s3": s3,
        "bucket": bucket,
        "input_prefix": input_prefix,
        "output_prefix": output_prefix,
        "api_key": api_key,
        "endpoint_id": endpoint_id,
        "runpod_request": runpod_request,
    }
    if event.get("action") == "submit":
        return _submit(event, **arguments)
    if event.get("action") == "status":
        return _status(event, **arguments)
    raise ValueError("action must be submit or status")


def lambda_handler(event: dict, context: Any) -> dict:
    del context
    import boto3
    from botocore.config import Config

    s3 = boto3.client(
        "s3", region_name=_required_env("S3_REGION"),
        config=Config(signature_version="s3v4"),
    )
    return process(event, s3=s3, secrets_client=boto3.client("secretsmanager"))
