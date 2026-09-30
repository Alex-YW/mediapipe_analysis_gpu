from __future__ import annotations

import io
import json

import pytest

from aws_submitter.lambda_function import process
from contract import ARTIFACTS


class FakeS3:
    def __init__(self) -> None:
        self.signed: list[tuple[str, dict]] = []
        self.report: dict = {}

    def head_object(self, **kwargs):
        if kwargs["Key"].startswith("inputs/"):
            return {"VersionId": "source-v1", "ContentLength": 1000, "ETag": '"etag"'}
        return {"VersionId": "out-v1", "ContentLength": 123}

    def generate_presigned_url(self, operation, *, Params, ExpiresIn, HttpMethod):
        self.signed.append((operation, Params))
        return f"https://bucket.s3.ap-southeast-1.amazonaws.com/{Params['Key']}?sig=x"

    def get_object(self, **kwargs):
        return {"Body": io.BytesIO(json.dumps(self.report).encode())}


class FakeSecrets:
    def get_secret_value(self, **kwargs):
        return {"SecretString": "test-api-key"}


@pytest.fixture(autouse=True)
def settings(monkeypatch: pytest.MonkeyPatch) -> None:
    for key, value in {
        "RUNPOD_ENDPOINT_ID": "endpoint123",
        "RUNPOD_SECRET_ARN": "secret-arn",
        "S3_BUCKET": "bucket",
        "INPUT_PREFIX": "inputs",
        "OUTPUT_PREFIX": "gpu-tests",
    }.items():
        monkeypatch.setenv(key, value)


def test_submit_and_status_verify_exact_artifacts() -> None:
    s3 = FakeS3()
    submitted = {}

    def runpod(method, path, key, body=None):
        if method == "POST":
            submitted.update(body)
            return {"id": "runpod123", "status": "IN_QUEUE"}
        return {
            "status": "COMPLETED",
            "output": {
                "run_id": submitted["input"]["run_id"],
                "frames": 523,
                "model_api": "mediapipe.tasks.vision.PoseLandmarker",
                "artifacts": submitted["input"]["output_s3_uris"],
            },
        }

    start = process(
        {"action": "submit", "input_key": "inputs/skier.mov"},
        s3=s3, secrets_client=FakeSecrets(), runpod_request=runpod,
    )
    assert start["job_id"] == "runpod123"
    assert start["source_version_id"] == "source-v1"
    assert "sig=x" not in json.dumps(start)
    assert submitted["input"]["source"]["version_id"] == "source-v1"
    assert submitted["policy"]["ttl"] < 3600_000
    s3.report = {
        "run_id": start["run_id"],
        "gpu_delegate_initialized": True,
        "gpu": {"name": "NVIDIA A4500"},
        "source": {"key": "inputs/skier.mov"},
        "artifacts_sha256": {
            name: "a" * 64 for name in ARTIFACTS if name != "performance"
        },
    }
    done = process(
        start["status_event"], s3=s3, secrets_client=FakeSecrets(),
        runpod_request=runpod,
    )
    assert done["status"] == "COMPLETED"
    assert len(done["artifacts"]) == len(ARTIFACTS)
    assert done["frames"] == 523


def test_submit_rejects_out_of_scope_key() -> None:
    with pytest.raises(ValueError, match="INPUT_PREFIX"):
        process(
            {"action": "submit", "input_key": "other/skier.mov"},
            s3=FakeS3(), secrets_client=FakeSecrets(),
            runpod_request=lambda *args: {},
        )
