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


def test_production_submission_is_cached_and_completion_matches_plan(monkeypatch):
    monkeypatch.setenv("INPUT_PREFIX", "uploads")
    monkeypatch.setenv("OUTPUT_PREFIX", "results")
    job = "job_abcdefgh"
    source = {"bucket": "bucket", "key": f"uploads/{job}/source.mov", "version_id": "source-v1"}
    plan_key = f"results/{job}/analysis-plan.json"
    plan = {"schema_version": 1, "accepted": True, "source": source, "duration_ms": 10000,
            "intervals": [{"interval_id": "a", "start_ms": 0, "end_ms": 4000}]}
    class ProductionS3(FakeS3):
        def __init__(self):
            super().__init__()
            self.objects = {plan_key: json.dumps(plan).encode()}
        def head_object(self, **kwargs):
            if kwargs["Key"].startswith("uploads/"):
                return {"VersionId": "source-v1", "ContentLength": 1000}
            return super().head_object(**kwargs)
        def get_object(self, **kwargs):
            if kwargs["Key"].endswith("performance.json"):
                return {"Body": io.BytesIO(json.dumps(self.report).encode())}
            if kwargs["Key"] not in self.objects:
                error = RuntimeError("missing")
                error.response = {"Error": {"Code": "NoSuchKey"}}
                raise error
            return {"Body": io.BytesIO(self.objects[kwargs["Key"]])}
        def put_object(self, **kwargs):
            if kwargs.get("IfNoneMatch"):
                assert kwargs["Key"] not in self.objects
            self.objects[kwargs["Key"]] = kwargs["Body"]
            return {"VersionId": "v1"}
    s3, requests = ProductionS3(), []
    def provider(method, path, key, body=None):
        if method == "POST":
            requests.append(body)
            return {"id": "runpod123", "status": "IN_QUEUE"}
        return {"status": "COMPLETED", "output": {"run_id": job,
            "artifacts": requests[0]["input"]["output_s3_uris"], "frames": 300}}
    event = {"action": "submit_arcs", "jobId": job, "input_key": source["key"],
             "input_version_id": "source-v1", "analysis_plan": {
                 "bucket": "bucket", "key": plan_key, "version_id": "plan-v1"}}
    result = process(event, s3=s3, secrets_client=FakeSecrets(), runpod_request=provider)
    assert process(event, s3=s3, secrets_client=FakeSecrets(), runpod_request=provider) == result
    assert len(requests) == 1 and requests[0]["input"]["analysis_plan"] == plan
    s3.report = {"run_id": job, "gpu_delegate_initialized": True, "gpu": {"name": "GPU"},
                 "source": source, "analysis": {"analysis_plan": plan},
                 "artifacts_sha256": {k: "a" * 64 for k in ARTIFACTS if k != "performance"}}
    done = process(result["status_event"], s3=s3, secrets_client=FakeSecrets(), runpod_request=provider)
    assert done["render"]["outputs"]["annotated_video"]["key"] == f"results/{job}/annotated.mp4"
    assert done["analysis"]["source"] == source
    s3.report["analysis"]["analysis_plan"] = None
    with pytest.raises(ValueError, match="approved interval plan"):
        process(result["status_event"], s3=s3, secrets_client=FakeSecrets(), runpod_request=provider)


def test_ambiguous_provider_post_is_not_repeated(monkeypatch):
    from aws_submitter.lambda_function import _submit_arcs
    class ClaimedS3:
        def get_object(self, **kwargs):
            return {"Body": io.BytesIO(json.dumps({"state": "SUBMITTING", "source_version_id": "v1",
                "analysis_plan": {"key": "plan"}}).encode())}
    with pytest.raises(RuntimeError, match="SUBMISSION_UNCERTAIN"):
        _submit_arcs({"jobId": "job_abcdefgh", "input_key": "uploads/job_abcdefgh/source.mov",
                      "input_version_id": "v1", "analysis_plan": {"key": "plan"}},
                     s3=ClaimedS3(), bucket="bucket", output_prefix="results/")
