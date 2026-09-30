# Isolated MediaPipe Heavy GPU test service

This is a **test path**, alongside `rtmw3d_landmark_test_service`. It does not
change the current AWS analysis or render Lambdas, the website control plane,
or the production report path. An AWS Lambda submits and checks a RunPod
serverless job. The RunPod worker reads one *versioned* S3 video through a
presigned URL, runs MediaPipe Pose Landmarker **Heavy with the GPU delegate**,
reuses the existing ARCS analysis/metric/turn code and renderer, then writes
artifacts to a unique S3 output prefix through presigned PUT URLs. No AWS
credentials are placed in the RunPod worker.

The worker fails if `nvidia-smi` or GPU task initialization fails; it never
silently retries on CPU. The performance artifact records the GPU device and
delegate initialization. This is an **initialization check**, not yet a GPU
utilization or speed proof; confirm the first run in RunPod logs/metrics.

## Outputs

Each run writes `s3://BUCKET/OUTPUT_PREFIX/mpgpu-.../`:

| File | Meaning |
| --- | --- |
| `landmarks.jsonl` | One row per analyzed frame; source-frame ID, timestamp, raw and smoothed image/world landmarks, turn-detection world landmarks, quality flag |
| `angles.csv` | Existing Service 1 frame-level metric definitions |
| `turns.json` | Existing Service 1 turn detector and turn schema |
| `condensed_angles.csv` | Existing Service 1 condensed metric definitions |
| `overlay-manifest.jsonl.gz` | Existing renderer input, useful for diagnostics |
| `annotated.mp4` | H.264 visual verification video, with invalid-frame overlays omitted as in the current renderer |
| `performance.json` | Source identity, GPU/model details, timings and SHA-256 checksums |

This preserves the *calculation definitions*, not an assertion that Heavy Tasks
and the legacy MediaPipe Solutions Pose model yield interchangeable landmark
coordinates or confidence values. The existing MediaPipe quality recovery
behavior is also retained for this test. Compare the output against the same
source clip and frame sampling before any production decision.

## 1. Prepare the GitHub repository for RunPod's image builder

RunPod's [**Import Git Repository** flow](https://docs.runpod.io/serverless/workers/github-integration)
selects a branch and Dockerfile path.
The Dockerfile now uses one repository-root build context: no GitHub Actions,
Docker Hub, named BuildKit contexts or build arguments are required. The
GitHub repository must contain this minimum layout, preserving the paths:

```text
repository-root/
  detection/
    mediapipe_gpu_test_service/
      contract.py
      runpod_worker/
        Dockerfile
        requirements.txt
        __init__.py
        analysis.py
        handler.py
        pose_adapter.py
    aws_lambda_analysis_service/
      src/ski_motion_lambda/       # entire existing Python package
    aws_lambda_render_service/
      src/ski_motion_render/        # entire existing Python package
```

You may include the remaining small service files (README, tests and
`aws_submitter`) too. **Uploading only `mediapipe_gpu_test_service` is not
enough:** the worker imports the existing analysis and renderer packages to
retain the exact ARCS metric and turn definitions. Do not upload videos,
generated artifacts, checkpoints, AWS credentials, or the RunPod key. The
official ~31 MB Heavy model is downloaded from Google's versioned `/1/` URL
during the RunPod image build and checked against the SHA-256 pinned in the
Dockerfile. If Google serves different bytes, the build fails rather than
silently changing the test model. That pinned digest has not yet been confirmed
by a completed build in this workspace; the first RunPod build is the required
verification.

The image uses Ubuntu 24.04, Python 3.12, MediaPipe 0.10.21, the existing ARCS
source, FFmpeg/libx264 and graphics libraries. Each source file to upload is
well below GitHub's 100 MB single-file limit; the built image stays in
RunPod's registry.

## 2. Create the RunPod serverless endpoint

1. In RunPod settings, connect the GitHub account with access to the
   repository. Go to **Serverless → New Endpoint → Import Git Repository**,
   select the repository and branch, and set **Dockerfile Path** to
   `detection/mediapipe_gpu_test_service/runpod_worker/Dockerfile`. If your
   RunPod screen also asks for a **Context Path**, choose the repository root
   (`/`), not the Dockerfile's directory.
2. Choose a **Queue** endpoint and an NVIDIA GPU type with Linux graphics/EGL
   support. Set max concurrent jobs
   per worker to **1** during validation; start with one worker and enough
   container disk for a 1 GiB input, intermediate artifacts and the rendered
   video (for example, 30 GiB). Leave autoscaling/min workers according to
   the desired cost/cold-start tradeoff. Click **Deploy Endpoint**, inspect
   the **Builds** tab, and copy the **endpoint ID** once the image build and
   worker test complete. The image's default command starts the RunPod Python
   handler; no HTTP server or exposed port is needed.
3. Create a RunPod API key permitted to submit and check jobs on this
   endpoint. Do not paste the key into the container or GitHub repository.
4. The first job is a smoke test: its status must be `COMPLETED`,
   `performance.json` must say `gpu_delegate_initialized: true`, and the
   RunPod logs/metrics must show inference on the intended GPU. Failure to
   initialize EGL/GPU is a real test failure, not a cue to auto-fallback.

The worker expects the endpoint's standard RunPod `/run` and `/status` API.
Its maximum requested execution is 20 minutes, job TTL 45 minutes, and S3
presigned URLs live for 60 minutes. Longer queues/runs require changing these
limits together before use. The worker currently runs all frames (sampled to
30 fps by default); view intervals are not part of this isolated test.
RunPod's GitHub builder has a 30-minute Docker-build limit; this image does
not need a GPU during build, only at worker runtime. To update the endpoint
later, follow RunPod's GitHub release/rebuild flow and check the new build in
the Builds tab.

## 3. Prepare S3 and AWS credentials

1. Create or select a private S3 bucket in the desired AWS region. **Enable
   bucket versioning** before uploading the test video. Keep public access
   blocked. Choose separate prefixes, for example `mediapipe-gpu-test/input/`
   and `mediapipe-gpu-test/output/`. Upload a short `.MOV` or `.mp4` test video
   under the input prefix. The test limit is 1 GiB. Do not put output under
   the input prefix.
2. In AWS Secrets Manager, create a **plaintext** secret containing only the
   RunPod API key. Copy its ARN. If using a customer-managed KMS key, grant
   Lambda's role `kms:Decrypt` for that key.
3. Create a Lambda execution role with basic CloudWatch Logs access and these
   additional permissions, substituting exact account, bucket and prefixes:

   ```json
   {
     "Version": "2012-10-17",
     "Statement": [
       {
         "Effect": "Allow",
         "Action": ["s3:GetObject", "s3:GetObjectVersion"],
         "Resource": "arn:aws:s3:::BUCKET/mediapipe-gpu-test/input/*"
       },
       {
         "Effect": "Allow",
         "Action": ["s3:GetObject", "s3:PutObject"],
         "Resource": "arn:aws:s3:::BUCKET/mediapipe-gpu-test/output/*"
       },
       {
         "Effect": "Allow",
         "Action": "secretsmanager:GetSecretValue",
         "Resource": "RUNPOD_SECRET_ARN"
       }
     ]
   }
   ```

   The Lambda role's S3 permissions authorize the presigned GET/PUT requests
   that the RunPod worker later makes. Check the bucket policy for any deny
   rules that might also affect presigned access (VPC-only access is a common
   example). Restrict Lambda invocation to test operators; do not expose it
   as an unauthenticated public endpoint.

## 4. Create the AWS submit/status Lambda

1. From the repository root, package only the coordinator code and shared contract:

   ```bash
   cd detection/mediapipe_gpu_test_service
   zip -r /tmp/arcs-mediapipe-gpu-submit.zip aws_submitter contract.py \
     -x '*/__pycache__/*' '*.pyc'
   ```

   If already in this directory, omit the `cd`. No third-party Python package
   is needed in the zip; use the AWS-managed Python 3.12 runtime with its
   included `boto3`/`botocore`.
2. Create a Python 3.12 Lambda, upload the zip, set handler to
   `aws_submitter.lambda_function.lambda_handler`, attach the role above, set
   timeout to 30 seconds and memory to 256 MB. Keep it outside a private VPC
   unless that VPC has outbound access to RunPod, S3 and Secrets Manager.
3. Set these environment variables:

   ```text
   RUNPOD_ENDPOINT_ID=<endpoint ID>
   RUNPOD_SECRET_ARN=<plaintext secret ARN>
   S3_BUCKET=<bucket name>
   S3_REGION=<bucket region, such as ap-southeast-1>
   INPUT_PREFIX=mediapipe-gpu-test/input
   OUTPUT_PREFIX=mediapipe-gpu-test/output
   ```

   Prefixes may omit their trailing `/`; the Lambda normalizes them.

## 5. Submit, check, and inspect

Invoke the Lambda directly (console test event or AWS CLI) with:

```json
{
  "action": "submit",
  "input_key": "mediapipe-gpu-test/input/rear_view_17s.MOV",
  "processing_frame_rate": 30
}
```

Optionally include `input_version_id` to pin a specific version; otherwise
the Lambda pins the latest version at submission time. Save the returned
`status_event` and invoke the **same Lambda** with it:

```json
{"action":"status","run_id":"mpgpu-...","job_id":"..."}
```

Repeat status checks while `IN_QUEUE` or `IN_PROGRESS`. A `COMPLETED` status
returns each artifact's S3 URI/size/version plus GPU and frame metadata only
after checking all seven objects and the performance report. Download
`annotated.mp4` for visual inspection and compare `angles.csv`, `turns.json`
and `condensed_angles.csv` to the current AWS MediaPipe run of the exact same
video and processing frame rate. This Lambda does **not** invoke the ARCS
production control plane or generate a paid user report. RunPod retains async
job results for only 30 minutes after completion; poll promptly and save the
returned S3 prefix. The S3 objects themselves remain until your bucket's
lifecycle policy removes them.

## Local checks and current validation boundary

From this directory, with the existing ARCS Python test environment:

```bash
PYTHONPATH=.:../aws_lambda_analysis_service/src:../aws_lambda_render_service/src \
  /Users/aw/miniconda3/envs/new_env/bin/python -m pytest -q tests
```

Unit tests check payload validation, GPU-fail-closed behavior, landmark/angles
alignment and mocked submit/status. They do not prove the image builds or GPU
inference works on a chosen RunPod host. The next gate is the small real-job
smoke test above. If its GPU delegate fails, inspect the RunPod GPU/driver and
EGL runtime before changing model or disabling the GPU-only constraint.
