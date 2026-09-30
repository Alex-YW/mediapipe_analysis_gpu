# RunPod GitHub integration: select this Dockerfile and provide a repository
# whose root contains the three detection/ directories copied below.
# The build context is the GitHub repository root, with no extra BuildKit
# contexts or build arguments.
FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics,video \
    PATH=/opt/venv/bin:$PATH \
    PYTHONPATH=/app:/app/analysis/src:/app/render/src \
    POSE_MODEL_PATH=/opt/models/pose_landmarker_heavy.task

RUN apt-get update && apt-get install -y --no-install-recommends \
    ca-certificates curl ffmpeg python3 python3-pip python3-venv \
    libegl1 libgles2 libgl1 libglib2.0-0 libgomp1 && \
    rm -rf /var/lib/apt/lists/* && \
    python3 -m venv /opt/venv && \
    test "$(uname -m)" = x86_64 && \
    ffmpeg -hide_banner -encoders 2>/dev/null | grep -q libx264

COPY detection/mediapipe_gpu_test_service/runpod_worker/requirements.txt /tmp/requirements.txt
RUN python -m pip install --no-cache-dir -r /tmp/requirements.txt && \
    python -m pip check

# Download the versioned official Heavy task during the build, not via GitHub.
# The checksum is pinned so a changed model fails the build. This expected
# digest is verified against the downloaded Google-hosted file at build time.
RUN mkdir -p /opt/models && \
    curl -fL --retry 5 --retry-all-errors \
      https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task \
      -o /opt/models/pose_landmarker_heavy.task && \
    echo '64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b  /opt/models/pose_landmarker_heavy.task' | sha256sum -c -

WORKDIR /app
COPY detection/mediapipe_gpu_test_service/contract.py /app/
COPY detection/mediapipe_gpu_test_service/runpod_worker/ /app/runpod_worker/
COPY detection/aws_lambda_analysis_service/src/ski_motion_lambda/ /app/analysis/src/ski_motion_lambda/
COPY detection/aws_lambda_render_service/src/ski_motion_render/ /app/render/src/ski_motion_render/

CMD ["python", "-u", "-m", "runpod_worker.handler"]
