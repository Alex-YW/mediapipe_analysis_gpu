"""Static checks for RunPod's single repository-root Docker build context."""

from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
DOCKERFILE = (
    REPO_ROOT
    / "detection/mediapipe_gpu_test_service/runpod_worker/Dockerfile"
)


def test_runpod_dockerfile_needs_only_the_repository_root_context() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")
    assert "COPY --from=" not in source
    assert "ARG MODEL_SHA256" not in source
    for line in source.splitlines():
        if line.startswith("COPY "):
            source_path = line.split()[1]
            assert source_path.startswith("detection/")
            assert (REPO_ROOT / source_path).exists(), source_path


def test_runpod_build_pins_model_version_and_digest() -> None:
    source = DOCKERFILE.read_text(encoding="utf-8")
    assert "/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task" in source
    assert "sha256sum -c -" in source
    assert "64437af838a65d18e5ba7a0d39b465540069bc8aae8308de3e318aad31fcbc7b" in source
