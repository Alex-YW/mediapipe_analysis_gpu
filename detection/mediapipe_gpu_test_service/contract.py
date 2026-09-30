"""Shared, small artifact contract for the isolated RunPod GPU test."""

ARTIFACTS = {
    "landmarks": ("landmarks.jsonl", "application/x-ndjson"),
    "angles": ("angles.csv", "text/csv"),
    "turns": ("turns.json", "application/json"),
    "condensed_angles": ("condensed_angles.csv", "text/csv"),
    "overlay_manifest": ("overlay-manifest.jsonl.gz", "application/x-ndjson"),
    "annotated_video": ("annotated.mp4", "video/mp4"),
    "performance": ("performance.json", "application/json"),
}

MAX_INPUT_BYTES = 1024 * 1024 * 1024
URL_EXPIRY_SECONDS = 3600
EXECUTION_TIMEOUT_MS = 20 * 60 * 1000
JOB_TTL_MS = 45 * 60 * 1000
