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


def validate_analysis_plan(plan, source):
    if not isinstance(plan, dict) or plan.get("schema_version") != 1 or plan.get("accepted") is not True:
        raise ValueError("an approved analysis plan is required")
    expected = {key: source[key] for key in ("bucket", "key", "version_id")}
    if plan.get("source") != expected:
        raise ValueError("analysis plan source/version mismatch")
    duration = plan.get("duration_ms")
    if type(duration) is not int or not 0 < duration <= 30000:
        raise ValueError("invalid plan duration")
    intervals = plan.get("intervals")
    if not isinstance(intervals, list) or not intervals or len(intervals) > 1000:
        raise ValueError("plan has no accepted intervals")
    previous_end, ids = -1, set()
    for interval in intervals:
        if not isinstance(interval, dict):
            raise ValueError("invalid plan interval")
        start, end = interval.get("start_ms"), interval.get("end_ms")
        name = interval.get("interval_id")
        if (type(start) is not int or type(end) is not int or not 0 <= start < end <= duration
                or start < previous_end or not isinstance(name, str) or not name or name in ids):
            raise ValueError("invalid plan interval")
        previous_end = end
        ids.add(name)
    return plan
