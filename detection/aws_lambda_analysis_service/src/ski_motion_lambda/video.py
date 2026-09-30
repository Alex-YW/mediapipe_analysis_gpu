from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path

import cv2
import numpy as np

from .types import DecodedFrame, DecoderDiagnostics, VideoMetadata


def executable(name: str) -> str:
    adjacent = Path(sys.executable).parent / name
    if adjacent.is_file():
        return str(adjacent)
    found = shutil.which(name)
    if found:
        return found
    raise FileNotFoundError(
        f"Required executable `{name}` was not found in PATH or beside Python."
    )


def probe_video(path: Path) -> VideoMetadata:
    command = [
        executable("ffprobe"),
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=width,height,r_frame_rate,avg_frame_rate,nb_frames,duration:stream_tags=rotate:stream_side_data=rotation",
        "-of",
        "json",
        str(path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, check=True)
    streams = json.loads(completed.stdout).get("streams", [])
    if not streams:
        raise ValueError(f"No video stream found in {path}")
    stream = streams[0]
    fps = (
        _fraction(stream.get("r_frame_rate"))
        or _fraction(stream.get("avg_frame_rate"))
        or 30.0
    )
    rotation = int(float(stream.get("tags", {}).get("rotate", 0)))
    for side_data in stream.get("side_data_list", []):
        if "rotation" in side_data:
            rotation = int(round(float(side_data["rotation"])))
    return VideoMetadata(
        width=int(stream["width"]),
        height=int(stream["height"]),
        fps=fps,
        frame_count=(
            int(stream["nb_frames"])
            if stream.get("nb_frames", "N/A") not in (None, "N/A")
            else None
        ),
        duration_seconds=(
            float(stream["duration"])
            if stream.get("duration", "N/A") not in (None, "N/A")
            else None
        ),
        rotation=rotation % 360,
    )


def iter_video(
    path: Path,
    decoder_threads: int,
    diagnostics: DecoderDiagnostics,
):
    thread_property = getattr(cv2, "CAP_PROP_N_THREADS", None)
    if thread_property is None:
        capture = cv2.VideoCapture(str(path), cv2.CAP_FFMPEG)
    else:
        capture = cv2.VideoCapture(
            str(path),
            cv2.CAP_FFMPEG,
            [thread_property, decoder_threads],
        )
    if not capture.isOpened():
        capture.release()
        raise ValueError(f"OpenCV could not open video: {path}")
    if thread_property is not None:
        reported_threads = float(capture.get(thread_property))
        if np.isfinite(reported_threads) and reported_threads >= 0:
            diagnostics.actual_threads = int(round(reported_threads))
    if hasattr(cv2, "CAP_PROP_ORIENTATION_AUTO"):
        capture.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    source_frame_index = 0
    previous_timestamp = -1.0
    try:
        while True:
            ok, image = capture.read()
            if not ok:
                break
            reported_ms = float(capture.get(cv2.CAP_PROP_POS_MSEC))
            valid_timestamp = (
                np.isfinite(reported_ms)
                and reported_ms >= 0
                and (source_frame_index == 0 or reported_ms > previous_timestamp)
            )
            timestamp_ms = int(
                round(
                    reported_ms
                    if valid_timestamp
                    else source_frame_index * 1000 / fps
                )
            )
            previous_timestamp = float(timestamp_ms)
            yield DecodedFrame(
                image=image,
                source_frame_index=source_frame_index,
                timestamp_ms=timestamp_ms,
            )
            source_frame_index += 1
    finally:
        capture.release()


def rotate_sampled_frames(
    frames: Iterator[DecodedFrame],
    rotation: int,
    diagnostics: DecoderDiagnostics,
) -> Iterator[DecodedFrame]:
    """Orient only frames retained by temporal sampling."""

    for frame in frames:
        stage = time.perf_counter()
        image = rotate_frame(frame.image, rotation)
        diagnostics.rotation_seconds += time.perf_counter() - stage
        diagnostics.rotated_frames += 1
        yield DecodedFrame(
            image=image,
            source_frame_index=frame.source_frame_index,
            timestamp_ms=frame.timestamp_ms,
        )


def rotate_frame(frame: np.ndarray, rotation: int) -> np.ndarray:
    rotation %= 360
    if rotation == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    if rotation == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    if rotation == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    return frame


def _fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)
