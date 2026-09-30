from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from collections.abc import Iterator, Sequence
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


def iter_selected_frames(
    path: Path,
    source_frame_indices: Sequence[int],
    rotation: int,
    decoder_threads: int,
    diagnostics: DecoderDiagnostics,
) -> Iterator[DecodedFrame]:
    """Traverse the codec stream but retrieve and rotate only requested frames."""

    if not source_frame_indices:
        return
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

    target_position = 0
    source_frame_index = 0
    try:
        while target_position < len(source_frame_indices):
            stage = time.perf_counter()
            grabbed = capture.grab()
            diagnostics.decode_seconds += time.perf_counter() - stage
            if not grabbed:
                break
            diagnostics.source_frames_traversed += 1
            target_index = source_frame_indices[target_position]
            if source_frame_index == target_index:
                stage = time.perf_counter()
                retrieved, image = capture.retrieve()
                diagnostics.retrieval_seconds += time.perf_counter() - stage
                if not retrieved or image is None:
                    raise ValueError(
                        f"OpenCV could not retrieve source frame {source_frame_index}"
                    )
                stage = time.perf_counter()
                image = rotate_frame(image, rotation)
                diagnostics.rotation_seconds += time.perf_counter() - stage
                diagnostics.selected_frames_retrieved += 1
                yield DecodedFrame(
                    image=image,
                    source_frame_index=source_frame_index,
                )
                target_position += 1
            source_frame_index += 1
    finally:
        capture.release()

    if target_position != len(source_frame_indices):
        missing = source_frame_indices[target_position]
        raise ValueError(
            f"Source video ended before manifest source frame {missing} was found"
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


class VideoWriter:
    def __init__(
        self,
        output: Path,
        width: int,
        height: int,
        fps: float,
        crf: int,
        preset: str,
    ) -> None:
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            executable("ffmpeg"),
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s:v",
            f"{width}x{height}",
            "-r",
            f"{fps:.8f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            preset,
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        self._process = subprocess.Popen(
            command,
            stdin=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self._closed = False

    def write(self, frame: np.ndarray) -> None:
        if self._closed or self._process.stdin is None:
            raise RuntimeError("Video writer is closed")
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError as error:
            stderr = self._read_stderr()
            raise RuntimeError(f"FFmpeg stopped while encoding: {stderr}") from error

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._process.stdin:
            self._process.stdin.close()
        stderr = self._read_stderr()
        return_code = self._process.wait()
        if return_code:
            raise RuntimeError(f"FFmpeg encoding failed ({return_code}): {stderr}")

    def _read_stderr(self) -> str:
        return (
            self._process.stderr.read().decode("utf-8", errors="replace")
            if self._process.stderr
            else ""
        )


def _fraction(value: str | None) -> float | None:
    if not value or value == "0/0":
        return None
    numerator, denominator = value.split("/", 1)
    return float(numerator) / float(denominator)
