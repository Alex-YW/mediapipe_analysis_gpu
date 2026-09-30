from __future__ import annotations

import csv
import logging
import math
import queue
import threading
import time
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any, Callable

from .artifacts import OverlayManifestWriter
from .body_tilt import (
    BODY_TILT_FIELDS,
    body_tilt_csv_values,
    estimate_skier_relative_body_tilt,
)
from .center_of_gravity import (
    CENTER_OF_GRAVITY_FIELDS,
    CENTER_OF_GRAVITY_LANDMARK,
    KneePlaneDistanceTracker,
    center_of_gravity_csv_values,
    estimate_body_up_plane_displacement,
    estimate_center_of_gravity,
)
from .condensed_angles import write_condensed_angles_csv
from .geometry import ANGLE_NAMES, calculate_angles
from .pose import PoseEstimator
from .signal_smoothing import (
    SHOULDER_HIP_ROTATION_SMOOTHED_FIELD,
    write_smoothed_rotation_column,
)
from .pose_quality import INVALID_FRAME_FIELD, INVALID_FRAME_REASON_FIELD
from .shin_symmetry import SHIN_SYMMETRY_FIELDS, calculate_shin_symmetry
from .turn_detection import write_outside_knee_plane_columns, write_turn_report
from .types import DecodedFrame, DecoderDiagnostics
from .video import iter_video, probe_video, rotate_sampled_frames


LOGGER = logging.getLogger(__name__)


def validate_pipeline_queue_size(queue_size: int) -> int:
    if (
        isinstance(queue_size, bool)
        or not isinstance(queue_size, int)
        or not 1 <= queue_size <= 32
    ):
        raise ValueError("pipeline_queue_size must be an integer from 1 through 32")
    return queue_size


def validate_processing_frame_rate(frame_rate: float | int) -> float:
    if (
        isinstance(frame_rate, bool)
        or not isinstance(frame_rate, (int, float))
        or not math.isfinite(float(frame_rate))
        or not 1 <= float(frame_rate) <= 240
    ):
        raise ValueError("processing_frame_rate must be a number from 1 through 240")
    return float(frame_rate)


def validate_decoder_threads(decoder_threads: int) -> int:
    if (
        isinstance(decoder_threads, bool)
        or not isinstance(decoder_threads, int)
        or not 0 <= decoder_threads <= 64
    ):
        raise ValueError("decoder_threads must be an integer from 0 through 64")
    return decoder_threads


def analyze_video(
    input_path: Path,
    output_dir: Path,
    *,
    source: Mapping[str, object] | None = None,
    visibility_threshold: float = 0.45,
    smoothing_alpha: float = 0.45,
    pipeline_queue_size: int = 4,
    processing_frame_rate: float = 30.0,
    decoder_threads: int = 0,
) -> dict[str, object]:
    """Decode and infer concurrently, producing only reusable analysis artifacts."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    if input_path.suffix.lower() not in {".mov", ".mp4"}:
        raise ValueError("Input must be a .mov or .mp4 video file.")
    if not 0 < visibility_threshold <= 1:
        raise ValueError("visibility_threshold must be in (0, 1]")
    if not 0 < smoothing_alpha <= 1:
        raise ValueError("smoothing_alpha must be in (0, 1]")
    pipeline_queue_size = validate_pipeline_queue_size(pipeline_queue_size)
    processing_frame_rate = validate_processing_frame_rate(processing_frame_rate)
    decoder_threads = validate_decoder_threads(decoder_threads)

    started = time.perf_counter()
    timings: dict[str, float] = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    angles_path = output_dir / "angles.csv"
    turns_path = output_dir / "turns.json"
    condensed_angles_path = output_dir / "condensed_angles.csv"
    manifest_path = output_dir / "overlay-manifest.jsonl.gz"

    stage = time.perf_counter()
    metadata = probe_video(input_path)
    timings["video_probe"] = time.perf_counter() - stage
    effective_frame_rate = min(processing_frame_rate, metadata.fps)
    decoder_diagnostics = DecoderDiagnostics(requested_threads=decoder_threads)

    estimator: PoseEstimator | None = None
    angle_file: Any | None = None
    manifest: OverlayManifestWriter | None = None
    decoder: _BackgroundDecoder | None = None
    manifest_complete = False
    frame_count = 0
    pose_diagnostics = {
        "frames_with_raw_image_landmarks": 0,
        "frames_with_accepted_image_landmarks": 0,
        "frames_with_raw_world_landmarks": 0,
        "frames_with_calculated_angles": 0,
        "frames_with_center_of_gravity": 0,
        "frames_with_invalid_frame": 0,
    }
    last_timestamp = -1
    failure: BaseException | None = None
    try:
        stage = time.perf_counter()
        estimator = PoseEstimator(visibility_threshold, smoothing_alpha)
        knee_plane_distance_tracker = KneePlaneDistanceTracker()
        timings["pose_initialization"] = time.perf_counter() - stage

        stage = time.perf_counter()
        angle_file = angles_path.open("w", newline="", encoding="utf-8")
        angle_writer = csv.DictWriter(
            angle_file,
            fieldnames=[
                "frame_index",
                "source_frame_index",
                "timestamp_ms",
                INVALID_FRAME_FIELD,
                INVALID_FRAME_REASON_FIELD,
                *ANGLE_NAMES,
                SHOULDER_HIP_ROTATION_SMOOTHED_FIELD,
                *BODY_TILT_FIELDS,
                *CENTER_OF_GRAVITY_FIELDS,
                *SHIN_SYMMETRY_FIELDS,
            ],
        )
        angle_writer.writeheader()
        manifest = OverlayManifestWriter(
            manifest_path,
            source=source or {"local_path": input_path.name},
            metadata=metadata,
            processing_frame_rate=effective_frame_rate,
            landmark_names=(*estimator.landmark_names, CENTER_OF_GRAVITY_LANDMARK),
        )
        timings["artifact_initialization"] = time.perf_counter() - stage

        decoder = _BackgroundDecoder(
            lambda: iter(
                rotate_sampled_frames(
                    _sample_video_frames(
                        iter_video(
                            input_path,
                            decoder_threads,
                            decoder_diagnostics,
                        ),
                        source_fps=metadata.fps,
                        target_fps=effective_frame_rate,
                    ),
                    metadata.rotation,
                    decoder_diagnostics,
                )
            ),
            pipeline_queue_size,
        )
        threaded_pipeline_started = time.perf_counter()
        decoder.start()
        frames = iter(decoder)

        while True:
            stage = time.perf_counter()
            try:
                decoded = next(frames)
            except StopIteration:
                timings["video_decode_wait"] = (
                    timings.get("video_decode_wait", 0.0)
                    + time.perf_counter()
                    - stage
                )
                break
            timings["video_decode_wait"] = (
                timings.get("video_decode_wait", 0.0)
                + time.perf_counter()
                - stage
            )

            timestamp_ms = max(decoded.timestamp_ms, last_timestamp + 1)
            last_timestamp = timestamp_ms

            stage = time.perf_counter()
            pose = estimator.process(decoded.image, timestamp_ms)
            _add_timing(timings, "pose_inference", stage)
            if pose.raw_image:
                pose_diagnostics["frames_with_raw_image_landmarks"] += 1
            if pose.image:
                pose_diagnostics["frames_with_accepted_image_landmarks"] += 1
            if pose.raw_world:
                pose_diagnostics["frames_with_raw_world_landmarks"] += 1
            if pose.invalid_frame:
                pose_diagnostics["frames_with_invalid_frame"] += 1

            stage = time.perf_counter()
            angles = calculate_angles(pose.world)
            body_tilt = estimate_skier_relative_body_tilt(pose.world)
            turn_body_tilt = (
                estimate_skier_relative_body_tilt(pose.turn_world)
                if pose.turn_world
                else body_tilt
            )
            body_tilt_values = body_tilt_csv_values(body_tilt)
            # Keep the established, continuous input to turn detection exactly
            # separate from the quality-filtered metric stream. Invalid rows
            # are still marked and excluded downstream, while turn boundaries
            # remain a faithful continuation of the prior implementation.
            body_tilt_values["ankle_to_shoulder_tilt_x_degrees"] = (
                turn_body_tilt.x if turn_body_tilt is not None else None
            )
            image_height, image_width = decoded.image.shape[:2]
            shin_symmetry = calculate_shin_symmetry(
                pose.image,
                image_width,
                image_height,
                minimum_confidence=visibility_threshold,
            )
            _add_timing(timings, "angle_calculation", stage)
            if any(value is not None for value in angles.values()):
                pose_diagnostics["frames_with_calculated_angles"] += 1

            stage = time.perf_counter()
            world_center_of_gravity = estimate_center_of_gravity(pose.world)
            image_center_of_gravity = estimate_center_of_gravity(pose.image)
            body_up_plane_displacement = estimate_body_up_plane_displacement(
                world_center_of_gravity,
                pose.world,
            )
            knee_plane_distances = knee_plane_distance_tracker.update(
                world_center_of_gravity,
                pose.world,
            )
            _add_timing(timings, "center_of_gravity_calculation", stage)
            if world_center_of_gravity is not None:
                pose_diagnostics["frames_with_center_of_gravity"] += 1

            stage = time.perf_counter()
            angle_writer.writerow(
                {
                    "frame_index": frame_count,
                    "source_frame_index": decoded.source_frame_index,
                    "timestamp_ms": timestamp_ms,
                    INVALID_FRAME_FIELD: pose.invalid_frame,
                    INVALID_FRAME_REASON_FIELD: pose.invalid_frame_reason,
                    **angles,
                    **body_tilt_values,
                    **center_of_gravity_csv_values(
                        world_center_of_gravity,
                        body_up_plane_displacement,
                        knee_plane_distances,
                    ),
                    **shin_symmetry,
                }
            )
            _add_timing(timings, "angle_csv_write", stage)

            stage = time.perf_counter()
            manifest_landmarks = dict(pose.image)
            if image_center_of_gravity is not None:
                manifest_landmarks[CENTER_OF_GRAVITY_LANDMARK] = (
                    image_center_of_gravity.point
                )
            manifest.write_frame(
                output_frame_index=frame_count,
                source_frame_index=decoded.source_frame_index,
                timestamp_ms=timestamp_ms,
                landmarks=manifest_landmarks,
                angles=angles,
                invalid_frame=pose.invalid_frame,
            )
            _add_timing(timings, "overlay_manifest_write", stage)
            frame_count += 1

        decoder.join()
        timings["video_decode"] = decoder.decode_seconds
        timings["sampled_frame_rotation"] = decoder_diagnostics.rotation_seconds
        timings["threaded_pipeline_wall"] = (
            time.perf_counter() - threaded_pipeline_started
        )
        parallel_work = sum(
            timings.get(name, 0.0)
            for name in (
                "video_decode",
                "pose_inference",
                "angle_calculation",
                "center_of_gravity_calculation",
                "angle_csv_write",
                "overlay_manifest_write",
            )
        )
        timings["parallel_overlap_estimate"] = max(
            0.0,
            parallel_work - timings["threaded_pipeline_wall"],
        )

        if frame_count == 0:
            raise ValueError("The input contains no decodable video frames.")
        if pose_diagnostics["frames_with_accepted_image_landmarks"] == 0:
            LOGGER.warning(
                "Pose detection produced zero accepted landmark frames out of %d "
                "processed video frames",
                frame_count,
            )

        stage = time.perf_counter()
        angle_file.flush()
        angle_file.close()
        angle_file = None
        write_smoothed_rotation_column(
            angles_path,
            sampling_rate_hz=effective_frame_rate,
        )
        timings["angle_smoothing"] = time.perf_counter() - stage

        stage = time.perf_counter()
        turn_report = write_turn_report(angles_path, turns_path)
        write_outside_knee_plane_columns(angles_path, turn_report)
        timings["turn_detection"] = time.perf_counter() - stage

        stage = time.perf_counter()
        write_condensed_angles_csv(angles_path, turns_path, condensed_angles_path)
        timings["condensed_data_export"] = time.perf_counter() - stage

        stage = time.perf_counter()
        manifest.close()
        manifest_complete = True
        timings["artifact_finalize"] = time.perf_counter() - stage
    except BaseException as error:
        failure = error
    finally:
        if decoder is not None:
            decoder.cancel()
            decoder.join()
        if manifest is not None and not manifest_complete:
            try:
                manifest.abort()
            except BaseException as error:
                failure = failure or error
        if angle_file is not None:
            angle_file.close()
        if estimator is not None:
            stage = time.perf_counter()
            try:
                estimator.close()
            except BaseException as error:
                failure = failure or error
            finally:
                timings["pose_shutdown"] = time.perf_counter() - stage

    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)

    elapsed = time.perf_counter() - started
    serial_accounted = sum(
        timings.get(name, 0.0)
        for name in (
            "video_probe",
            "pose_initialization",
            "artifact_initialization",
            "threaded_pipeline_wall",
            "artifact_finalize",
            "pose_shutdown",
        )
    )
    timings["other_overhead"] = max(0.0, elapsed - serial_accounted)
    timings["total_processing"] = elapsed
    return {
        "angles_csv": angles_path,
        "turns_json": turns_path,
        "condensed_angles_csv": condensed_angles_path,
        "overlay_manifest": manifest_path,
        "frames": frame_count,
        "processing_fps": frame_count / elapsed,
        "source_frame_rate": metadata.fps,
        "processing_frame_rate": effective_frame_rate,
        "decoder_threads_requested": decoder_diagnostics.requested_threads,
        "decoder_threads_actual": decoder_diagnostics.actual_threads,
        "rotated_frames": decoder_diagnostics.rotated_frames,
        "pose_diagnostics": pose_diagnostics,
        "timings": {name: round(value, 6) for name, value in timings.items()},
    }


def _sample_video_frames(
    frames: Iterator[DecodedFrame],
    *,
    source_fps: float,
    target_fps: float,
) -> Iterator[DecodedFrame]:
    """Select frames by source timestamp while retaining source frame identity."""

    if target_fps >= source_fps - 1e-6:
        yield from frames
        return

    interval_ms = 1000.0 / target_fps
    timestamp_tolerance_ms = 500.0 / source_fps
    next_sample_ms = 0.0
    for frame in frames:
        eligible_timestamp = frame.timestamp_ms + timestamp_tolerance_ms
        if eligible_timestamp < next_sample_ms:
            continue
        yield frame
        intervals_elapsed = max(
            1,
            math.floor((eligible_timestamp - next_sample_ms) / interval_ms) + 1,
        )
        next_sample_ms += intervals_elapsed * interval_ms


class _BackgroundDecoder:
    def __init__(
        self,
        iterator_factory: Callable[[], Iterator[DecodedFrame]],
        queue_size: int,
    ) -> None:
        self._iterator_factory = iterator_factory
        self._queue: queue.Queue[DecodedFrame] = queue.Queue(maxsize=queue_size)
        self._cancelled = threading.Event()
        self._thread = threading.Thread(
            target=self._run,
            name="video-decoder",
            daemon=True,
        )
        self._error: BaseException | None = None
        self._started = False
        self.decode_seconds = 0.0

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def __iter__(self) -> _BackgroundDecoder:
        return self

    def __next__(self) -> DecodedFrame:
        while True:
            try:
                return self._queue.get(timeout=0.05)
            except queue.Empty:
                if not self._thread.is_alive():
                    if self._error is not None:
                        raise RuntimeError("Video decoder thread failed") from self._error
                    raise StopIteration

    def cancel(self) -> None:
        self._cancelled.set()

    def join(self) -> None:
        if self._started:
            self._thread.join()

    def _run(self) -> None:
        frames: Iterator[DecodedFrame] | None = None
        try:
            frames = self._iterator_factory()
            while not self._cancelled.is_set():
                stage = time.perf_counter()
                try:
                    frame = next(frames)
                except StopIteration:
                    self.decode_seconds += time.perf_counter() - stage
                    break
                self.decode_seconds += time.perf_counter() - stage
                while not self._cancelled.is_set():
                    try:
                        self._queue.put(frame, timeout=0.05)
                        break
                    except queue.Full:
                        continue
        except BaseException as error:
            self._error = error
        finally:
            close = getattr(frames, "close", None)
            if callable(close):
                try:
                    close()
                except BaseException as error:
                    if self._error is None and not self._cancelled.is_set():
                        self._error = error


def _add_timing(timings: dict[str, float], name: str, started: float) -> None:
    timings[name] = timings.get(name, 0.0) + time.perf_counter() - started
