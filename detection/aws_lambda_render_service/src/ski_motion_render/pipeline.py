from __future__ import annotations

import math
import queue
import threading
import time
from collections.abc import Callable, Iterator
from pathlib import Path

from .overlay import draw_overlay
from .types import DecodedFrame, DecoderDiagnostics, RenderManifest
from .video import VideoWriter, iter_selected_frames, probe_video


X264_PRESETS = frozenset(
    {
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    }
)


def validate_encoder_settings(crf: int, preset: str) -> tuple[int, str]:
    if isinstance(crf, bool) or not isinstance(crf, int) or not 0 <= crf <= 51:
        raise ValueError("crf must be an integer from 0 through 51")
    normalized = preset.strip().lower() if isinstance(preset, str) else ""
    if normalized not in X264_PRESETS:
        choices = ", ".join(sorted(X264_PRESETS))
        raise ValueError(f"encoder_preset must be one of: {choices}")
    return crf, normalized


def validate_pipeline_queue_size(queue_size: int) -> int:
    if (
        isinstance(queue_size, bool)
        or not isinstance(queue_size, int)
        or not 1 <= queue_size <= 32
    ):
        raise ValueError("pipeline_queue_size must be an integer from 1 through 32")
    return queue_size


def validate_decoder_threads(decoder_threads: int) -> int:
    if (
        isinstance(decoder_threads, bool)
        or not isinstance(decoder_threads, int)
        or not 0 <= decoder_threads <= 64
    ):
        raise ValueError("decoder_threads must be an integer from 0 through 64")
    return decoder_threads


def render_video(
    input_path: Path,
    output_path: Path,
    manifest: RenderManifest,
    *,
    crf: int = 20,
    encoder_preset: str = "veryfast",
    pipeline_queue_size: int = 4,
    decoder_threads: int = 0,
) -> dict[str, object]:
    """Render Service 1 landmarks and angles onto their exact source frames."""

    if not input_path.is_file():
        raise FileNotFoundError(f"Input video not found: {input_path}")
    crf, encoder_preset = validate_encoder_settings(crf, encoder_preset)
    pipeline_queue_size = validate_pipeline_queue_size(pipeline_queue_size)
    decoder_threads = validate_decoder_threads(decoder_threads)

    started = time.perf_counter()
    timings: dict[str, float] = {}
    stage = time.perf_counter()
    metadata = probe_video(input_path)
    timings["video_probe"] = time.perf_counter() - stage
    video = manifest.header["video"]
    assert isinstance(video, dict)
    _validate_video_matches_manifest(metadata, video)
    output_width = int(video["display_width"])
    output_height = int(video["display_height"])
    output_fps = float(video["processing_frame_rate"])
    rotation = int(video["rotation_degrees"])
    diagnostics = DecoderDiagnostics(requested_threads=decoder_threads)

    writer: VideoWriter | None = None
    decoder: _BackgroundSelectedDecoder | None = None
    failure: BaseException | None = None
    rendered_frames = 0
    try:
        stage = time.perf_counter()
        writer = VideoWriter(
            output_path,
            output_width,
            output_height,
            output_fps,
            crf,
            encoder_preset,
        )
        timings["encoder_initialization"] = time.perf_counter() - stage
        indices = tuple(frame.source_frame_index for frame in manifest.frames)
        decoder = _BackgroundSelectedDecoder(
            lambda: iter(
                iter_selected_frames(
                    input_path,
                    indices,
                    rotation,
                    decoder_threads,
                    diagnostics,
                )
            ),
            pipeline_queue_size,
        )
        threaded_started = time.perf_counter()
        decoder.start()
        decoded_frames = iter(decoder)

        for manifest_frame in manifest.frames:
            stage = time.perf_counter()
            try:
                try:
                    decoded = next(decoded_frames)
                except StopIteration as error:
                    raise ValueError(
                        "source video produced fewer frames than the overlay manifest"
                    ) from error
            finally:
                timings["video_decode_wait"] = (
                    timings.get("video_decode_wait", 0.0)
                    + time.perf_counter()
                    - stage
                )
            if decoded.source_frame_index != manifest_frame.source_frame_index:
                raise ValueError("decoded source frame does not match overlay manifest")
            image_height, image_width = decoded.image.shape[:2]
            if (image_width, image_height) != (output_width, output_height):
                raise ValueError(
                    "oriented source frame dimensions do not match overlay manifest"
                )

            stage = time.perf_counter()
            annotated = (
                decoded.image
                if manifest_frame.invalid_frame
                else draw_overlay(
                    decoded.image,
                    manifest_frame.landmarks,
                    manifest_frame.angles,
                    manifest_frame.timestamp_ms,
                )
            )
            timings["overlay_rendering"] = (
                timings.get("overlay_rendering", 0.0)
                + time.perf_counter()
                - stage
            )

            stage = time.perf_counter()
            writer.write(annotated)
            timings["video_encode_write"] = (
                timings.get("video_encode_write", 0.0)
                + time.perf_counter()
                - stage
            )
            rendered_frames += 1

        decoder.join()
        decoder.raise_if_failed()
        try:
            next(decoded_frames)
        except StopIteration:
            pass
        else:
            raise ValueError("decoder produced more frames than the manifest contains")
        timings["threaded_pipeline_wall"] = time.perf_counter() - threaded_started
    except BaseException as error:
        failure = error
    finally:
        if decoder is not None:
            decoder.cancel()
            decoder.join()
        if writer is not None:
            stage = time.perf_counter()
            try:
                writer.close()
            except BaseException as error:
                failure = failure or error
            finally:
                timings["video_encode_finalize"] = time.perf_counter() - stage

    if failure is not None:
        raise failure.with_traceback(failure.__traceback__)
    if rendered_frames != len(manifest.frames):
        raise ValueError("rendered frame count does not match overlay manifest")

    timings["video_decode"] = diagnostics.decode_seconds
    timings["selected_frame_retrieval"] = diagnostics.retrieval_seconds
    timings["sampled_frame_rotation"] = diagnostics.rotation_seconds
    parallel_work = sum(
        timings.get(name, 0.0)
        for name in (
            "video_decode",
            "selected_frame_retrieval",
            "sampled_frame_rotation",
            "overlay_rendering",
            "video_encode_write",
        )
    )
    timings["parallel_overlap_estimate"] = max(
        0.0,
        parallel_work - timings.get("threaded_pipeline_wall", 0.0),
    )
    elapsed = time.perf_counter() - started
    serial_accounted = sum(
        timings.get(name, 0.0)
        for name in (
            "video_probe",
            "encoder_initialization",
            "threaded_pipeline_wall",
            "video_encode_finalize",
        )
    )
    timings["other_overhead"] = max(0.0, elapsed - serial_accounted)
    timings["total_processing"] = elapsed
    return {
        "annotated_video": output_path,
        "frames": rendered_frames,
        "processing_fps": rendered_frames / elapsed,
        "output_frame_rate": output_fps,
        "decoder_threads_requested": diagnostics.requested_threads,
        "decoder_threads_actual": diagnostics.actual_threads,
        "source_frames_traversed": diagnostics.source_frames_traversed,
        "selected_frames_retrieved": diagnostics.selected_frames_retrieved,
        "timings": {name: round(value, 6) for name, value in timings.items()},
    }


def _validate_video_matches_manifest(metadata: object, video: dict[str, object]) -> None:
    expected = (
        int(video["encoded_width"]),
        int(video["encoded_height"]),
        int(video["rotation_degrees"]),
    )
    actual = (metadata.width, metadata.height, metadata.rotation)
    if actual != expected:
        raise ValueError(
            "source video dimensions or rotation do not match overlay manifest"
        )
    source_fps = float(video["source_frame_rate"])
    if not math.isclose(metadata.fps, source_fps, rel_tol=1e-4, abs_tol=1e-3):
        raise ValueError("source video frame rate does not match overlay manifest")


class _BackgroundSelectedDecoder:
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
            name="selected-video-decoder",
            daemon=True,
        )
        self._error: BaseException | None = None
        self._started = False

    def start(self) -> None:
        self._thread.start()
        self._started = True

    def __iter__(self) -> _BackgroundSelectedDecoder:
        return self

    def __next__(self) -> DecodedFrame:
        while True:
            try:
                return self._queue.get(timeout=0.05)
            except queue.Empty:
                if not self._thread.is_alive():
                    self.raise_if_failed()
                    raise StopIteration

    def cancel(self) -> None:
        self._cancelled.set()

    def join(self) -> None:
        if self._started:
            self._thread.join()

    def raise_if_failed(self) -> None:
        if self._error is not None:
            raise RuntimeError("selected video decoder thread failed") from self._error

    def _run(self) -> None:
        frames: Iterator[DecodedFrame] | None = None
        try:
            frames = self._iterator_factory()
            while not self._cancelled.is_set():
                try:
                    frame = next(frames)
                except StopIteration:
                    break
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
