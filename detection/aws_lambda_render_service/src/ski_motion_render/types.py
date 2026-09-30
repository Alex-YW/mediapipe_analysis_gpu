from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True, slots=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0


@dataclass(frozen=True, slots=True)
class ManifestFrame:
    output_frame_index: int
    source_frame_index: int
    timestamp_ms: int
    landmarks: dict[str, Point3D]
    angles: dict[str, float | None]
    invalid_frame: bool = False


@dataclass(frozen=True, slots=True)
class RenderManifest:
    header: dict[str, object]
    frames: tuple[ManifestFrame, ...]
    footer: dict[str, object]


@dataclass(frozen=True, slots=True)
class VideoMetadata:
    width: int
    height: int
    fps: float
    frame_count: int | None
    duration_seconds: float | None
    rotation: int = 0


@dataclass(slots=True)
class DecodedFrame:
    image: object
    source_frame_index: int


@dataclass(slots=True)
class DecoderDiagnostics:
    requested_threads: int
    actual_threads: int | None = None
    source_frames_traversed: int = 0
    selected_frames_retrieved: int = 0
    decode_seconds: float = 0.0
    retrieval_seconds: float = 0.0
    rotation_seconds: float = 0.0


@dataclass(slots=True)
class PipelineTimings:
    values: dict[str, float] = field(default_factory=dict)

    def add(self, name: str, value: float) -> None:
        self.values[name] = self.values.get(name, 0.0) + value
