from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class Point3D:
    x: float
    y: float
    z: float = 0.0
    visibility: float = 1.0
    presence: float = 1.0

    def array(self) -> tuple[float, float, float]:
        return self.x, self.y, self.z


@dataclass(slots=True)
class PoseResult:
    raw_image: dict[str, Point3D] = field(default_factory=dict)
    image: dict[str, Point3D] = field(default_factory=dict)
    raw_world: dict[str, Point3D] = field(default_factory=dict)
    world: dict[str, Point3D] = field(default_factory=dict)
    turn_world: dict[str, Point3D] = field(default_factory=dict)
    invalid_frame: bool = False
    invalid_frame_reason: str = ""


@dataclass(slots=True)
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
    timestamp_ms: int


@dataclass(slots=True)
class DecoderDiagnostics:
    requested_threads: int
    actual_threads: int | None = None
    rotated_frames: int = 0
    rotation_seconds: float = 0.0
