from __future__ import annotations

from dataclasses import dataclass

from .types import Point3D


@dataclass(slots=True)
class _PointState:
    point: Point3D
    missed: int = 0


class LandmarkSmoother:
    """Confidence-aware exponential smoother with short-gap expiry."""

    def __init__(self, alpha: float = 0.45, max_gap: int = 3) -> None:
        if not 0 < alpha <= 1:
            raise ValueError("alpha must be in (0, 1]")
        self.alpha = alpha
        self.max_gap = max_gap
        self._states: dict[str, _PointState] = {}

    def update(
        self,
        points: dict[str, Point3D],
        minimum_confidence: float,
    ) -> dict[str, Point3D]:
        output: dict[str, Point3D] = {}
        names = set(points) | set(self._states)
        for name in names:
            incoming = points.get(name)
            valid = (
                incoming is not None
                and min(incoming.visibility, incoming.presence) >= minimum_confidence
            )
            state = self._states.get(name)
            if valid and incoming is not None:
                if state is None:
                    smoothed = incoming
                else:
                    alpha = self.alpha
                    old = state.point
                    smoothed = Point3D(
                        x=alpha * incoming.x + (1 - alpha) * old.x,
                        y=alpha * incoming.y + (1 - alpha) * old.y,
                        z=alpha * incoming.z + (1 - alpha) * old.z,
                        visibility=incoming.visibility,
                        presence=incoming.presence,
                    )
                self._states[name] = _PointState(smoothed)
                output[name] = smoothed
            elif state is not None:
                state.missed += 1
                if state.missed <= self.max_gap:
                    output[name] = Point3D(
                        state.point.x,
                        state.point.y,
                        state.point.z,
                        state.point.visibility * (0.65**state.missed),
                        state.point.presence * (0.65**state.missed),
                    )
                else:
                    del self._states[name]
        return output

    def reset(self) -> None:
        """Discard temporal state after a frame rejected by an external gate."""

        self._states.clear()
