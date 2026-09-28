# SPDX-License-Identifier: AGPL-3.0-only
"""Seat geometry and time filtering; no camera, ML, or network dependencies."""

from dataclasses import dataclass, field
import math

Point = tuple[float, float]


@dataclass(frozen=True)
class Seat:
    id: str
    polygon: tuple[Point, ...]

    @property
    def center(self) -> Point:
        return (
            sum(p[0] for p in self.polygon) / len(self.polygon),
            sum(p[1] for p in self.polygon) / len(self.polygon),
        )


@dataclass(frozen=True)
class Detection:
    # Normalized coordinates in the original camera frame, not letterboxed input.
    box: tuple[float, float, float, float]
    confidence: float

    @property
    def anchor(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)


def contains(polygon: tuple[Point, ...], point: Point) -> bool:
    """Ray casting, including boundary points."""
    x, y = point
    inside = False
    for i, (ax, ay) in enumerate(polygon):
        bx, by = polygon[(i + 1) % len(polygon)]
        cross = (x - ax) * (by - ay) - (y - ay) * (bx - ax)
        if abs(cross) < 1e-10 and min(ax, bx) <= x <= max(ax, bx) and min(ay, by) <= y <= max(ay, by):
            return True
        if (ay > y) != (by > y) and x < (bx - ax) * (y - ay) / (by - ay) + ax:
            inside = not inside
    return inside


def assign_people(seats: list[Seat], detections: list[Detection]) -> dict[str, float]:
    """Assign each person to at most one seat; ignore people in the aisle.

    Draw each ROI around where a seated person's BOX CENTER is visible. This
    simple heuristic is not pose recognition and cannot prove someone is seated.
    """
    scores = {seat.id: 0.0 for seat in seats}
    for detection in detections:
        candidates = [s for s in seats if contains(s.polygon, detection.anchor)]
        if candidates:
            nearest = min(candidates, key=lambda s: math.dist(s.center, detection.anchor))
            scores[nearest.id] = max(scores[nearest.id], detection.confidence)
    return scores


@dataclass
class SeatState:
    occupied_seconds: float = 1.0
    empty_seconds: float = 8.0
    max_gap_seconds: float = 5.0
    state: str = field(default="UNKNOWN", init=False)
    candidate: str | None = field(default=None, init=False)
    candidate_since: float | None = field(default=None, init=False)
    last_observed: float | None = field(default=None, init=False)
    state_since: float | None = field(default=None, init=False)

    def __post_init__(self):
        for value in (self.occupied_seconds, self.empty_seconds, self.max_gap_seconds):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("Time thresholds must be finite and greater than zero.")

    def invalidate(self):
        self.state = "UNKNOWN"
        self.candidate = self.candidate_since = self.last_observed = self.state_since = None

    def update(self, detected: bool, now: float) -> str:
        if not math.isfinite(now):
            raise ValueError("Observation time must be finite.")
        if self.last_observed is not None:
            gap = now - self.last_observed
            if gap < 0 or gap > self.max_gap_seconds:
                self.invalidate()
        self.last_observed = now
        desired = "OCCUPIED" if detected else "EMPTY"
        if desired == self.state:
            self.candidate = self.candidate_since = None
            return self.state
        if self.candidate != desired:
            self.candidate, self.candidate_since = desired, now
        threshold = self.occupied_seconds if detected else self.empty_seconds
        if now - self.candidate_since >= threshold:
            self.state, self.state_since = desired, now
            self.candidate = self.candidate_since = None
        return self.state
