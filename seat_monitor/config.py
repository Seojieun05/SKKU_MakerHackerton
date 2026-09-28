# SPDX-License-Identifier: AGPL-3.0-only
import json
import math
from pathlib import Path

from .core import Seat


def parse_config(data: dict) -> tuple[list[Seat], tuple[int, int]]:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Seat config requires schema_version: 1.")
    size = data.get("frame_size")
    if not isinstance(size, list) or len(size) != 2 or any(type(v) is not int or v <= 0 for v in size):
        raise ValueError("frame_size must be [width, height] with positive integers.")
    entries = data.get("seats")
    if not isinstance(entries, list) or not entries:
        raise ValueError("At least one seat is required. Run calibrate first.")
    seats, ids = [], set()
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("Each seat must be an object.")
        seat_id = entry.get("id")
        if not isinstance(seat_id, str) or not seat_id.strip() or seat_id in ids:
            raise ValueError("Seat IDs must be non-empty and unique.")
        polygon = entry.get("polygon")
        if not isinstance(polygon, list) or len(polygon) < 3:
            raise ValueError(f"{seat_id}: a polygon needs at least three vertices.")
        for p in polygon:
            if not isinstance(p, list) or len(p) != 2 or any(
                type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 1 for v in p
            ):
                raise ValueError(f"{seat_id}: coordinates must be finite numbers between 0 and 1.")
        # Accept convex polygons only. This catches bow-ties and accidental clicks.
        points = tuple(tuple(p) for p in polygon)
        if len(set(points)) != len(points):
            raise ValueError(f"{seat_id}: duplicate polygon vertex.")
        crosses = []
        for i in range(len(points)):
            a, b, c = points[i - 2], points[i - 1], points[i]
            crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
        if not (all(c > 1e-10 for c in crosses) or all(c < -1e-10 for c in crosses)):
            raise ValueError(f"{seat_id}: use a convex polygon, clicking corners in order.")
        # Every other vertex must lie on the same side of every polygon edge.
        direction = 1 if crosses[0] > 0 else -1
        for i, a in enumerate(points):
            b = points[(i + 1) % len(points)]
            for j, p in enumerate(points):
                if j not in (i, (i + 1) % len(points)):
                    cross = (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])
                    if cross * direction <= 1e-10:
                        raise ValueError(f"{seat_id}: polygon intersects itself or is not convex.")
        seats.append(Seat(seat_id, points))
        ids.add(seat_id)
    return seats, tuple(size)


def load_config(path: Path) -> tuple[list[Seat], tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Seat config not found: {path}. Run python -m seat_monitor calibrate first.")
    return parse_config(json.loads(path.read_text(encoding="utf-8")))


def config_document(seats: list[Seat], size: tuple[int, int]) -> dict:
    return {
        "schema_version": 1,
        "frame_size": list(size),
        "seats": [{"id": s.id, "polygon": [list(p) for p in s.polygon]} for s in seats],
    }
