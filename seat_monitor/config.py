# SPDX-License-Identifier: AGPL-3.0-only
import json
import math
from pathlib import Path

from .core import Seat


def _parse_polygon(raw, seat_id: str, field_name: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw, list) or len(raw) < 3:
        raise ValueError(f"{seat_id}: {field_name} needs at least three vertices.")
    for point in raw:
        if not isinstance(point, list) or len(point) != 2 or any(
            type(value) not in (int, float) or not math.isfinite(value) or not 0 <= value <= 1
            for value in point
        ):
            raise ValueError(f"{seat_id}: {field_name} coordinates must be finite numbers between 0 and 1.")
    points = tuple(tuple(point) for point in raw)
    if len(set(points)) != len(points):
        raise ValueError(f"{seat_id}: duplicate {field_name} vertex.")
    crosses = []
    for index in range(len(points)):
        a, b, c = points[index - 2], points[index - 1], points[index]
        crosses.append((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]))
    if not (all(cross > 1e-10 for cross in crosses) or all(cross < -1e-10 for cross in crosses)):
        raise ValueError(f"{seat_id}: use a convex {field_name}, clicking corners in order.")
    direction = 1 if crosses[0] > 0 else -1
    for index, a in enumerate(points):
        b = points[(index + 1) % len(points)]
        for other_index, point in enumerate(points):
            if other_index not in (index, (index + 1) % len(points)):
                cross = (b[0] - a[0]) * (point[1] - a[1]) - (b[1] - a[1]) * (point[0] - a[0])
                if cross * direction <= 1e-10:
                    raise ValueError(f"{seat_id}: {field_name} intersects itself or is not convex.")
    return points


def parse_config(data: dict) -> tuple[list[Seat], tuple[int, int]]:
    if not isinstance(data, dict) or data.get("schema_version") != 1:
        raise ValueError("Seat config requires schema_version: 1.")
    size = data.get("frame_size")
    if not isinstance(size, list) or len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
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
        polygon = _parse_polygon(entry.get("polygon"), seat_id, "polygon")
        desk_raw = entry.get("desk_polygon")
        desk_polygon = _parse_polygon(desk_raw, seat_id, "desk_polygon") if desk_raw is not None else None
        seats.append(Seat(seat_id, polygon, desk_polygon))
        ids.add(seat_id)
    return seats, tuple(size)


def load_config(path: Path) -> tuple[list[Seat], tuple[int, int]]:
    if not path.is_file():
        raise FileNotFoundError(f"Seat config not found: {path}. Run python -m seat_monitor calibrate first.")
    return parse_config(json.loads(path.read_text(encoding="utf-8")))


def config_document(seats: list[Seat], size: tuple[int, int]) -> dict:
    entries = []
    for seat in seats:
        entry = {"id": seat.id, "polygon": [list(point) for point in seat.polygon]}
        if seat.desk_polygon is not None:
            entry["desk_polygon"] = [list(point) for point in seat.desk_polygon]
        entries.append(entry)
    return {"schema_version": 1, "frame_size": list(size), "seats": entries}
