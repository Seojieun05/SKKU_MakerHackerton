# SPDX-License-Identifier: AGPL-3.0-only
"""Seat geometry, time filtering, and reservation/occupancy reconciliation.

Camera primitives remain dependency-free. The optional SeatMonitorCore stores
metadata only (never images) in local SQLite for a user/admin web application.
"""

from __future__ import annotations

from contextlib import closing
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import sqlite3
from typing import Any

Point = tuple[float, float]
Box = tuple[float, float, float, float]


@dataclass(frozen=True)
class Seat:
    id: str
    polygon: tuple[Point, ...]
    desk_polygon: tuple[Point, ...] | None = None

    @property
    def center(self) -> Point:
        return (
            sum(p[0] for p in self.polygon) / len(self.polygon),
            sum(p[1] for p in self.polygon) / len(self.polygon),
        )

    @property
    def chair_roi(self) -> tuple[Point, ...]:
        return self.polygon

    @property
    def desk_roi(self) -> tuple[Point, ...]:
        return self.desk_polygon if self.desk_polygon is not None else self.polygon


@dataclass(frozen=True)
class Detection:
    # Normalized coordinates in the original camera frame, not letterboxed input.
    box: Box
    confidence: float
    label: str = "person"
    class_id: int = 0

    @property
    def anchor(self) -> Point:
        """Box centre used by the existing calibrated chair ROIs."""
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def bottom_anchor(self) -> Point:
        """Optional foot/seat-side anchor for a future calibration mode."""
        x1, _, x2, y2 = self.box
        return ((x1 + x2) / 2, y2)


@dataclass(frozen=True)
class DeskObject:
    box: Box
    label: str = "item"
    confidence: float = 0.5

    @property
    def center(self) -> Point:
        x1, y1, x2, y2 = self.box
        return ((x1 + x2) / 2, (y1 + y2) / 2)

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.box
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)


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


def _nearest_seat(seats: list[Seat], point: Point, *, desk: bool = False) -> Seat | None:
    candidates = [
        seat for seat in seats
        if contains(seat.desk_roi if desk else seat.chair_roi, point)
    ]
    return min(candidates, key=lambda seat: math.dist(seat.center, point)) if candidates else None


def assign_people(seats: list[Seat], detections: list[Detection]) -> dict[str, float]:
    """Assign every person to at most one chair and ignore people in aisles."""
    scores = {seat.id: 0.0 for seat in seats}
    for detection in detections:
        if detection.label != "person":
            continue
        nearest = _nearest_seat(seats, detection.anchor)
        if nearest is not None:
            scores[nearest.id] = max(scores[nearest.id], detection.confidence)
    return scores


def assign_desk_objects(seats: list[Seat], detections: list[Detection]) -> dict[str, list[DeskObject]]:
    """Assign every non-person detection to at most one desk region."""
    assigned: dict[str, list[DeskObject]] = {seat.id: [] for seat in seats}
    for detection in detections:
        if detection.label == "person":
            continue
        nearest = _nearest_seat(seats, detection.anchor, desk=True)
        if nearest is not None:
            assigned[nearest.id].append(DeskObject(detection.box, detection.label, detection.confidence))
    return assigned


def compare_desk_objects(
    previous: list[DeskObject],
    current: list[DeskObject],
    distance_threshold: float = 0.08,
) -> bool:
    """Return whether two non-empty sets of object boxes are spatially similar."""
    if not math.isfinite(distance_threshold) or distance_threshold <= 0:
        raise ValueError("distance_threshold must be finite and greater than zero.")
    if not previous or not current or abs(len(previous) - len(current)) > 1:
        return False
    matched = 0
    used: set[int] = set()
    for old in previous:
        candidates = [
            (math.dist(old.center, new.center), index)
            for index, new in enumerate(current)
            if index not in used
        ]
        if not candidates:
            continue
        distance, index = min(candidates)
        new = current[index]
        larger_area = max(old.area, new.area)
        area_is_similar = larger_area == 0 or min(old.area, new.area) / larger_area >= 0.5
        if distance <= distance_threshold and area_is_similar:
            used.add(index)
            matched += 1
    required = max(1, math.ceil(max(len(previous), len(current)) * 0.7))
    return matched >= required


@dataclass
class SeatState:
    """Debounce the low-level person-present signal used by the current CLI."""

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


class PhysicalState:
    UNKNOWN = "UNKNOWN"
    EMPTY = "EMPTY"
    PERSON_PRESENT = "PERSON_PRESENT"
    ITEMS_PRESENT = "ITEMS_PRESENT"
    AWAY_WITH_BELONGINGS = "AWAY_WITH_BELONGINGS"
    CLEANUP_PENDING = "CLEANUP_PENDING"


class ReservationState:
    NONE = "NONE"
    RESERVED = "RESERVED"
    CHECKED_IN = "CHECKED_IN"
    NO_SHOW = "NO_SHOW"


class SeatStatus:
    """Combined status for the UI; physical and reservation states stay separate."""

    UNKNOWN = "UNKNOWN"
    AVAILABLE = "AVAILABLE"
    RESERVED_WAITING = "RESERVED_WAITING"
    IN_USE = "IN_USE"
    CHECKED_IN_AWAY = "CHECKED_IN_AWAY"
    OCCUPANCY_PENDING = "OCCUPANCY_PENDING"
    UNRESERVED_OCCUPIED = "UNRESERVED_OCCUPIED"
    RESERVATION_UNVERIFIED_OCCUPIED = "RESERVATION_UNVERIFIED_OCCUPIED"
    ITEMS_PRESENT = "ITEMS_PRESENT"
    AWAY_WITH_BELONGINGS = "AWAY_WITH_BELONGINGS"
    NO_SHOW = "NO_SHOW"
    CLEANUP_PENDING = "CLEANUP_PENDING"
    CLEANUP_REQUESTED = "CLEANUP_REQUESTED"


@dataclass(frozen=True)
class Reservation:
    id: int
    seat_id: str
    user_token: str
    starts_at: float
    ends_at: float
    checked_in_at: float | None = None

    def state_at(self, now: float, no_show_seconds: float) -> str:
        if self.checked_in_at is not None:
            return ReservationState.CHECKED_IN
        if now >= self.starts_at + no_show_seconds:
            return ReservationState.NO_SHOW
        return ReservationState.RESERVED


def _valid_time(value: float, name: str = "time") -> float:
    value = float(value)
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite.")
    return value


@dataclass
class SeatOperationalTracker:
    """Track continuous physical conditions using elapsed time, never frame count."""

    seat_id: str
    unreserved_grace_seconds: float = 60.0
    away_seconds: float = 600.0
    cleanup_seconds: float = 1800.0
    no_show_seconds: float = 900.0
    max_gap_seconds: float = 120.0
    physical_state: str = field(default=PhysicalState.UNKNOWN, init=False)
    person_since: float | None = field(default=None, init=False)
    belongings_since: float | None = field(default=None, init=False)
    last_observed: float | None = field(default=None, init=False)
    last_has_person: bool | None = field(default=None, init=False)
    last_person_confidence: float = field(default=0.0, init=False)
    last_objects: list[DeskObject] = field(default_factory=list, init=False)
    cleanup_requested: bool = field(default=False, init=False)

    def __post_init__(self):
        thresholds = (
            self.unreserved_grace_seconds,
            self.away_seconds,
            self.cleanup_seconds,
            self.no_show_seconds,
            self.max_gap_seconds,
        )
        if any(not math.isfinite(value) or value <= 0 for value in thresholds):
            raise ValueError("Time thresholds must be finite and greater than zero.")
        if self.cleanup_seconds <= self.away_seconds:
            raise ValueError("cleanup_seconds must be greater than away_seconds.")

    def restore(self, row: dict[str, Any]):
        self.physical_state = row["physical_state"]
        self.person_since = row["person_since"]
        self.belongings_since = row["belongings_since"]
        self.last_observed = row["observed_at"]
        self.last_has_person = row["has_person"]
        self.last_person_confidence = row["person_confidence"]
        self.last_objects = row["desk_objects"]

    def observe(
        self,
        has_person: bool,
        objects: list[DeskObject],
        now: float,
        person_confidence: float = 0.0,
    ) -> str:
        now = _valid_time(now, "Observation time")
        if self.last_observed is not None:
            gap = now - self.last_observed
            if gap < 0 or gap > self.max_gap_seconds:
                self.person_since = self.belongings_since = None
        if has_person:
            if self.person_since is None:
                self.person_since = now
            self.belongings_since = None
            self.physical_state = PhysicalState.PERSON_PRESENT
            self.cleanup_requested = False
        elif objects:
            self.person_since = None
            if self.belongings_since is None:
                self.belongings_since = now
            elif self.last_objects and not compare_desk_objects(self.last_objects, objects):
                # A changed layout is new evidence, not continuation of the old
                # unattended-belongings interval.
                self.belongings_since = now
            elapsed = now - self.belongings_since
            if elapsed >= self.cleanup_seconds:
                self.physical_state = PhysicalState.CLEANUP_PENDING
            elif elapsed >= self.away_seconds:
                self.physical_state = PhysicalState.AWAY_WITH_BELONGINGS
            else:
                self.physical_state = PhysicalState.ITEMS_PRESENT
        else:
            self.person_since = self.belongings_since = None
            self.physical_state = PhysicalState.EMPTY
            self.cleanup_requested = False
        self.last_observed = now
        self.last_has_person = bool(has_person)
        self.last_person_confidence = float(person_confidence)
        self.last_objects = list(objects)
        return self.physical_state

    def status(self, reservation: Reservation | None, now: float) -> str:
        now = _valid_time(now)
        if self.last_observed is None or self.physical_state == PhysicalState.UNKNOWN:
            return SeatStatus.UNKNOWN
        if self.cleanup_requested and self.physical_state == PhysicalState.CLEANUP_PENDING:
            return SeatStatus.CLEANUP_REQUESTED
        if self.physical_state == PhysicalState.CLEANUP_PENDING:
            return SeatStatus.CLEANUP_PENDING
        if self.physical_state == PhysicalState.AWAY_WITH_BELONGINGS:
            return SeatStatus.AWAY_WITH_BELONGINGS
        if self.physical_state == PhysicalState.ITEMS_PRESENT:
            return SeatStatus.ITEMS_PRESENT
        reservation_state = reservation.state_at(now, self.no_show_seconds) if reservation else ReservationState.NONE
        if self.physical_state == PhysicalState.PERSON_PRESENT:
            if reservation_state == ReservationState.CHECKED_IN:
                return SeatStatus.IN_USE
            if reservation is not None:
                return SeatStatus.RESERVATION_UNVERIFIED_OCCUPIED
            if self.person_since is not None and now - self.person_since >= self.unreserved_grace_seconds:
                return SeatStatus.UNRESERVED_OCCUPIED
            return SeatStatus.OCCUPANCY_PENDING
        if reservation_state == ReservationState.NO_SHOW:
            return SeatStatus.NO_SHOW
        if reservation_state == ReservationState.CHECKED_IN:
            return SeatStatus.CHECKED_IN_AWAY
        if reservation_state == ReservationState.RESERVED:
            return SeatStatus.RESERVED_WAITING
        return SeatStatus.AVAILABLE


def _iso_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class SeatDatabase:
    """SQLite metadata store. No frame or face data is accepted by this API."""

    def __init__(self, db_path: str | Path = "seat_monitor.db"):
        self.db_path = str(db_path)
        self._init_tables()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_tables(self):
        with closing(self._connect()) as connection:
            connection.executescript(
                """
                CREATE TABLE IF NOT EXISTS reservations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    seat_id TEXT NOT NULL,
                    user_token TEXT NOT NULL,
                    starts_at REAL NOT NULL,
                    ends_at REAL NOT NULL,
                    checked_in_at REAL,
                    cancelled_at REAL,
                    created_at TEXT NOT NULL,
                    CHECK (ends_at > starts_at)
                );
                CREATE INDEX IF NOT EXISTS reservations_by_seat_time
                    ON reservations(seat_id, starts_at, ends_at);
                CREATE TABLE IF NOT EXISTS operational_observations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    seat_id TEXT NOT NULL,
                    observed_at REAL NOT NULL,
                    recorded_at TEXT NOT NULL,
                    has_person INTEGER NOT NULL,
                    person_confidence REAL NOT NULL,
                    desk_objects_json TEXT NOT NULL,
                    physical_state TEXT NOT NULL,
                    person_since REAL,
                    belongings_since REAL
                );
                CREATE INDEX IF NOT EXISTS observations_by_seat_time
                    ON operational_observations(seat_id, observed_at DESC);
                CREATE TABLE IF NOT EXISTS admin_alerts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    seat_id TEXT NOT NULL,
                    alert_type TEXT NOT NULL,
                    message TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    is_resolved INTEGER NOT NULL DEFAULT 0,
                    resolved_at TEXT
                );
                """
            )
            connection.commit()

    @staticmethod
    def _reservation(row: sqlite3.Row) -> Reservation:
        return Reservation(row["id"], row["seat_id"], row["user_token"], row["starts_at"], row["ends_at"], row["checked_in_at"])

    def create_reservation(self, seat_id: str, user_token: str, starts_at: float, ends_at: float) -> Reservation:
        starts_at = _valid_time(starts_at, "starts_at")
        ends_at = _valid_time(ends_at, "ends_at")
        if not user_token:
            raise ValueError("user_token must not be empty.")
        if ends_at <= starts_at:
            raise ValueError("ends_at must be later than starts_at.")
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            conflict = connection.execute(
                "SELECT 1 FROM reservations WHERE seat_id=? AND cancelled_at IS NULL AND starts_at < ? AND ends_at > ? LIMIT 1",
                (seat_id, ends_at, starts_at),
            ).fetchone()
            if conflict:
                raise ValueError("The seat already has an overlapping reservation.")
            cursor = connection.execute(
                "INSERT INTO reservations (seat_id,user_token,starts_at,ends_at,created_at) VALUES (?,?,?,?,?)",
                (seat_id, user_token, starts_at, ends_at, _iso_now()),
            )
            reservation_id = int(cursor.lastrowid)
            connection.commit()
        return Reservation(reservation_id, seat_id, user_token, starts_at, ends_at)

    def get_active_reservation(self, seat_id: str, now: float) -> Reservation | None:
        now = _valid_time(now)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM reservations WHERE seat_id=? AND cancelled_at IS NULL AND starts_at <= ? AND ends_at > ? ORDER BY starts_at DESC LIMIT 1",
                (seat_id, now, now),
            ).fetchone()
        return self._reservation(row) if row else None

    def check_in(self, seat_id: str, user_token: str, now: float) -> Reservation:
        now = _valid_time(now)
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM reservations WHERE seat_id=? AND user_token=? AND cancelled_at IS NULL AND starts_at <= ? AND ends_at > ? ORDER BY starts_at DESC LIMIT 1",
                (seat_id, user_token, now, now),
            ).fetchone()
            if row is None:
                raise ValueError("No active reservation matches this seat and user token.")
            connection.execute("UPDATE reservations SET checked_in_at=COALESCE(checked_in_at,?) WHERE id=?", (now, row["id"]))
            connection.commit()
            updated = connection.execute("SELECT * FROM reservations WHERE id=?", (row["id"],)).fetchone()
        return self._reservation(updated)

    def cancel_reservation(self, reservation_id: int, now: float) -> bool:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE reservations SET cancelled_at=? WHERE id=? AND cancelled_at IS NULL",
                (_valid_time(now), reservation_id),
            )
            connection.commit()
            return cursor.rowcount > 0

    def save_observation(self, tracker: SeatOperationalTracker):
        if tracker.last_observed is None:
            return
        payload = json.dumps([asdict(item) for item in tracker.last_objects], separators=(",", ":"))
        with closing(self._connect()) as connection:
            connection.execute(
                "INSERT INTO operational_observations (seat_id,observed_at,recorded_at,has_person,person_confidence,desk_objects_json,physical_state,person_since,belongings_since) VALUES (?,?,?,?,?,?,?,?,?)",
                (
                    tracker.seat_id, tracker.last_observed, _iso_now(), int(bool(tracker.last_has_person)),
                    tracker.last_person_confidence, payload, tracker.physical_state,
                    tracker.person_since, tracker.belongings_since,
                ),
            )
            connection.commit()

    def get_last_observation(self, seat_id: str) -> dict[str, Any] | None:
        with closing(self._connect()) as connection:
            row = connection.execute(
                "SELECT * FROM operational_observations WHERE seat_id=? ORDER BY observed_at DESC,id DESC LIMIT 1",
                (seat_id,),
            ).fetchone()
        if row is None:
            return None
        result = dict(row)
        result["has_person"] = bool(result["has_person"])
        result["desk_objects"] = [
            DeskObject(tuple(item["box"]), item["label"], item["confidence"])
            for item in json.loads(result.pop("desk_objects_json"))
        ]
        return result

    def create_alert_once(self, seat_id: str, alert_type: str, message: str) -> tuple[int, bool]:
        with closing(self._connect()) as connection:
            connection.execute("BEGIN IMMEDIATE")
            existing = connection.execute(
                "SELECT id FROM admin_alerts WHERE seat_id=? AND alert_type=? AND is_resolved=0 ORDER BY id DESC LIMIT 1",
                (seat_id, alert_type),
            ).fetchone()
            if existing:
                connection.commit()
                return int(existing["id"]), False
            cursor = connection.execute(
                "INSERT INTO admin_alerts (seat_id,alert_type,message,created_at,is_resolved) VALUES (?,?,?,?,0)",
                (seat_id, alert_type, message, _iso_now()),
            )
            alert_id = int(cursor.lastrowid)
            connection.commit()
            return alert_id, True

    def unresolved_alerts(self) -> list[dict[str, Any]]:
        with closing(self._connect()) as connection:
            rows = connection.execute("SELECT * FROM admin_alerts WHERE is_resolved=0 ORDER BY id").fetchall()
        return [dict(row) for row in rows]

    def resolve_alerts(self, seat_id: str) -> int:
        with closing(self._connect()) as connection:
            cursor = connection.execute(
                "UPDATE admin_alerts SET is_resolved=1,resolved_at=? WHERE seat_id=? AND is_resolved=0",
                (_iso_now(), seat_id),
            )
            connection.commit()
            return cursor.rowcount


class SeatMonitorCore:
    """Metadata-only integration API for the camera process and a local web app."""

    def __init__(
        self,
        seats: list[Seat],
        db_path: str | Path = "seat_monitor.db",
        *,
        unreserved_grace_seconds: float = 60.0,
        away_seconds: float = 600.0,
        cleanup_seconds: float = 1800.0,
        no_show_seconds: float = 900.0,
        max_gap_seconds: float = 120.0,
        persistence_interval_seconds: float = 60.0,
    ):
        if not seats or len({seat.id for seat in seats}) != len(seats):
            raise ValueError("At least one seat with a unique ID is required.")
        self.seats = {seat.id: seat for seat in seats}
        self.db = SeatDatabase(db_path)
        if not math.isfinite(persistence_interval_seconds) or persistence_interval_seconds <= 0:
            raise ValueError("persistence_interval_seconds must be finite and greater than zero.")
        self.persistence_interval_seconds = persistence_interval_seconds
        self._last_persisted: dict[str, float | None] = {seat.id: None for seat in seats}
        self.trackers = {
            seat.id: SeatOperationalTracker(
                seat.id, unreserved_grace_seconds, away_seconds,
                cleanup_seconds, no_show_seconds, max_gap_seconds,
            )
            for seat in seats
        }
        for seat_id, tracker in self.trackers.items():
            previous = self.db.get_last_observation(seat_id)
            if previous:
                tracker.restore(previous)
                self._last_persisted[seat_id] = previous["observed_at"]
        requested_cleanup_seats = {
            alert["seat_id"]
            for alert in self.db.unresolved_alerts()
            if alert["alert_type"] == "CLEANUP_REQUIRED"
        }
        for seat_id in requested_cleanup_seats & self.trackers.keys():
            self.trackers[seat_id].cleanup_requested = True

    @staticmethod
    def _now(timestamp: float | None) -> float:
        return datetime.now(timezone.utc).timestamp() if timestamp is None else _valid_time(timestamp)

    def _tracker(self, seat_id: str) -> SeatOperationalTracker:
        try:
            return self.trackers[seat_id]
        except KeyError as error:
            raise ValueError(f"Unknown seat: {seat_id}") from error

    def feed_detections(self, detections: list[Detection], timestamp: float | None = None) -> dict[str, str]:
        now = self._now(timestamp)
        seats = list(self.seats.values())
        people = assign_people(seats, detections)
        items = assign_desk_objects(seats, detections)
        return {
            seat.id: self.feed_seat_status(seat.id, people[seat.id] > 0, items[seat.id], people[seat.id], now)
            for seat in seats
        }

    def feed_seat_status(
        self,
        seat_id: str,
        has_person: bool,
        desk_objects: list[DeskObject],
        person_confidence: float = 0.0,
        timestamp: float | None = None,
    ) -> str:
        now = self._now(timestamp)
        tracker = self._tracker(seat_id)
        previous_physical_state = tracker.physical_state
        tracker.observe(has_person, desk_objects, now, person_confidence)
        last_saved = self._last_persisted[seat_id]
        if (
            last_saved is None
            or now < last_saved
            or now - last_saved >= self.persistence_interval_seconds
            or tracker.physical_state != previous_physical_state
        ):
            self.db.save_observation(tracker)
            self._last_persisted[seat_id] = now
        return tracker.status(self.db.get_active_reservation(seat_id, now), now)

    def reserve_seat(
        self,
        seat_id: str,
        user_token: str,
        starts_at: float,
        ends_at: float,
        *,
        timestamp: float | None = None,
    ) -> dict[str, Any]:
        tracker = self._tracker(seat_id)
        now = self._now(timestamp)
        status = tracker.status(self.db.get_active_reservation(seat_id, now), now)
        if status in (SeatStatus.CLEANUP_PENDING, SeatStatus.CLEANUP_REQUESTED):
            return {"success": False, "seat_id": seat_id, "status": status, "user_message": "자리 정리가 필요한 좌석입니다."}
        try:
            reservation = self.db.create_reservation(seat_id, user_token, starts_at, ends_at)
        except ValueError as error:
            return {"success": False, "seat_id": seat_id, "user_message": str(error)}
        return {
            "success": True, "seat_id": seat_id, "reservation_id": reservation.id,
            "status": "RESERVED", "user_message": f"좌석 {seat_id} 예약이 완료되었습니다. 현장에서 체크인해 주세요.",
        }

    def check_in_seat(self, seat_id: str, user_token: str, timestamp: float | None = None) -> dict[str, Any]:
        now = self._now(timestamp)
        self._tracker(seat_id)
        try:
            reservation = self.db.check_in(seat_id, user_token, now)
        except ValueError as error:
            return {"success": False, "seat_id": seat_id, "user_message": str(error)}
        return {
            "success": True, "seat_id": seat_id, "reservation_id": reservation.id,
            "status": "CHECKED_IN", "user_message": f"좌석 {seat_id} 체크인이 완료되었습니다.",
        }

    def request_cleanup(self, seat_id: str, timestamp: float | None = None) -> dict[str, Any]:
        now = self._now(timestamp)
        tracker = self._tracker(seat_id)
        reservation = self.db.get_active_reservation(seat_id, now)
        status = tracker.status(reservation, now)
        if status not in (SeatStatus.CLEANUP_PENDING, SeatStatus.CLEANUP_REQUESTED, SeatStatus.NO_SHOW):
            return {"success": False, "seat_id": seat_id, "user_message": "아직 관리자 정리 요청 대상 좌석이 아닙니다."}
        if tracker.physical_state == PhysicalState.CLEANUP_PENDING:
            tracker.cleanup_requested = True
            alert_type = "CLEANUP_REQUIRED"
            message = f"[좌석 {seat_id}] 30분 이상 사람 없이 소지품만 감지되어 자리 확인이 필요합니다."
        else:
            alert_type = "NO_SHOW_REVIEW"
            message = f"[좌석 {seat_id}] 예약 시작 후 15분 동안 체크인되지 않아 좌석 확인이 필요합니다."
        alert_id, created = self.db.create_alert_once(seat_id, alert_type, message)
        return {
            "success": True, "seat_id": seat_id, "status": SeatStatus.CLEANUP_REQUESTED,
            "alert_id": alert_id, "admin_notified": created,
            "user_message": "관리자에게 자리 확인 요청을 전달했습니다.",
        }

    def select_seat_by_user(
        self, seat_id: str, user_id: str = "guest_user", *,
        timestamp: float | None = None, duration_seconds: float = 7200.0,
    ) -> dict[str, Any]:
        now = self._now(timestamp)
        tracker = self._tracker(seat_id)
        status = tracker.status(self.db.get_active_reservation(seat_id, now), now)
        if status in (SeatStatus.CLEANUP_PENDING, SeatStatus.CLEANUP_REQUESTED, SeatStatus.NO_SHOW):
            return self.request_cleanup(seat_id, now)
        return self.reserve_seat(seat_id, user_id, now, now + duration_seconds, timestamp=now)

    def _seat_view(self, seat_id: str, now: float) -> dict[str, Any]:
        tracker = self._tracker(seat_id)
        reservation = self.db.get_active_reservation(seat_id, now)
        status = tracker.status(reservation, now)
        reservation_state = reservation.state_at(now, tracker.no_show_seconds) if reservation else ReservationState.NONE
        colors = {
            SeatStatus.AVAILABLE: "GREEN", SeatStatus.RESERVED_WAITING: "BLUE",
            SeatStatus.IN_USE: "BLUE", SeatStatus.CHECKED_IN_AWAY: "BLUE",
            SeatStatus.OCCUPANCY_PENDING: "ORANGE", SeatStatus.UNRESERVED_OCCUPIED: "ORANGE",
            SeatStatus.RESERVATION_UNVERIFIED_OCCUPIED: "ORANGE", SeatStatus.ITEMS_PRESENT: "ORANGE",
            SeatStatus.AWAY_WITH_BELONGINGS: "YELLOW", SeatStatus.NO_SHOW: "YELLOW",
            SeatStatus.CLEANUP_PENDING: "YELLOW", SeatStatus.CLEANUP_REQUESTED: "YELLOW",
        }
        requires_admin = status in (SeatStatus.NO_SHOW, SeatStatus.CLEANUP_PENDING, SeatStatus.CLEANUP_REQUESTED)
        return {
            "seat_id": seat_id, "status": status, "physical_state": tracker.physical_state,
            "reservation_state": reservation_state, "display_color": colors.get(status, "GRAY"),
            "has_person": tracker.last_has_person, "desk_items_count": len(tracker.last_objects),
            "last_observed": tracker.last_observed,
            "is_selectable": status == SeatStatus.AVAILABLE or requires_admin,
            "requires_admin": requires_admin,
        }

    def get_user_seat_map(self, timestamp: float | None = None) -> list[dict[str, Any]]:
        now = self._now(timestamp)
        return [self._seat_view(seat_id, now) for seat_id in self.seats]

    def get_admin_dashboard(self, timestamp: float | None = None) -> dict[str, Any]:
        now = self._now(timestamp)
        seats = [self._seat_view(seat_id, now) for seat_id in self.seats]
        alerts = self.db.unresolved_alerts()
        return {
            "total_seats": len(seats),
            "unreserved_occupied_count": sum(seat["status"] == SeatStatus.UNRESERVED_OCCUPIED for seat in seats),
            "attention_seats_count": sum(seat["requires_admin"] for seat in seats),
            "pending_alerts_count": len(alerts), "pending_alerts": alerts, "seats": seats,
        }

    def get_pending_alerts(self) -> list[dict[str, Any]]:
        return self.db.unresolved_alerts()

    def resolve_cleanup(self, seat_id: str, admin_id: str = "admin", timestamp: float | None = None) -> dict[str, Any]:
        del admin_id  # Reserved for the web layer's audit log.
        now = self._now(timestamp)
        tracker = self._tracker(seat_id)
        reservation = self.db.get_active_reservation(seat_id, now)
        if reservation and reservation.checked_in_at is None:
            self.db.cancel_reservation(reservation.id, now)
        tracker.observe(False, [], now, 0.0)
        self.db.save_observation(tracker)
        self._last_persisted[seat_id] = now
        resolved = self.db.resolve_alerts(seat_id)
        return {
            "success": True, "seat_id": seat_id, "status": SeatStatus.AVAILABLE,
            "resolved_alerts_count": resolved, "message": f"좌석 {seat_id} 정리 완료를 기록했습니다.",
        }
