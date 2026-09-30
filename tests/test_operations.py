from pathlib import Path
import tempfile
import unittest

from seat_monitor.core import (
    DeskObject,
    PhysicalState,
    Reservation,
    Seat,
    SeatMonitorCore,
    SeatOperationalTracker,
    SeatStatus,
)


SEAT = Seat("A01", ((0, 0), (1, 0), (1, 1), (0, 1)))
BOOK = DeskObject((0.2, 0.2, 0.4, 0.4), "book", 0.8)
MOVED_BOOK = DeskObject((0.6, 0.2, 0.8, 0.4), "book", 0.8)


class OperationalTrackerTests(unittest.TestCase):
    def test_unreserved_occupancy_uses_elapsed_time_not_frame_count(self):
        tracker = SeatOperationalTracker("A01", max_gap_seconds=120)
        for _ in range(100):
            tracker.observe(True, [], 0, 0.9)
            self.assertEqual(tracker.status(None, 0), SeatStatus.OCCUPANCY_PENDING)
        tracker.observe(True, [], 59, 0.9)
        self.assertEqual(tracker.status(None, 59), SeatStatus.OCCUPANCY_PENDING)
        tracker.observe(True, [], 60, 0.9)
        self.assertEqual(tracker.status(None, 60), SeatStatus.UNRESERVED_OCCUPIED)

    def test_no_show_means_reserved_but_not_checked_in(self):
        tracker = SeatOperationalTracker("A01", max_gap_seconds=1000)
        reservation = Reservation(1, "A01", "session-token", 0, 7200)
        tracker.observe(False, [], 0)
        self.assertEqual(tracker.status(reservation, 899), SeatStatus.RESERVED_WAITING)
        self.assertEqual(tracker.status(reservation, 900), SeatStatus.NO_SHOW)

    def test_stable_belongings_use_time_and_movement_restarts_timer(self):
        tracker = SeatOperationalTracker("A01", max_gap_seconds=601)
        tracker.observe(False, [BOOK], 0)
        tracker.observe(False, [BOOK], 600)
        self.assertEqual(tracker.physical_state, PhysicalState.AWAY_WITH_BELONGINGS)
        tracker.observe(False, [MOVED_BOOK], 1200)
        self.assertEqual(tracker.physical_state, PhysicalState.ITEMS_PRESENT)
        tracker.observe(False, [MOVED_BOOK], 1800)
        self.assertEqual(tracker.physical_state, PhysicalState.AWAY_WITH_BELONGINGS)
        tracker.observe(False, [MOVED_BOOK], 2400)
        tracker.observe(False, [MOVED_BOOK], 3000)
        self.assertEqual(tracker.physical_state, PhysicalState.CLEANUP_PENDING)


class CoreIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temporary.name) / "seat-monitor.db"

    def tearDown(self):
        self.temporary.cleanup()

    def test_reservation_does_not_claim_physical_occupancy(self):
        core = SeatMonitorCore([SEAT], self.db_path, max_gap_seconds=1000)
        core.db.create_reservation("A01", "session-token", 0, 7200)
        status = core.feed_seat_status("A01", False, [], timestamp=0)
        view = core.get_user_seat_map(timestamp=0)[0]
        self.assertEqual(status, SeatStatus.RESERVED_WAITING)
        self.assertEqual(view["physical_state"], PhysicalState.EMPTY)
        self.assertFalse(view["has_person"])

    def test_check_in_and_person_presence_produce_in_use(self):
        core = SeatMonitorCore([SEAT], self.db_path, max_gap_seconds=1000)
        core.db.create_reservation("A01", "session-token", 0, 7200)
        self.assertTrue(core.check_in_seat("A01", "session-token", timestamp=10)["success"])
        self.assertEqual(
            core.feed_seat_status("A01", True, [], timestamp=10),
            SeatStatus.IN_USE,
        )

    def test_cleanup_alert_is_idempotent_and_state_survives_restart(self):
        core = SeatMonitorCore(
            [SEAT], self.db_path, away_seconds=600, cleanup_seconds=1800,
            max_gap_seconds=601,
        )
        for timestamp in (0, 600, 1200, 1800):
            core.feed_seat_status("A01", False, [BOOK], timestamp=timestamp)
        first = core.request_cleanup("A01", timestamp=1800)
        second = core.request_cleanup("A01", timestamp=1800)
        self.assertTrue(first["admin_notified"])
        self.assertFalse(second["admin_notified"])
        self.assertEqual(len(core.get_pending_alerts()), 1)

        restarted = SeatMonitorCore(
            [SEAT], self.db_path, away_seconds=600, cleanup_seconds=1800,
            max_gap_seconds=601,
        )
        view = restarted.get_admin_dashboard(timestamp=1800)["seats"][0]
        self.assertEqual(view["physical_state"], PhysicalState.CLEANUP_PENDING)
        self.assertEqual(view["status"], SeatStatus.CLEANUP_REQUESTED)


if __name__ == "__main__":
    unittest.main()
