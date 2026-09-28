import copy
from datetime import datetime, timedelta, timezone
import json
import math
from pathlib import Path
import tempfile
import unittest

from seat_monitor.config import config_document, load_config, parse_config
from seat_monitor.core import Detection, Seat, SeatState, assign_people, contains
from seat_monitor.status import snapshot, write_json

LEFT = Seat("A01", ((0, 0), (0.45, 0), (0.45, 1), (0, 1)))
RIGHT = Seat("A02", ((0.55, 0), (1, 0), (1, 1), (0.55, 1)))


class GeometryTests(unittest.TestCase):
    def test_person_is_assigned_to_correct_seat(self):
        detections = [Detection((0.1, 0.1, 0.3, 0.9), 0.85)]
        self.assertEqual(assign_people([LEFT, RIGHT], detections), {"A01": 0.85, "A02": 0})

    def test_aisle_box_overlapping_seats_is_not_occupancy(self):
        person_in_aisle = Detection((0.3, 0.1, 0.7, 0.9), 0.95)
        self.assertEqual(assign_people([LEFT, RIGHT], [person_in_aisle]), {"A01": 0, "A02": 0})

    def test_overlapping_rois_do_not_duplicate_a_person(self):
        a = Seat("a", ((0, 0), (0.65, 0), (0.65, 1), (0, 1)))
        b = Seat("b", ((0.35, 0), (1, 0), (1, 1), (0.35, 1)))
        scores = assign_people([a, b], [Detection((0.35, 0.1, 0.55, 0.9), 0.9)])
        self.assertEqual(sum(v > 0 for v in scores.values()), 1)
        self.assertEqual(scores["a"], 0.9)

    def test_boundary_is_inside(self):
        self.assertTrue(contains(LEFT.polygon, (0.45, 0.5)))
        self.assertFalse(contains(LEFT.polygon, (0.46, 0.5)))


class StateTests(unittest.TestCase):
    def test_startup_is_unknown_until_enough_observations(self):
        state = SeatState()
        for t in range(8):
            self.assertEqual(state.update(False, t), "UNKNOWN")
        self.assertEqual(state.update(False, 8), "EMPTY")

    def test_short_missed_detection_does_not_vacate_seat(self):
        state = SeatState()
        state.update(True, 0)
        self.assertEqual(state.update(True, 1), "OCCUPIED")
        state.update(False, 2)
        self.assertEqual(state.update(False, 5), "OCCUPIED")
        self.assertEqual(state.update(True, 6), "OCCUPIED")
        self.assertIsNone(state.candidate)

    def test_empty_requires_continuous_absence(self):
        state = SeatState(empty_seconds=3)
        state.update(True, 0)
        state.update(True, 1)
        state.update(False, 2)
        state.update(True, 4)
        state.update(False, 5)
        self.assertEqual(state.update(False, 7), "OCCUPIED")
        self.assertEqual(state.update(False, 8), "EMPTY")

    def test_passerby_does_not_change_empty_seat(self):
        state = SeatState(empty_seconds=1)
        state.update(False, 0)
        state.update(False, 1)
        self.assertEqual(state.update(True, 2), "EMPTY")
        self.assertEqual(state.update(False, 2.5), "EMPTY")

    def test_camera_gap_and_clock_reversal_restart_evidence(self):
        for second in (20, -1):
            with self.subTest(second=second):
                state = SeatState()
                state.update(True, 0)
                state.update(True, 1)
                self.assertEqual(state.update(False, second), "UNKNOWN")
                self.assertEqual(state.candidate_since, second)

    def test_invalid_time_thresholds(self):
        for value in (0, -1, math.inf, math.nan):
            with self.assertRaises(ValueError):
                SeatState(empty_seconds=value)


class ConfigAndStatusTests(unittest.TestCase):
    def test_config_round_trip(self):
        data = config_document([LEFT, RIGHT], (1280, 720))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "seats.json"
            write_json(path, data)
            self.assertEqual(load_config(path), ([LEFT, RIGHT], (1280, 720)))

    def test_reject_malformed_polygons(self):
        for polygon in (
            [[0, 0], [1, 1]],
            [[0, 0], [1, 1], [0, 1], [1, 0]],
            [[0, 0], [math.nan, 0], [1, 1]],
            [[0, 0], [2, 0], [1, 1]],
            [[0, 0], [0.5, 0], [1, 0]],
        ):
            data = copy.deepcopy(config_document([LEFT], (1280, 720)))
            data["seats"][0]["polygon"] = polygon
            with self.assertRaises(ValueError):
                parse_config(data)

    def test_reject_duplicate_seat_ids(self):
        with self.assertRaises(ValueError):
            parse_config(config_document([LEFT, LEFT], (1280, 720)))

    def test_status_is_metadata_only_and_unknown_is_not_false(self):
        state = SeatState()
        data = snapshot({"A01": state}, {"A01": 0}, now=0, health="error")
        seat = data["seats"][0]
        self.assertIsNone(seat["has_person"])
        self.assertEqual(seat["state"], "UNKNOWN")
        self.assertIn("valid_until", data)
        self.assertEqual(set(seat), {"seat_id", "state", "has_person", "current_person_confidence", "pending_state", "state_duration_seconds"})
        json.dumps(data, allow_nan=False)

    def test_freshness_uses_capture_time(self):
        captured = datetime(2026, 9, 28, tzinfo=timezone.utc)
        data = snapshot({"A01": SeatState()}, {}, 0, ttl_seconds=5, observed_at=captured)
        self.assertEqual(datetime.fromisoformat(data["valid_until"]), captured + timedelta(seconds=5))


if __name__ == "__main__":
    unittest.main()
