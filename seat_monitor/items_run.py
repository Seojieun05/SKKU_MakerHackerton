# SPDX-License-Identifier: AGPL-3.0-only
"""Optional person and selected personal-item detection; writes no images."""
from contextlib import redirect_stdout
from datetime import datetime, timezone
from pathlib import Path
import json
import math
import signal
import sys
import time

from .__main__ import parser, positive_int
from .camera import FrameSource
from .config import load_config
from .core import Detection, SeatState, assign_people
from .detector import PersonDetector
from .status import snapshot, write_json

ITEM_NAMES = frozenset(('laptop', 'backpack', 'handbag', 'suitcase', 'cup', 'book'))


class SceneDetector(PersonDetector):
    def detect_scene(self, frame, item_confidence):
        with redirect_stdout(sys.stderr):
            names = self.model.names
        names = dict(names.items() if isinstance(names, dict) else enumerate(names))
        if 'person' not in names.values():
            raise ValueError("Model has no person class")
        if not ITEM_NAMES.intersection(names.values()):
            raise ValueError("Model has no supported personal-item classes")
        selected = [i for i, name in names.items() if name == 'person' or name in ITEM_NAMES]
        with redirect_stdout(sys.stderr):
            result = self.model.predict(
                source=frame, classes=selected,
                conf=min(self.confidence, item_confidence),
                imgsz=self.imgsz, device='cpu', verbose=False,
                save=False, save_txt=False, save_crop=False, show=False, rect=False,
            )[0]
        people, items = [], []
        if result.boxes is not None:
            boxes = result.boxes.xyxyn.cpu().tolist()
            scores = result.boxes.conf.cpu().tolist()
            classes = result.boxes.cls.cpu().tolist()
            for box, score, cls in zip(boxes, scores, classes):
                name = names.get(int(cls))
                if name == 'person' and score >= self.confidence:
                    people.append(Detection(tuple(box), float(score)))
                elif name in ITEM_NAMES and score >= item_confidence:
                    items.append(Detection(tuple(box), float(score)))
        return people, items


def add_item_fields(data, item_states, item_scores, health):
    data['item_meaning'] = 'selected_personal_items'
    data['item_classes'] = sorted(ITEM_NAMES)
    for seat in data['seats']:
        sid = seat['seat_id']
        item = item_states[sid]
        usable = health == 'ok'
        item_state = item.state if usable else 'UNKNOWN'
        has_item = None if item_state == 'UNKNOWN' else item_state == 'OCCUPIED'
        seat.update(
            has_item=has_item,
            item_state=item_state,
            current_item_confidence=item_scores[sid] if usable else 0.0,
            item_pending_state=item.candidate if usable else None,
        )
        person = seat.get('has_person') if usable else None
        if person is True:
            presence = 'PERSON_PRESENT'
        elif person is False and has_item is True:
            presence = 'ITEMS_ONLY'
        elif person is False and has_item is False:
            presence = 'EMPTY'
        else:
            presence = 'UNKNOWN'
        seat['presence_state'] = presence
    return data


def run(args):
    if args.whole_frame or args.preview:
        raise ValueError('Use seat configs without --whole-frame or --preview')
    if args.fps * args.max_gap_seconds <= 1:
        raise ValueError('max-gap-seconds must exceed the inference interval')
    if not math.isfinite(args.item_confidence) or not 0 < args.item_confidence <= 1:
        raise ValueError('item-confidence must be greater than zero and at most one')
    if bool(args.sensor_width) != bool(args.sensor_height):
        raise ValueError('Provide both --sensor-width and --sensor-height')
    if args.sensor_width and args.source != 'picamera2':
        raise ValueError('Sensor mode options require --source picamera2')
    sensor_size = (args.sensor_width, args.sensor_height) if args.sensor_width else None
    seats, size = load_config(args.config)
    item_seats, item_size = load_config(args.items_config)
    if {s.id for s in seats} != {s.id for s in item_seats}:
        raise ValueError('Person and item seat IDs must match')
    if not size or not item_size or abs((size[0]/size[1])/(item_size[0]/item_size[1])-1) > .02:
        raise ValueError('Person and item calibration aspect ratios must match')
    states = {s.id: SeatState(args.occupied_seconds, args.empty_seconds, args.max_gap_seconds) for s in seats}
    item_states = {s.id: SeatState(2.0, args.empty_seconds, args.max_gap_seconds) for s in seats}
    scores = {s.id: 0.0 for s in seats}
    item_scores = scores.copy()
    now = time.monotonic()
    observation_wall = None
    final_health = 'stopped'

    def emit(health):
        data = snapshot(states, scores, now, health, args.max_gap_seconds,
                        observed_at=observation_wall if health == 'ok' else None)
        add_item_fields(data, item_states, item_scores, health)
        write_json(args.status_file, data)
        print(json.dumps(data, ensure_ascii=False), flush=True)

    def invalidate():
        for state in (*states.values(), *item_states.values()):
            state.invalidate()

    try:
        emit('starting')
        detector = SceneDetector(args.model, args.confidence, args.imgsz, args.threads)
        with FrameSource(args.source, args.width, args.height, args.camera_index,
                         sensor_size=sensor_size) as camera:
            next_sample = -math.inf
            analyzed = 0
            while True:
                frame = camera.read()
                if frame is None:
                    final_health = 'end_of_video'
                    break
                observed = camera.observation_time
                if observed < next_sample:
                    if not camera.is_video:
                        time.sleep(.002)
                    continue
                next_sample = observed + 1/args.fps
                observation_wall = datetime.now(timezone.utc)
                h, w = frame.shape[:2]
                if abs((w/h)/(size[0]/size[1])-1) > .02:
                    raise ValueError('Camera aspect ratio changed; recalibrate')
                start = time.monotonic()
                people, items = detector.detect_scene(frame, args.item_confidence)
                now = observed
                if time.monotonic() - start > args.max_gap_seconds:
                    invalidate()
                    scores = {s.id: 0.0 for s in seats}
                    item_scores = scores.copy()
                    health = 'inference_too_slow'
                else:
                    scores = assign_people(seats, people)
                    item_scores = assign_people(item_seats, items)
                    for sid in states:
                        states[sid].update(scores[sid] > 0, now)
                        item_states[sid].update(item_scores[sid] > 0, now)
                    health = 'ok'
                emit(health)
                analyzed += 1
                if args.max_frames and analyzed >= args.max_frames:
                    break
    except KeyboardInterrupt:
        pass
    except Exception:
        final_health = 'error'
        raise
    finally:
        invalidate()
        scores = {s.id: 0.0 for s in seats}
        item_scores = scores.copy()
        emit(final_health)
    return 0


def items_parser():
    root = parser()
    run_parser = root._subparsers._group_actions[0].choices['run']
    run_parser.add_argument('--items-config', type=Path, default=Path('config/items.json'))
    run_parser.add_argument('--item-confidence', type=float, default=.30)
    run_parser.add_argument('--sensor-width', type=positive_int)
    run_parser.add_argument('--sensor-height', type=positive_int)
    return root


def main(argv=None):
    root = items_parser()
    args = root.parse_args(argv)
    if args.command != 'run':
        root.error('This entry point supports run only')
    if args.camera_index < 0:
        root.error('camera-index must be non-negative')
    def stop(*_):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, stop)
    try:
        return run(args)
    except Exception as exc:
        print(f'Error: {exc}', file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
