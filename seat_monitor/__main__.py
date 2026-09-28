# SPDX-License-Identifier: AGPL-3.0-only
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import signal
import sys
import time

from .camera import FrameSource
from .config import load_config
from .core import Seat, SeatState, assign_people
from .detector import DEFAULT_MODEL, PersonDetector, prepare_model
from .status import snapshot, write_json


def positive(value):
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError("must be a finite number greater than zero")
    return number


def positive_int(value):
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return number


def probability(value):
    number = positive(value)
    if number > 1:
        raise argparse.ArgumentTypeError("must be greater than 0 and at most 1")
    return number


def parser():
    root = argparse.ArgumentParser(description="Raspberry Pi 5 local person/seat occupancy detector")
    commands = root.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="download the pretrained YOLO11n model once")
    prepare.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    prepare.add_argument("--ncnn", action="store_true", help="also export NCNN (install ncnn and pnnx first)")
    prepare.add_argument("--imgsz", type=positive_int, default=640)
    calibrate = commands.add_parser("calibrate", help="click seat polygon corners on a local preview")
    calibrate.add_argument("--overwrite", action="store_true")
    run = commands.add_parser("run", help="infer locally; write only seat state JSON")
    for command in (calibrate, run):
        command.add_argument("--source", default="picamera2", help="picamera2, USB index (0), or local video path")
        command.add_argument("--camera-index", type=int, default=0)
        command.add_argument("--width", type=positive_int, default=1280)
        command.add_argument("--height", type=positive_int, default=720)
        command.add_argument("--config", type=Path, default=Path("config/seats.json"))
    run.add_argument("--whole-frame", action="store_true", help="treat the entire view as one seat, for a quick check")
    run.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    run.add_argument("--confidence", type=probability, default=0.4)
    run.add_argument("--imgsz", type=positive_int, default=640)
    run.add_argument("--threads", type=positive_int, default=4)
    run.add_argument("--fps", type=positive, default=3.0, help="maximum inference FPS (actual speed depends on hardware)")
    run.add_argument("--occupied-seconds", type=positive, default=1.0)
    run.add_argument("--empty-seconds", type=positive, default=8.0)
    run.add_argument("--max-gap-seconds", type=positive, default=5.0)
    run.add_argument("--preview", action="store_true", help="show live video ONLY on the local desktop")
    run.add_argument("--status-file", type=Path, default=Path("output/status.json"))
    run.add_argument("--max-frames", type=positive_int, help="stop after this many analyzed frames (testing)")
    return root


def run(args):
    if args.fps * args.max_gap_seconds <= 1:
        raise ValueError("--max-gap-seconds must be longer than one inference interval (1 / fps).")
    if args.whole_frame:
        seats = [Seat("A01", ((0, 0), (1, 0), (1, 1), (0, 1)))]
        calibrated_size = None
    else:
        seats, calibrated_size = load_config(args.config)
    states = {s.id: SeatState(args.occupied_seconds, args.empty_seconds, args.max_gap_seconds) for s in seats}
    scores = {s.id: 0.0 for s in seats}
    now, last_write, last_key = time.monotonic(), -math.inf, None
    final_health = "stopped"
    observation_wall = None

    def emit(health):
        data = snapshot(
            states, scores, now, health, args.max_gap_seconds,
            observed_at=observation_wall if health == "ok" else None,
        )
        write_json(args.status_file, data)
        print(json.dumps(data, ensure_ascii=False), flush=True)

    try:
        emit("starting")
        detector = PersonDetector(args.model, args.confidence, args.imgsz, args.threads)
        with FrameSource(args.source, args.width, args.height, args.camera_index) as camera:
            analyzed = 0
            next_sample = -math.inf
            while True:
                frame = camera.read()
                if frame is None:
                    final_health = "end_of_video"
                    break
                observed = camera.observation_time
                if observed < next_sample:
                    # Continuously drain camera frames so a stale capture queue
                    # doesn't make seat state lag behind the real scene.
                    if not camera.is_video:
                        time.sleep(0.002)
                    continue
                next_sample = observed + 1 / args.fps
                observation_wall = datetime.now(timezone.utc)
                h, w = frame.shape[:2]
                if calibrated_size and abs((w / h) / (calibrated_size[0] / calibrated_size[1]) - 1) > 0.02:
                    raise ValueError("Camera aspect ratio changed. Recalibrate with this camera resolution.")
                start = time.monotonic()
                detections = detector.detect(frame)
                now = observed
                # Slow or stalled inference is not evidence of an empty seat.
                if time.monotonic() - start > args.max_gap_seconds:
                    for state in states.values():
                        state.invalidate()
                    scores = {s.id: 0.0 for s in seats}
                    health = "inference_too_slow"
                else:
                    scores = assign_people(seats, detections)
                    for seat_id, state in states.items():
                        state.update(scores[seat_id] > 0, now)
                    health = "ok"
                key = (health, tuple((s.state, s.candidate) for s in states.values()))
                wall = time.monotonic()
                if key != last_key or wall - last_write >= min(1.0, args.max_gap_seconds / 2):
                    emit(health)
                    last_key, last_write = key, wall
                analyzed += 1
                if args.preview:
                    import cv2
                    from .display import annotate
                    cv2.imshow("Seat monitor - Q to quit", annotate(frame, seats, states, detections))
                    if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                        break
                    if cv2.getWindowProperty("Seat monitor - Q to quit", cv2.WND_PROP_VISIBLE) < 1:
                        break
                if args.max_frames and analyzed >= args.max_frames:
                    break
    except KeyboardInterrupt:
        pass
    except Exception:
        final_health = "error"
        raise
    finally:
        for state in states.values():
            state.invalidate()
        scores = {s.id: 0.0 for s in seats}
        emit(final_health)
        if args.preview:
            import cv2
            cv2.destroyAllWindows()
    return 0


def main(argv=None):
    args = parser().parse_args(argv)
    try:
        if args.command == "prepare":
            prepare_model(args.model, args.ncnn, args.imgsz)
            return 0
        if args.camera_index < 0:
            raise ValueError("--camera-index must be non-negative")
        if args.command == "calibrate":
            from .calibrate import calibrate
            return calibrate(args)
        def stop(signum, frame):
            raise KeyboardInterrupt
        signal.signal(signal.SIGTERM, stop)
        return run(args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
