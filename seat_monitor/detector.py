# SPDX-License-Identifier: AGPL-3.0-only
from contextlib import redirect_stdout
import os
from pathlib import Path
import sys
import urllib.request

from .core import Detection

MODEL_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt"
PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PROJECT_ROOT / "models" / "yolo11n.pt"


def load_yolo():
    # Set these BEFORE importing Ultralytics: no runtime auto-download/install,
    # telemetry, or shared user settings. 'prepare' is the explicit online step.
    os.environ["YOLO_OFFLINE"] = "True"
    os.environ["YOLO_AUTOINSTALL"] = "False"
    os.environ["YOLO_VERBOSE"] = "False"
    os.environ["YOLO_CONFIG_DIR"] = str(PROJECT_ROOT / ".cache" / "ultralytics")
    os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".cache" / "matplotlib"))
    with redirect_stdout(sys.stderr):
        from ultralytics import YOLO, settings
        settings.update({"sync": False})
    return YOLO


def prepare_model(path: Path, export_ncnn: bool = False, imgsz: int = 640):
    if path.suffix != ".pt":
        raise ValueError("prepare --model must end in .pt")
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".pt.part")
        try:
            print(f"Downloading official pretrained YOLO11n to {path}", file=sys.stderr)
            with urllib.request.urlopen(MODEL_URL, timeout=60) as response, temporary.open("wb") as target:
                while chunk := response.read(1024 * 1024):
                    target.write(chunk)
            if temporary.stat().st_size < 1_000_000:
                raise RuntimeError("Model download is unexpectedly small; refusing to load it.")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)
    if export_ncnn:
        # Install these explicitly first; automatic dependency installation is off.
        import importlib.util
        if any(importlib.util.find_spec(name) is None for name in ("ncnn", "pnnx")):
            raise RuntimeError("NCNN export needs: python -m pip install ncnn pnnx")
        model = load_yolo()(str(path), task="detect")
        with redirect_stdout(sys.stderr):
            exported = model.export(format="ncnn", imgsz=imgsz, device="cpu", half=False)
        print(f"NCNN model: {exported}", file=sys.stderr)
    print(f"Model ready: {path}", file=sys.stderr)


class PersonDetector:
    def __init__(self, path: Path, confidence: float, imgsz: int, threads: int):
        if not path.exists():
            raise FileNotFoundError(f"Model missing: {path}. Run python -m seat_monitor prepare first.")
        if not (path.is_file() and path.suffix == ".pt") and not (
            path.is_dir() and path.name.endswith("_ncnn_model")
        ):
            raise ValueError("Use a local .pt file or an exported *_ncnn_model directory.")
        YOLO = load_yolo()
        import torch
        torch.set_num_threads(threads)
        with redirect_stdout(sys.stderr):
            self.model = YOLO(str(path), task="detect")
        self.confidence, self.imgsz = confidence, imgsz
        self.person_class = None

    def detect(self, frame) -> list[Detection]:
        # Find class by name, never assume arbitrary custom models use class 0.
        if self.person_class is None:
            with redirect_stdout(sys.stderr):
                names = self.model.names
            names = names.items() if isinstance(names, dict) else enumerate(names)
            self.person_class = next((i for i, name in names if name == "person"), None)
            if self.person_class is None:
                raise ValueError("The selected model has no 'person' class.")
        with redirect_stdout(sys.stderr):
            result = self.model.predict(
                source=frame, classes=[self.person_class], conf=self.confidence,
                imgsz=self.imgsz, device="cpu", verbose=False, save=False,
                save_txt=False, save_crop=False, show=False, rect=False,
            )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxyn.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        return [Detection(tuple(box), float(score)) for box, score in zip(boxes, scores)]


class SeatObjectDetector(PersonDetector):
    """Optional COCO detector for people and common desk belongings.

    The default command-line runner intentionally keeps using PersonDetector
    for speed. A web integration can use this detector when belongings states
    are needed and the Pi's measured inference speed is acceptable.
    """

    DEFAULT_LABELS = (
        "person", "backpack", "handbag", "suitcase", "bottle",
        "cup", "book", "laptop", "cell phone",
    )

    def __init__(self, path: Path, confidence: float, imgsz: int, threads: int, labels=None):
        super().__init__(path, confidence, imgsz, threads)
        self.requested_labels = tuple(labels or self.DEFAULT_LABELS)
        self.class_ids = None
        self.class_names = None

    def detect(self, frame) -> list[Detection]:
        if self.class_ids is None:
            with redirect_stdout(sys.stderr):
                names = self.model.names
            names = dict(names.items() if isinstance(names, dict) else enumerate(names))
            self.class_names = names
            wanted = set(self.requested_labels)
            self.class_ids = [class_id for class_id, name in names.items() if name in wanted]
            if not self.class_ids or "person" not in {names[i] for i in self.class_ids}:
                raise ValueError("The selected model must contain the 'person' class.")
        with redirect_stdout(sys.stderr):
            result = self.model.predict(
                source=frame, classes=self.class_ids, conf=self.confidence,
                imgsz=self.imgsz, device="cpu", verbose=False, save=False,
                save_txt=False, save_crop=False, show=False, rect=False,
            )[0]
        if result.boxes is None:
            return []
        boxes = result.boxes.xyxyn.cpu().tolist()
        scores = result.boxes.conf.cpu().tolist()
        classes = result.boxes.cls.cpu().tolist()
        return [
            Detection(tuple(box), float(score), self.class_names[int(class_id)], int(class_id))
            for box, score, class_id in zip(boxes, scores, classes)
        ]
