# Open-source components

- Ultralytics YOLO11n: COCO-pretrained object detector; this app filters the `person` class.
  Code and pretrained weights: AGPL-3.0 / alternative Ultralytics commercial license.
  https://github.com/ultralytics/ultralytics
  https://docs.ultralytics.com/models/yolo11/
  Official weight URL: https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt
- Picamera2: BSD-2-Clause. Installed from Raspberry Pi OS packages, not vendored.
  https://github.com/raspberrypi/picamera2
- OpenCV: Apache-2.0. https://github.com/opencv/opencv
- NumPy: BSD-3-Clause. https://github.com/numpy/numpy
- PyTorch: BSD-style license; dependencies have their own terms. https://github.com/pytorch/pytorch
- ncnn (optional runtime): BSD-3-Clause. https://github.com/Tencent/ncnn
- pnnx (optional model conversion): BSD-3-Clause. https://github.com/pnnx/pnnx

This project uses AGPL-3.0-only (see LICENSE). Retain upstream notices when
redistributing dependencies. No training dataset is bundled or needed. A COCO
pretrained model does not make arbitrary photos or dataset images public domain.

References:
- Pi inference and NCNN: https://docs.ultralytics.com/guides/raspberry-pi/
- Camera setup: https://www.raspberrypi.com/documentation/computers/camera_software.html
- Picamera2 manual: https://datasheets.raspberrypi.com/camera/picamera2-manual.pdf
