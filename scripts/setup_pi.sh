#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ "$(uname -m)" != "aarch64" ]]; then
    echo "Use 64-bit Raspberry Pi OS (aarch64)." >&2
    exit 1
fi
python3 -c 'import sys; assert (3, 11) <= sys.version_info[:2] <= (3, 13), "Use the system Python on Raspberry Pi OS Bookworm or Trixie."'
sudo apt-get update
sudo apt-get install -y python3-venv python3-pip python3-picamera2 python3-opencv python3-numpy git
python3 -m venv --system-site-packages .venv
.venv/bin/python -m pip install --upgrade pip
# Keep the NumPy ABI used by apt's Picamera2/simplejpeg. pip selects a compatible
# OpenCV wheel instead of silently replacing the OS NumPy with a new major.
mkdir -p .cache
pi_numpy_version="$(python3 -c 'import numpy; print(numpy.__version__)')"
printf 'numpy==%s\n' "$pi_numpy_version" > .cache/pi-constraints.txt
.venv/bin/python -m pip install -r requirements.txt -c .cache/pi-constraints.txt
.venv/bin/python -c 'from picamera2 import Picamera2; import cv2; import numpy; print("Camera libraries OK; numpy", numpy.__version__)'
echo "Ready. Next: source .venv/bin/activate && python -m seat_monitor prepare"
