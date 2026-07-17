"""Camera warmup — standalone script, executed as subprocess for V4L2 device init.

Usage:
    python camera_warmup.py <source>

    source: camera index (int) or device path (str, e.g. /dev/video0)
    Exit 0 on success, 1 on failure.
"""
import sys
import cv2


def _resolve(source_arg: str):
    try:
        return int(source_arg)
    except ValueError:
        return source_arg


def main() -> int:
    if len(sys.argv) < 2:
        print("Usage: python camera_warmup.py <source>", file=sys.stderr)
        return 1

    source = _resolve(sys.argv[1])

    cap = cv2.VideoCapture(source, cv2.CAP_V4L2)
    if not cap.isOpened():
        cap = cv2.VideoCapture(source)

    if not cap.isOpened():
        return 1

    cap.read()
    cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
