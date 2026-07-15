#!/usr/bin/env python3
from utils.log_util import log_print

"""Download WeChatQRCode model files for OpenCV.

Run: python download_models.py
Downloads to the same directory as this script.
"""
import os
import sys
from urllib.request import urlretrieve

COMMIT = "a8b69ccc738421293254aec5ddb38bd523503252"
BASE = f"https://raw.githubusercontent.com/WeChatCV/opencv_3rdparty/{COMMIT}"

FILES = [
    "detect.prototxt",
    "detect.caffemodel",
    "sr.prototxt",
    "sr.caffemodel",
]

def main():
    target_dir = os.path.dirname(os.path.abspath(__file__))
    log_print(f"Target dir: {target_dir}")
    for fname in FILES:
        url = f"{BASE}/{fname}"
        dest = os.path.join(target_dir, fname)
        if os.path.isfile(dest):
            log_print(f"SKIP  {fname} (already exists)")
            continue
        log_print(f"GET   {url}")
        try:
            urlretrieve(url, dest)
            size_kb = os.path.getsize(dest) / 1024
            log_print(f"OK    {fname} ({size_kb:.1f} KB)")
        except Exception as e:
            log_print(f"FAIL  {fname}: {e}", file=sys.stderr)
    log_print("Done.")

if __name__ == "__main__":
    main()
