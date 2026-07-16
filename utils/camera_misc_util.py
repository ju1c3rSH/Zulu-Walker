import os
import re
import subprocess
from dataclasses import dataclass
from typing import List, Optional

import cv2

from utils.log_util import log_print


@dataclass
class DeviceCameraInfo:
    index: int
    name: str
    device: str
    type: str = "camera"


CameraInfo = DeviceCameraInfo


class CameraMiscUtil:
    BY_ID_DIR = "/dev/v4l/by-id"

    @staticmethod
    def verify_camera(index: int):
        cap = cv2.VideoCapture(index)
        if not cap.isOpened():
            return False
        ret, _ = cap.read()
        cap.release()
        return ret

    @staticmethod
    def find_working_cameras():
        cameras = []
        try:
            output = subprocess.check_output(
                ["v4l2-ctl", "--list-devices"],
                stderr=subprocess.STDOUT, text=True
            )
            current_name = None
            device_nodes = []
            for line in output.splitlines():
                if not line.startswith("\t"):
                    if current_name and device_nodes:
                        cameras.extend(
                            CameraMiscUtil.process_device_group(current_name, device_nodes)
                        )
                    current_name = line.rstrip(":")
                    device_nodes = []
                else:
                    match = re.search(r"/dev/video\d+", line)
                    if match:
                        device_nodes.append(match.group(0))

            if current_name and device_nodes:
                cameras.extend(
                    CameraMiscUtil.process_device_group(current_name, device_nodes)
                )
        except (FileNotFoundError, subprocess.CalledProcessError, PermissionError) as e:
            log_print(f"Error finding cameras: {e}")

        return cameras

    @staticmethod
    def process_device_group(device_name, nodes):
        video_indices = []
        for node in nodes:
            match = re.search(r"/dev/video(\d+)", node)
            if match:
                idx = int(match.group(1))
                video_indices.append((idx, node))

        if not video_indices:
            return []

        video_indices.sort()

        for idx, node in video_indices:
            if CameraMiscUtil.verify_camera(idx):
                return [DeviceCameraInfo(index=idx, name=device_name, device=node)]

        return []

    # ------------------------------------------------------------------ #
    #  /dev/v4l/by-id/  stable-path support                              #
    # ------------------------------------------------------------------ #

    @staticmethod
    def enumerate_by_id() -> List[DeviceCameraInfo]:
        """Enumerate cameras via /dev/v4l/by-id/ symlinks.

        Only returns *-video-index0 entries (the main video node
        per physical camera).  Returns an empty list when the
        directory does not exist (containers, WSL, etc.).
        """
        if not os.path.isdir(CameraMiscUtil.BY_ID_DIR):
            return []

        seen_device: set[str] = set()
        result: List[DeviceCameraInfo] = []

        try:
            for entry in os.listdir(CameraMiscUtil.BY_ID_DIR):
                full_link = os.path.join(CameraMiscUtil.BY_ID_DIR, entry)
                if not os.path.islink(full_link):
                    continue
                if not entry.endswith("-video-index0"):
                    continue

                target = os.readlink(full_link)
                video_match = re.search(r"video(\d+)", target)
                if not video_match:
                    continue

                idx = int(video_match.group(1))

                actual = os.path.normpath(
                    os.path.join(CameraMiscUtil.BY_ID_DIR, target)
                )

                if actual not in seen_device:
                    seen_device.add(actual)
                    result.append(DeviceCameraInfo(
                        index=idx,
                        name=entry,
                        device=full_link,
                    ))

        except PermissionError:
            log_print("Warning: no permission to read /dev/v4l/by-id/")

        return result

    @staticmethod
    def get_udev_property(device_path: str, key: str = "ID_SERIAL") -> Optional[str]:
        """Query a single udev property for the given device path.

        Returns None when udevadm is unavailable or the key is not found.
        """
        try:
            output = subprocess.check_output(
                ["udevadm", "info", "--query=property", device_path],
                stderr=subprocess.DEVNULL, text=True,
            )
            for line in output.splitlines():
                if line.startswith(key + "="):
                    return line.split("=", 1)[1]
        except (FileNotFoundError, subprocess.CalledProcessError):
            pass
        return None

    # ------------------------------------------------------------------ #
    #  Combined discovery & resolution                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def discover_all() -> List[DeviceCameraInfo]:
        """Comprehensive camera discovery.

        1. Try by-id enumeration first  (stable udev paths).
        2. Fall back to v4l2-ctl enumeration  (MIPI / non-USB cameras).
        3. Deduplicates by the resolved ``/dev/videoN`` node.
        """
        seen: set[str] = set()
        result: List[DeviceCameraInfo] = []

        for cam in CameraMiscUtil.enumerate_by_id():
            actual = os.path.normpath(
                os.path.join(CameraMiscUtil.BY_ID_DIR, os.readlink(cam.device))
            ) if os.path.islink(cam.device) else cam.device
            if actual not in seen:
                seen.add(actual)
                result.append(cam)

        for cam in CameraMiscUtil.find_working_cameras():
            if cam.device not in seen:
                seen.add(cam.device)
                result.append(cam)

        return result

    @staticmethod
    def resolve_camera_source(source: int | str) -> int | str:
        """Resolve *source* to a usable device path or index.

        Rules
        -----
        * ``int`` – returned unchanged (backward compatible).
        * ``str`` starting with ``"/dev/"`` – validated for existence;
          if missing it attempts to re-resolve via the embedded serial.
        * Other ``str`` – treated as a USB serial number and matched
          against discovered cameras.

        Returns the original value when resolution fails (the caller
        will eventually fail with a clear OS error).
        """
        if isinstance(source, int):
            return source

        if source.startswith("/dev/"):
            if os.path.exists(source):
                return source
            log_print(f"Camera source '{source}' not found, re-resolving...")
            resolved = CameraMiscUtil._match_by_id_serial(source)
            if resolved is not None:
                return resolved
            log_print(f"Re-resolution failed, falling back to original: {source}")
            return source

        matched = CameraMiscUtil._match_by_serial(source)
        if matched is not None:
            return matched

        log_print(f"Camera source '{source}' could not be resolved, using as-is")
        return source

    @staticmethod
    def _match_by_id_serial(old_path: str) -> Optional[str]:
        """Extract the serial from a stale by-id path and search for a live match."""
        # Match any by-id prefix (usb-, platform-, pci-, etc.)
        m = re.search(r"/[A-Za-z0-9_-]+-([A-Za-z0-9_]+)-video-index0$", old_path)
        if m:
            return CameraMiscUtil._match_by_serial(m.group(1))
        return None

    @staticmethod
    def _match_by_serial(serial: str) -> Optional[str]:
        """Find a by-id device whose link name contains *serial*."""
        for cam in CameraMiscUtil.enumerate_by_id():
            if serial in cam.device:
                return cam.device
        return None

    # ------------------------------------------------------------------ #
    #  CLI helper                                                        #
    # ------------------------------------------------------------------ #

    @staticmethod
    def print_discovery() -> None:
        """Print a human-readable camera list (for ``--discover``)."""
        cameras = CameraMiscUtil.discover_all()
        if not cameras:
            log_print("No cameras found.")
            return

        log_print(f"Found {len(cameras)} camera(s):")
        for i, cam in enumerate(cameras):
            serial = CameraMiscUtil.get_udev_property(cam.device, "ID_SERIAL")
            model = CameraMiscUtil.get_udev_property(cam.device, "ID_MODEL")
            extra = (
                f"Serial: {serial}" if serial else "",
                f"Model: {model}" if model else "",
            )
            extra_str = ", ".join(p for p in extra if p)
            log_print(f"  [{i}] {cam.name}")
            log_print(f"       Path: {cam.device}")
            if extra_str:
                log_print(f"       {extra_str}")


def main():
    """Entry point for ``python -m utils.camera_misc_util --discover``."""
    import argparse
    parser = argparse.ArgumentParser(description="Camera discovery utility")
    parser.add_argument("--discover", action="store_true", help="List all cameras")
    args = parser.parse_args()
    if args.discover:
        CameraMiscUtil.print_discovery()


if __name__ == "__main__":
    main()
