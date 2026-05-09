import cv2, subprocess, re
from dataclasses import dataclass


@dataclass
class CameraInfo:
    index: int
    name: str
    device: str
    type: str = "camera"


class CameraMiscUtil:
    @staticmethod
    def verify_camera(index):
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
        except Exception as e:
            print(f"Error finding cameras: {e}")

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
                return [CameraInfo(index=idx, name=device_name, device=node)]

        return []
