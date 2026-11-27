import cv2
import numpy as np
from .. import PendulumCalibrator, RailCalibration


class PendulumCalibDebugRunner:
    def __init__(self, camera_source=0, width=640, height=640):
        self.camera_source = camera_source
        self.width = width
        self.height = height
        self.calibrator = PendulumCalibrator(frame_w=width, frame_h=height)
        self._running = False

    def run(self):
        cap = cv2.VideoCapture(self.camera_source)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self._running = True

        while self._running:
            ret, frame = cap.read()
            if not ret:
                break

            try:
                result = self.calibrator.calibrate(frame)
                debug = self.calibrator.get_debug_frame()

                if result.calibrated:
                    cx, cy = int(result.origin_x), int(result.origin_y)
                    cv2.circle(frame, (cx, cy), 6, (255, 0, 0), -1)
                    length = 300
                    ex = int(cx + length * result.dir_cos)
                    ey = int(cy + length * result.dir_sin)
                    sx = int(cx - length * result.dir_cos)
                    sy = int(cy - length * result.dir_sin)
                    cv2.line(frame, (sx, sy), (ex, ey), (0, 255, 255), 2)

                    info = "angle={:.4f} origin=({:.0f},{:.0f})".format(
                        result.angle_rad, result.origin_x, result.origin_y)
                    cv2.putText(frame, info, (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "CALIB FAILED", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                dbg_h, dbg_w = debug.shape[:2]
                scale = frame.shape[0] / dbg_h
                new_w = int(dbg_w * scale)
                debug_resized = cv2.resize(debug, (new_w, frame.shape[0]))

                combined = np.hstack([frame, debug_resized])
                cv2.imshow("Pendulum Calib Debug", combined)
            except Exception as e:
                import traceback
                traceback.print_exc()
                cv2.putText(frame, f"ERROR: {e}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                cv2.imshow("Pendulum Calib Debug", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def stop(self):
        self._running = False
