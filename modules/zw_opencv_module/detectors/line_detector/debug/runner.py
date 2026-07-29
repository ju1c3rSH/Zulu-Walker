import cv2
import numpy as np
from .. import LineDetector


class LineDebugRunner:
    def __init__(self, camera_source=0, width=640, height=480):
        self.camera_source = camera_source
        self.width = width
        self.height = height
        self.detector = LineDetector()
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
            
            result = self.detector.detect(frame)
            debug_binary = self.detector.get_debug_frame()
            
            dbg_h, dbg_w = debug_binary.shape[:2]
            scale = frame.shape[0] / dbg_h
            new_w = int(dbg_w * scale)
            debug_binary = cv2.resize(debug_binary, (new_w, frame.shape[0]))
            
            if result.target_found:
                cv2.circle(frame, (result.center_x, result.center_y), 6, (0, 0, 255), -1)
                cv2.line(frame, (result.frame_width // 2, 0), (result.frame_width // 2, result.frame_height), (0, 255, 0), 1)
                cv2.line(frame, (result.center_x, 0), (result.center_x, result.frame_height), (0, 0, 255), 1)
                info = f"ERR x:{result.percent_error_x:+d}"
                cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            else:
                cv2.putText(frame, "LINE LOST", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            combined = np.hstack([frame, debug_binary])
            cv2.imshow("Line Debug", combined)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        cv2.destroyAllWindows()
    
    def stop(self):
        self._running = False
