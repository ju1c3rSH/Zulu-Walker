import cv2
import numpy as np
from typing import Optional, Tuple
from ..detectors.circle_target_detector import CircleTargetDetector, DetectMethod
from ..models.circle import CircleTargetItem, CircleTargets
from ..models import Color
from utils.focal_distance_util import reference_size_dict

from .registry import register_processor
from .base import Processor, VisionResult
from utils.log_util import log_print



@register_processor("CircleTargetProcessor")
class CircleTargetProcessor(Processor):
    """Detect circular and elliptical targets and report their offset."""

    def __init__(self, name: str = "circle_target_detect"):
        super().__init__(name)
        self.detector = CircleTargetDetector(name)
        self.target_max_radius = 10  # Maximum radius in centimeters.
        self.target_min_radius = 2  # Minimum radius in centimeters.
        self.target_color: Optional[Color] = Color.RED
        self._frame_width: int = 640
        self._frame_height: int = 480
        self._fps: float = 0.0  # FPS supplied through context.
        self._log_interval = 300  # Log every 300 frames (about 2 s at 150 fps).
        self._frame_count = 0
        self._last_logged_target_found: Optional[bool] = None

    def set_target_color(self, color: Optional[Color]):
        """Set the color to search for.

        Args:
            color: Target color, or None to detect any color.
        """
        self.target_color = color

    @property
    def _target_color_str(self) -> Optional[str]:
        """Convert Color enum to detector-compatible string (e.g. 'Red')."""
        if self.target_color is None:
            return None
        return self.target_color.name.lower().capitalize()

    def set_detect_method(self, method: DetectMethod):
        """Set the detection method used by the underlying detector.

        Args:
            method: One of the DetectMethod options, such as CONTOUR_ELLIPSE
                or EDGE_CONTOUR_ELLIPSE.
        """
        self.detector.set_detect_method(method)

    def process(self, frame: np.ndarray, context: dict = None) -> VisionResult:
        """Detect targets and compute their position error and distance.

        Args:
            context: May carry 'fps' and a 'focal_calculator' for distance.
        """

        if context and "fps" in context:
            self._fps = context["fps"]
        
        focal_calculator = context.get("focal_calculator") if context else None

        if frame is None:
            return VisionResult(
                task_name=self.name,
                success=False,
                error_message="Empty frame provided",
            )

        try:
            self._frame_height, self._frame_width = frame.shape[:2]

            targets = self.detector.detect_circle_targets(
                frame, target_color=self._target_color_str
            )

            target = self._find_target(targets)

            if target is None:
                error_msg = "Target not found"
                if self.target_color is not None:
                    error_msg = f"{self.target_color.name} target not found"
                self._log_status(False, None, 0, 0)
                return VisionResult(
                    task_name=self.name,
                    result_data={
                        "targets": targets,
                        "target_color": self._target_color_str,
                        "percent_error_x": 0,
                        "percent_error_y": 0,
                        "target_distance_mm": None,
                        "target_found": False,
                    },
                    success=False,
                    error_message=error_msg,
                )

            ordered_quad = self.detector._order_quad_points(target.quad_points)
            src_points = ordered_quad.reshape(4, 2).astype(np.float32)
            edges = [
                np.linalg.norm(src_points[i] - src_points[(i+1) % 4])
                for i in range(4)
            ]

            long_edge = max(edges)
            short_edge = min(edges)

            if self.detector.is_uv_spot_detected:
                # Fall back to the UV spot as the tracking reference when the target is occluded.
                uv_center = self.detector.uv_spot_center
            else:
                uv_center = None

            percent_error_x, percent_error_y = self._calculate_position_error(target, uv_center)

            target_distance_mm = None
            if focal_calculator and target is not None:
                quad_real_avg = reference_size_dict["quad"][2]  # Reference quad size in mm.
                avg_edge = (long_edge + short_edge) / 2  # Mean edge length in pixels.
                target_distance_mm = focal_calculator.calculate_distance(
                    real_size_mm=quad_real_avg,
                    pixel_size=avg_edge
                )

            self._log_status(True, target, percent_error_x, percent_error_y)
            return VisionResult(
                task_name=self.name,
                result_data={
                    "targets": targets,
                    "target": target,
                    "is_quad_detected": self.detector.is_detected_quad,
                    "is_uv_spot_detected": self.detector.is_uv_spot_detected,
                    "target_color": self.target_color,
                    "percent_error_x": percent_error_x,
                    "percent_error_y": percent_error_y,
                    "target_found": True,
                    "target_distance_mm": target_distance_mm,
                },
                success=True,
            )

        except Exception as e:
            log_print(f"[CircleTargetProcessor] Error processing frame: {e}")
            import traceback

            traceback.print_exc()
            return VisionResult(
                task_name=self.name,
                success=False,
                error_message=f"Error processing frame: {str(e)}",
            )

    def _find_target(self, targets: CircleTargets) -> Optional[CircleTargetItem]:
        """Pick the largest target matching the configured color.

        Returns None when no target matches.
        """
        if not targets.targets:
            return None

        # With no color filter, fall back to the largest target.
        if self.target_color is None:
            return max(targets.targets, key=lambda t: t.radius or 0)

        color_str = self._target_color_str
        matching = [t for t in targets.targets if t.color == color_str]
        if matching:
            return max(matching, key=lambda t: t.radius or 0)
        return None

    def _calculate_position_error(self, target: CircleTargetItem, uv_center: Optional[Tuple[int, int]] = None) -> tuple:
        """Compute the target offset from a reference point.

        Measures against uv_center when provided, otherwise against the frame
        center, and returns the error normalized to [-5000, 5000].

        Args:
            target: Detected target, whose center is the quad center.
            uv_center: Optional UV spot to measure against instead of the
                frame center.
        """
        target_x, target_y = target.center_coordinates

        if uv_center is not None:
            # UV mode: error drives the quad center toward the UV spot.
            center_x, center_y = uv_center
        else:
            center_x = self._frame_width // 2
            center_y = self._frame_height // 2

        pixel_error_x = target_x - center_x
        pixel_error_y = target_y - center_y

        # Normalize to the [-5000, 5000] scale used by cargo and ring discovery.
        half = max(self._frame_width, self._frame_height) / 2.0
        pe_x = int((pixel_error_x * 5000.0) / half)
        pe_y = int((pixel_error_y * 5000.0) / half)
        percent_error_x = max(-5000, min(5000, pe_x))
        percent_error_y = max(-5000, min(5000, pe_y))

        return percent_error_x, percent_error_y

    def _log_status(self, target_found: bool, target, percent_error_x: int, percent_error_y: int):
        self._frame_count += 1
        state_changed = self._last_logged_target_found != target_found
        if state_changed or self._frame_count % self._log_interval == 0:
            status = "TRACKING" if target_found else "SEARCHING"
            target_info = ""
            if target_found and target:
                target_info = f" | Target: {target.center_coordinates} | Error: X={percent_error_x:+d}, Y={percent_error_y:+d}"
            log_print(f"[CircleTargetProcessor] FPS: {self._fps:.1f} | State: {status}{target_info}")
            self._last_logged_target_found = target_found

    def draw_result(self, frame: np.ndarray, result: VisionResult) -> np.ndarray:
        """Overlay detected targets, center cross, UV spot, and status text."""
        if frame is None:
            return frame

        if result.result_data is None:
            return frame

        targets = result.result_data.get("targets")
        target = result.result_data.get("target")
        target_found = result.result_data.get("target_found", False)

        if targets and targets.targets:
            for t in targets.targets:
                self._draw_target(frame, t, is_target=(t == target))

        self._draw_center_cross(frame)

        if self.detector.is_uv_spot_detected and self.detector.uv_spot_center is not None:
            uv_x, uv_y = self.detector.uv_spot_center
            cv2.circle(frame, (uv_x, uv_y), 4, (255, 0, 255), -1)  # Magenta in BGR.
            cv2.putText(frame, f"UV({uv_x},{uv_y})", (uv_x + 10, uv_y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)

            if target is not None:
                quad_x, quad_y = target.center_coordinates
                cv2.line(frame, (quad_x, quad_y), (uv_x, uv_y), (255, 0, 255), 2)

        self._draw_status_info(frame, result, target_found)

        return frame

    def _draw_target(
        self, frame: np.ndarray, target: CircleTargetItem, is_target: bool = False
    ):
        """Draw one target center and its coordinates.

        Args:
            is_target: Whether this is the selected target, which adds a line
                from the frame center.
        """
        cx, cy = target.center_coordinates

        cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

        cv2.putText(
            frame,
            f"({cx},{cy})",
            (cx + 10, cy - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        if is_target:
            center_x = self._frame_width // 2
            center_y = self._frame_height // 2
            cv2.line(frame, (center_x, center_y), (cx, cy), (0, 255, 255), 2)

    def _draw_center_cross(self, frame: np.ndarray):
        """Draw a crosshair at the frame center."""
        center_x = self._frame_width // 2
        center_y = self._frame_height // 2

        cv2.line(
            frame,
            (center_x - 20, center_y),
            (center_x + 20, center_y),
            (255, 255, 255),
            1,
        )
        cv2.line(
            frame,
            (center_x, center_y - 20),
            (center_x, center_y + 20),
            (255, 255, 255),
            1,
        )


    def _draw_status_info(
        self, frame: np.ndarray, result: VisionResult, target_found: bool
    ):
        """Overlay tracking status, error, and distance information."""
        data = result.result_data

        # Draw text directly; no separate background layer is needed.
        lines = [
            (f"FPS: {self._fps:.1f}", (10, 25), (0, 255, 255)),
            (f"Method: {self.detector.detect_method.value}", (90, 25), (200, 200, 200)),
            (f"Target: {data.get('target_color', 'Any')}", (10, 50), (255, 255, 255)),
        ]

        if target_found:
            distance_mm = data.get('target_distance_mm')
            if distance_mm is not None:
                distance_m = distance_mm / 1000.0
                lines.append((f"Dist: {distance_m:.2f}m", (150, 50), (0, 255, 255)))

            lines.extend(
                [
                    (
                        f"Error: X={data.get('percent_error_x', 0):+d}, Y={data.get('percent_error_y', 0):+d}",
                        (10, 75),
                        (0, 255, 0),
                    ),
                    (
                        f"Count: {data.get('targets', {}).number if data.get('targets') else 0}",
                        (10, 100),
                        (255, 255, 255),
                    ),
                ]
            )
        else:
            lines.append(("Target lost", (10, 75), (0, 0, 255)))

        for text, pos, color in lines:
            cv2.putText(frame, text, pos, cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)


