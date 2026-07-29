from typing import Optional, Tuple
import time
import numpy as np
def update_transition_matrix(kf, dt: float, q_base: float, q_vel_base: float):
    kf.transitionMatrix = np.array([
        [1, 0, dt, 0],
        [0, 1, 0, dt],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ], dtype=np.float32)
    dt2 = dt * dt
    kf.processNoiseCov = np.diag(np.array(
        [q_base / dt2, q_base / dt2, q_vel_base * dt, q_vel_base * dt],
        dtype=np.float32))


def kalman_update(
    kf, measurement: Optional[Tuple[float, float]],
    tracking_initialized: bool, lost_frames: int, max_lost_frames: int,
    q_base: float, q_vel_base: float,
    last_predict_time: Optional[float],
) -> Tuple[Optional[Tuple[float, float]], bool, int, Optional[float]]:
    if measurement is not None:
        x, y = measurement
        measurement_array = np.array([[x], [y]], dtype=np.float32)

        if not tracking_initialized:
            kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            kf.errorCovPost = np.eye(4, dtype=np.float32)
            tracking_initialized = True
            lost_frames = 0
        else:
            kf.correct(measurement_array)
            lost_frames = 0
    else:
        if tracking_initialized:
            lost_frames += 1
            if lost_frames > max_lost_frames:
                tracking_initialized = False
                lost_frames = 0
                return (None, tracking_initialized, lost_frames, last_predict_time)

    if tracking_initialized:
        now = time.time()
        if last_predict_time is not None:
            dt = max(0.001, min(0.2, now - last_predict_time))
        else:
            dt = 1.0 / 90.0
        last_predict_time = now
        update_transition_matrix(kf, dt, q_base, q_vel_base)
        prediction = kf.predict()
        result = (float(prediction[0, 0]), float(prediction[1, 0]))
        return (result, tracking_initialized, lost_frames, last_predict_time)

    return (None, tracking_initialized, lost_frames, last_predict_time)
