from typing import Optional, Tuple
import time
import numpy as np
import cv2


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


def uv_kalman_update(
    uv_kf, uv_center: Optional[Tuple[float, float]],
    uv_tracking_initialized: bool, uv_lost_frames: int, uv_max_lost_frames: int,
    uv_q_base: float, uv_q_vel_base: float,
    uv_last_predict_time: Optional[float],
) -> Tuple[Optional[Tuple[int, int]], bool, int, Optional[float]]:
    if uv_center is not None:
        x, y = uv_center
        measurement = np.array([[np.float32(x)], [np.float32(y)]])

        if not uv_tracking_initialized:
            uv_kf.statePost = np.array([[x], [y], [0], [0]], dtype=np.float32)
            uv_kf.errorCovPost = np.eye(4, dtype=np.float32)
            uv_tracking_initialized = True
            uv_lost_frames = 0
        else:
            uv_kf.correct(measurement)
            uv_lost_frames = 0
    else:
        if uv_tracking_initialized:
            uv_lost_frames += 1
            if uv_lost_frames > uv_max_lost_frames:
                uv_tracking_initialized = False
                uv_lost_frames = 0
                return (None, uv_tracking_initialized, uv_lost_frames, uv_last_predict_time)

    if uv_tracking_initialized:
        now = time.time()
        if uv_last_predict_time is not None:
            dt = max(0.001, min(0.2, now - uv_last_predict_time))
        else:
            dt = 1.0 / 90.0
        uv_last_predict_time = now
        update_transition_matrix(uv_kf, dt, uv_q_base, uv_q_vel_base)
        prediction = uv_kf.predict()
        result = (int(prediction[0]), int(prediction[1]))
        return (result, uv_tracking_initialized, uv_lost_frames, uv_last_predict_time)

    return (None, uv_tracking_initialized, uv_lost_frames, uv_last_predict_time)
