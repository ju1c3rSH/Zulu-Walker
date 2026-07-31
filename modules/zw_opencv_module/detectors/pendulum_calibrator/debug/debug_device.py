#!/usr/bin/env python3
"""
MaixCAM2 standalone pendulum calibration debug tool.
Direct camera capture, interactive parameter tuning,
LCD real-time preview, detailed diagnostics, and frame saving.

Usage:
  cd /maixapp/apps/Zulu-Walker
  python3 modules/zw_opencv_module/detectors/pendulum_calibrator/debug/debug_device.py
"""

import os
import sys
import math
import time
import traceback

import cv2
import numpy as np

import yaml

import maix.camera
import maix.display
import maix.image

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_SCRIPT_DIR, '..', '..', '..', '..', '..'))
sys.path.insert(0, _PROJECT_ROOT)

from modules.zw_opencv_module.detectors.pendulum_calibrator import PendulumCalibrator

_YAML_PATH = os.path.join(_SCRIPT_DIR, 'calib_debug.yaml')


def _parse_value(v: str):
    v = v.strip()
    if v.lower() in ('true', 'yes', 'on', '1'):
        return True
    if v.lower() in ('false', 'no', 'off', '0'):
        return False
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    return v


def _load_config(path: str) -> dict:
    with open(path, 'r') as f:
        return yaml.safe_load(f)


def _save_config(path: str, config: dict):
    with open(path, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)


def _print_config(config: dict):
    for section, values in config.items():
        if not isinstance(values, dict):
            print(f"[CALIB]   {section} = {values}")
            continue
        for key, val in values.items():
            print(f"[CALIB]   {section}.{key} = {val}")


def _update_config(config: dict, key_str: str, value_str: str) -> bool:
    value = _parse_value(value_str)
    if '.' in key_str:
        section, key = key_str.split('.', 1)
        if section in config and isinstance(config[section], dict) and key in config[section]:
            config[section][key] = value
            print(f"[CALIB] set {section}.{key} = {config[section][key]}")
            return True
        print(f"[CALIB] set {section}.{key} = {value}")
        config.setdefault(section, {})[key] = value
        return True
    for section, values in config.items():
        if isinstance(values, dict) and key_str in values:
            values[key_str] = value
            print(f"[CALIB] set {section}.{key_str} = {values[key_str]}")
            return True
    print(f"[CALIB] set {key_str} = {value}")
    config[key_str] = value
    return True


def _print_diagnostics(diag: dict, frame_no: int):
    print(f"[CALIB] --- frame {frame_no} ---")

    gm = diag.get('gray_mean')
    gs = diag.get('gray_std')
    if gm is not None:
        print(f"[CALIB] gray: mean={gm:.1f} std={gs:.1f}")

    tm = diag.get('threshold_method')
    if tm:
        tr = diag.get('threshold_ret', '?')
        wp = diag.get('white_px_pct', 0)
        print(f"[CALIB] threshold: method={tm} ret={tr} white_px={wp}%")

    cf = diag.get('contours_found')
    if cf is not None:
        cma = diag.get('contours_max_area')
        cma_s = f"{cma}" if cma is not None else "None"
        print(f"[CALIB] contours: found={cf} max_area={cma_s}")

    fr = diag.get('fail_reason')

    # Column-centroid diagnostics always print (success or fallback), so a
    # failed primary stage is debuggable instead of being silently hidden.
    cpts = diag.get('column_points')
    ct = diag.get('column_threshold')
    cmed = diag.get('column_median_h')
    cfail = diag.get('column_fail_reason')
    if cpts is not None or ct is not None or cfail is not None:
        cpts_s = f" pts={cpts}" if cpts is not None else ""
        ct_s = f" th={ct}" if ct is not None else ""
        cmed_s = f" med_h={cmed}" if cmed is not None else ""
        cfail_s = f" col_fail={cfail}" if cfail is not None else ""
        print(f"[CALIB] column-centroid:{ct_s}{cpts_s}{cmed_s}{cfail_s}")

    if cfail in ('column_insufficient', 'column_median_height', 'column_off_band',
                 'column_fit_error', 'column_threshold_error'):
        print(f"[CALIB] \u2717 column FAIL: {cfail}")
        return

    if fr == 'max_area':
        ca = diag.get('contour_area', '?')
        ma = diag.get('max_area_limit', '?')
        print(f"[CALIB] \u2717 FAIL: max_area (area={ca} > max={ma}, frame-filling blob)")
        return

    if fr == 'no_contours':
        print(f"[CALIB] \u2717 FAIL: no contours found")
        return

    if fr == 'min_area':
        ca = diag.get('contour_area', '?')
        ma = diag.get('min_area_limit', '?')
        print(f"[CALIB] \u2717 FAIL: min_area (area={ca} < min={ma})")
        return

    if fr == 'aspect_ratio':
        ca = diag.get('contour_area', '?')
        ma = diag.get('min_area_limit', '?')
        asp = diag.get('contour_aspect', '?')
        masp = diag.get('min_aspect_limit', '?')
        print(f"[CALIB] contour: area={ca}(min={ma}) aspect={asp}(min={masp})")
        print(f"[CALIB] \u2717 FAIL: aspect_ratio (ratio={asp} < min={masp})")
        return

    if fr == 'center_out_of_bounds':
        rc = diag.get('rect_center', '?')
        print(f"[CALIB] \u2717 FAIL: center_out_of_bounds center={rc} bounds=[10%%,90%%]")
        return

    if fr in ('threshold_error', 'find_contours_error', 'min_area_rect_error', 'degenerate_rect'):
        print(f"[CALIB] \u2717 FAIL: {fr}")
        return

    ca = diag.get('contour_area')
    if ca is not None:
        ma = diag.get('min_area_limit', '?')
        asp = diag.get('contour_aspect', '?')
        masp = diag.get('min_aspect_limit', '?')
        check = '\u2713' if diag.get('contour_aspect_ok') else ''
        print(f"[CALIB] contour: area={ca}(min={ma}) aspect={asp}(min={masp}) {check}")

    rc = diag.get('rect_center')
    ra = diag.get('rect_angle')
    if rc is not None and ra is not None and not diag.get('used_fallback'):
        print(f"[CALIB] minAreaRect: center={rc} angle={ra}\u00b0")

    if diag.get('used_fallback'):
        print(f"[CALIB] \u2192 fallback: edge detection")
        cl = diag.get('canny_low', '?')
        ch = diag.get('canny_high', '?')
        print(f"[CALIB] canny: low={cl} high={ch}")
        elr = diag.get('edge_lines_raw', '?')
        elf = diag.get('edge_lines_filtered', '?')
        eam = diag.get('edge_angle_max_deg', '?')
        print(f"[CALIB] hough: lines={elr} filtered={elf} angle_max={eam}\u00b0")

        if fr == 'no_edges':
            print(f"[CALIB] \u2717 FAIL: no edges found (Canny returned empty)")
            return
        if fr == 'hough_insufficient_lines':
            print(f"[CALIB] \u2717 FAIL: HoughLinesP found only {elf} line(s) (need \u22653)")
            if elr is not None:
                print(f"[CALIB]   hint: raw Hough lines={elr}, filtered by angle<{eam}\u00b0={elf}")
            return
        if fr == 'hough_no_lines':
            print(f"[CALIB] \u2717 FAIL: HoughLinesP returned 0 lines")
            return

    if diag.get('calibrated'):
        osrc = diag.get('origin_source', '?')
        print(f"[CALIB] origin: {osrc}")
        method = diag.get('method', '?')
        print(f"[CALIB] method: {method}")
        angle = diag.get('angle_rad', 0.0)
        origin = diag.get('origin', (0.0, 0.0))
        d = diag.get('dir', (0.0, 0.0))
        print(f"[CALIB] \u2713 SUCCESS: angle={angle:.4f}rad dir=({d[0]:.3f},{d[1]:.3f}) origin=({origin[0]:.0f},{origin[1]:.0f})")
        return

    print(f"[CALIB] \u2717 FAIL: no detection")


def _make_overlay(frame_bgr: np.ndarray, calibrator: PendulumCalibrator, result) -> np.ndarray:
    overlay = frame_bgr.copy()
    h, w = overlay.shape[:2]

    contour = calibrator.last_contour
    if contour is not None:
        cv2.drawContours(overlay, [contour], -1, (0, 255, 0), 2)

    rect = calibrator.last_rect
    if rect is not None:
        box = cv2.boxPoints(rect)
        box = box.astype(np.int32)
        cv2.drawContours(overlay, [box], 0, (0, 0, 255), 2)
        cx, cy = rect[0]
        cv2.circle(overlay, (int(cx), int(cy)), 5, (255, 0, 0), -1)

    if result.calibrated:
        length = max(h, w)
        cx = int(result.origin_x)
        cy = int(result.origin_y)
        ex = int(cx + length * result.dir_cos)
        ey = int(cy + length * result.dir_sin)
        sx = int(cx - length * result.dir_cos)
        sy = int(cy - length * result.dir_sin)
        cv2.line(overlay, (sx, sy), (ex, ey), (0, 255, 255), 2)

    status = "\u2713 SUCCESS" if result.calibrated else "\u2717 FAIL"
    color = (0, 255, 0) if result.calibrated else (0, 0, 255)
    cv2.putText(overlay, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

    return overlay


def _show_lcd(disp: maix.display.Display, overlay: np.ndarray, debug_frame: np.ndarray):
    oh, ow = overlay.shape[:2]
    dh, dw = debug_frame.shape[:2]

    target_h = oh // 2
    if dw != ow or dh != target_h:
        debug_resized = cv2.resize(debug_frame, (ow, target_h), interpolation=cv2.INTER_NEAREST)
    else:
        debug_resized = debug_frame

    combined = np.vstack([overlay, debug_resized])
    img = maix.image.cv2image(combined, bgr=True)
    disp.show(img)


def _print_help():
    print("[CALIB] commands:")
    print("[CALIB]   Enter             capture & calibrate with current params")
    print("[CALIB]   key=value         set parameter (e.g. calib.min_aspect_ratio=3.0)")
    print("[CALIB]   ?                 show current config")
    print("[CALIB]   save              write config to calib_debug.yaml")
    print("[CALIB]   help              this help")
    print("[CALIB]   q / Ctrl+C        quit")


def main():
    if not os.path.exists(_YAML_PATH):
        print(f"[CALIB] \u2717 {_YAML_PATH} not found", file=sys.stderr)
        sys.exit(1)

    config = _load_config(_YAML_PATH)
    last_mtime = os.path.getmtime(_YAML_PATH)

    cam_cfg = config.get('camera', {})
    try:
        cam = maix.camera.Camera(
            width=cam_cfg.get('width', 1280),
            height=cam_cfg.get('height', 352),
            fps=cam_cfg.get('fps', 60),
        )
    except Exception as e:
        print(f"[CALIB] \u2717 camera init failed: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        disp = maix.display.Display()
    except Exception:
        disp = None
        print("[CALIB] ! no display available, LCD preview disabled")

    frame_no = 0

    print("[CALIB] ===== Pendulum Calibration Debug Tool =====")
    _print_help()
    print("[CALIB] ===========================================")

    try:
        while True:
            try:
                cur_mtime = os.path.getmtime(_YAML_PATH)
                if cur_mtime > last_mtime:
                    config = _load_config(_YAML_PATH)
                    last_mtime = cur_mtime
                    print("[CALIB] config reloaded from calib_debug.yaml")
            except Exception:
                pass

            try:
                line = input(">> ").strip()
            except (EOFError, KeyboardInterrupt):
                print()
                break

            if not line:
                pass
            elif line == 'q':
                break
            elif line == '?':
                _print_config(config)
                continue
            elif line == 'save':
                _save_config(_YAML_PATH, config)
                last_mtime = os.path.getmtime(_YAML_PATH)
                print("[CALIB] config saved to calib_debug.yaml")
                continue
            elif line == 'help':
                _print_help()
                continue
            elif '=' in line:
                k, v = line.split('=', 1)
                _update_config(config, k.strip(), v.strip())
                continue
            else:
                print(f"[CALIB] \u2717 unknown command: {line}")
                continue

            try:
                raw = cam.read()
                if raw is None:
                    print("[CALIB] \u2717 camera read failed")
                    continue
            except Exception as e:
                print(f"[CALIB] \u2717 camera read error: {e}")
                continue

            try:
                rgb = maix.image.image2cv(raw, ensure_bgr=False, copy=True)
                frame_bgr = rgb[:, :, ::-1]
            except Exception as e:
                print(f"[CALIB] \u2717 image2cv error: {e}")
                continue

            frame_no += 1

            cal_cfg = config.get('calib', {})
            calibrator = PendulumCalibrator(
                frame_w=frame_bgr.shape[1],
                frame_h=frame_bgr.shape[0],
                binary_threshold=cal_cfg.get('binary_threshold', 127),
                min_contour_area_ratio=cal_cfg.get('min_contour_area_ratio', 0.04),
                max_contour_area_ratio=cal_cfg.get('max_contour_area_ratio', 0.55),
                min_aspect_ratio=cal_cfg.get('min_aspect_ratio', 1.0),
                canny_low=cal_cfg.get('canny_low', 50),
                canny_high=cal_cfg.get('canny_high', 150),
                hough_threshold=cal_cfg.get('hough_threshold', 50),
                hough_min_line_len=cal_cfg.get('hough_min_line_len', 150),
                edge_angle_max_deg=cal_cfg.get('edge_angle_max_deg', 15),
                column_threshold=cal_cfg.get('column_threshold', 180),
            )

            try:
                result = calibrator.calibrate(frame_bgr)
                diag = calibrator.get_last_diagnostics()
            except Exception as e:
                print(f"[CALIB] \u2717 calibration error: {e}")
                traceback.print_exc()
                continue

            _print_diagnostics(diag, frame_no)

            overlay = _make_overlay(frame_bgr, calibrator, result)

            if disp is not None:
                try:
                    debug_frame = calibrator.get_debug_frame()
                    _show_lcd(disp, overlay, debug_frame)
                except Exception as e:
                    print(f"[CALIB] ! display error: {e}")

            dbg_cfg = config.get('debug', {})
            if dbg_cfg.get('save_frame', True):
                try:
                    cv2.imwrite("/root/calib_debug_frame.png", overlay)
                except Exception as e:
                    print(f"[CALIB] ! save frame error: {e}")
            if dbg_cfg.get('save_binary', True):
                try:
                    cv2.imwrite("/root/calib_debug_binary.png", calibrator.get_debug_frame())
                except Exception as e:
                    print(f"[CALIB] ! save binary error: {e}")
            if dbg_cfg.get('save_column_binary', True):
                try:
                    col_bin = calibrator.last_column_binary
                    if col_bin is not None:
                        cv2.imwrite("/root/calib_debug_column.png", col_bin)
                except Exception as e:
                    print(f"[CALIB] ! save column binary error: {e}")
    finally:
        try:
            cam.close()
        except Exception:
            pass


if __name__ == '__main__':
    main()
