# -*- coding: utf-8 -*-
"""
MaixPy hardware-accelerated fast paths for EOD detectors (device-only).

On MaixCAM2, MaixPy v4 ships optimized implementations of the same
primitives this project otherwise runs through OpenCV on CPU:

  - maix.image.Image.find_blobs   MaixCV C++ (official benchmark ~11ms @640x480 RGB)
  - maix.image.QRCodeDetector     NPU-accelerated QR (60+ fps @320x224, MaixPy >= 4.7.9)
  - maix.image.Image.find_qrcodes ZBAR/QUIRC CPU decoders (MaixPy >= 4.7.7)

Both operate directly on the raw maix Image exposed by the camera HAL as
`context["camera"].last_raw` — the same zero-copy path AIInferenceProcessor
uses for the NPU — so the accelerated path avoids the per-frame
image2cv numpy conversion entirely.

Every helper degrades gracefully (returns None / False) when maix is not
importable (PC pytest, `run.py debug` with a USB camera); callers then fall
back to the cv2 path unchanged. Engine selection lives in
`eod_config.yaml` under `accel:`.

Threshold convention note: maix find_blobs uses the SAME OpenMV-style LAB
thresholds as `eod_config.yaml` (L 0..100, A/B signed) — passed through
1:1, no conversion (the cv2 path converts internally instead).

Channel-order note: imlib's convert_to_imlib_image treats FMT_BGR888 as
PIXFORMAT_RGB888 (fall-through, no swap), so pixels handed to find_blobs
MUST be real RGB888. Feed BGR data and red/blue silently trade places —
`_as_rgb888()` converts defensively, callers should still pass RGB888.
"""
import logging
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from maix import image as _maix_image
    _HAVE_MAIX = True
except ImportError:
    _maix_image = None
    _HAVE_MAIX = False

_NPU_QR_DETECTOR = None          # lazy singleton (loads NPU model once)
_NPU_QR_PROBED = False


def maix_image_from_context(context: Optional[dict]):
    """Raw maix Image for this frame, or None on PC / when unavailable.

    Mirrors AIInferenceProcessor: the camera HAL caches the last raw frame
    (RGB888 maix Image) which read_raw() also converted to numpy.
    """
    if not _HAVE_MAIX or not context:
        return None
    camera = context.get("camera")
    if camera is None:
        return None
    raw = getattr(camera, "last_raw", None)
    return raw


_BGR_GUARD_LOGGED = False


def _as_rgb888(img):
    """BGR 标记图 -> 真正的 RGB888（避免 imlib 侧红蓝互换）。

    MaixCDK `convert_to_imlib_image()`（components/vision/src/maix_image_ops.cpp）对
    FMT_BGR888 是 fall-through 到 PIXFORMAT_RGB888 —— 源码注释原文就是
    `case Format::FMT_BGR888:    // fall through`，**只认标记、不做通道交换**。
    于是 BGR 字节会被 imlib 当 RGB 去算 LAB: 蓝色目标落进"红色"阈值盒
    （HUD 把蓝色框标成 C1）、红色落进"蓝色"盒、绿色不受影响。
    调用方最好直接交 RGB888（如 gks_task_processor._downscale）; 这里兜底转一次,
    比错一整场的坑位定序便宜。
    """
    global _BGR_GUARD_LOGGED
    fmt_fn = getattr(img, "format", None)
    formats = getattr(_maix_image, "Format", None)
    if not callable(fmt_fn) or formats is None:
        return img
    try:
        bgr_like = [f for f in (getattr(formats, "FMT_BGR888", None),
                               getattr(formats, "FMT_BGRA8888", None)) if f is not None]
        if bgr_like and fmt_fn() in bgr_like and hasattr(img, "to_format"):
            converted = img.to_format(formats.FMT_RGB888)
            if converted is not None:
                if not _BGR_GUARD_LOGGED:
                    _BGR_GUARD_LOGGED = True
                    logger.warning("find_blobs_maix: BGR 标记图已转 RGB888 "
                                   "(imlib 把 FMT_BGR888 当 RGB888, 不转会红蓝互换)")
                return converted
    except Exception as e:
        logger.warning("BGR->RGB888 转换失败(%s), 用原图继续", e)
    return img


def find_blobs_maix(img, thresholds: list, roi=None,
                    area_threshold: int = 400,
                    pixels_threshold: int = 100,
                    merge: bool = False) -> Optional[list]:
    """C++ find_blobs on a raw maix Image.

    Returns plain dicts {x, y, w, h, cx, cy, pixels} (device-safe across
    maix versions) or None when maix/img unavailable (caller falls back
    to the cv2 path).

    `merge` (merge overlapping blobs, OpenMV convention) is attempted first
    and dropped gracefully when the firmware build does not accept the kwarg
    (official docs only pin thresholds/invert/roi/area_threshold/pixels_threshold).
    """
    if not _HAVE_MAIX or img is None:
        return None
    img = _as_rgb888(img)
    try:
        # roi 是 list[int] 参数, 传 None 会被 pybind11 拒绝(TypeError) —— 全帧
        # 检测必须整体省略该 kwargs（EXPLOSIVE 三色扫描即此形态）
        kwargs = dict(
            invert=False,
            area_threshold=area_threshold,
            pixels_threshold=pixels_threshold,
        )
        if roi:
            kwargs["roi"] = list(roi)
        if merge:
            try:
                blobs = img.find_blobs(thresholds=[list(thresholds)],
                                       merge=True, margin=1, **kwargs)
            except TypeError:
                blobs = img.find_blobs(thresholds=[list(thresholds)], **kwargs)
        else:
            blobs = img.find_blobs(thresholds=[list(thresholds)], **kwargs)
    except Exception as e:
        logger.warning("find_blobs_maix failed, falling back to cv2: %s", e)
        return None
    out = []
    for b in blobs or []:
        x, y, w, h = b.x(), b.y(), b.w(), b.h()
        # cxf(): 浮点中心(v4); 老固件回退 cx() 整数 —— 与 cv2 路径的 x+w/2 语义对齐
        cx = b.cxf() if hasattr(b, "cxf") else b.cx()
        out.append({
            "x": x, "y": y, "w": w, "h": h,
            "cx": cx, "cy": b.cy(),
            "pixels": b.pixels(),
        })
    return out


def _npu_qr_detector():
    """Lazily create the NPU QRCodeDetector (MaixPy >= 4.7.9)."""
    global _NPU_QR_DETECTOR, _NPU_QR_PROBED
    if _NPU_QR_PROBED:
        return _NPU_QR_DETECTOR
    _NPU_QR_PROBED = True
    if not _HAVE_MAIX or not hasattr(_maix_image, "QRCodeDetector"):
        return None
    try:
        _NPU_QR_DETECTOR = _maix_image.QRCodeDetector()
    except Exception as e:
        logger.warning("NPU QRCodeDetector unavailable (%s); cv2 fallback", e)
        _NPU_QR_DETECTOR = None
    return _NPU_QR_DETECTOR


def qr_detect_npu(img) -> Optional[list]:
    """NPU QR detection on a raw maix Image -> [{payload, corners}] or None."""
    detector = _npu_qr_detector()
    if detector is None or img is None:
        return None
    try:
        qrcodes = detector.detect(img)
    except Exception as e:
        logger.warning("qr_detect_npu failed: %s", e)
        return None
    out = []
    for q in qrcodes or []:
        corners = None
        try:
            corners = [[float(p[0]), float(p[1])] for p in q.corners()]
        except Exception:
            corners = None
        out.append({"payload": q.payload() or "", "corners": corners})
    return out


def qr_detect_zbar(img, roi=None) -> Optional[list]:
    """CPU ZBAR QR decode on a raw maix Image -> [{payload, corners}] or None."""
    if not _HAVE_MAIX or img is None:
        return None
    try:
        kwargs = {"roi": list(roi)} if roi else {}
        try:
            qrcodes = img.find_qrcodes(
                decoder_type=_maix_image.QRCodeDecoderType.QRCODE_DECODER_TYPE_ZBAR,
                **kwargs)
        except (AttributeError, TypeError):
            # older MaixPy without decoder_type / roi kwargs
            qrcodes = img.find_qrcodes(**kwargs)
    except Exception as e:
        logger.warning("qr_detect_zbar failed: %s", e)
        return None
    out = []
    for q in qrcodes or []:
        try:
            corners = [[float(p[0]), float(p[1])] for p in q.corners()]
        except Exception:
            corners = None
        out.append({"payload": q.payload() or "", "corners": corners})
    return out
