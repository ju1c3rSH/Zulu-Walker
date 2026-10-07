from __future__ import annotations

import logging
from typing import Optional, Union

import numpy as np

import maix.image
import maix.nn

from framework.hal.interface.ai import Detection, Keypoint, MaskStats, SegmentResult

logger = logging.getLogger(__name__)

# MaixCAM (cv181x) 的模型类。人质模型走 YOLOv5 + .cvimodel（CVITEK 格式）。
# ⚠ **不要用别的类兜底缺失的类型**：固件缺 YOLOv8/YOLO11 时若回退到 YOLOv5，
#   会用 YOLOv5 的后处理去解 YOLOv8/YOLO11 的输出——**不抛异常、直接吐错误坐标**，
#   比"加载失败"危险得多。缺失一律置None，由 _resolve_class → switch 显式失败。
#   （CAM1 固件其实支持 yolov8/yolo11，见 docs/doc/zh/ai_model_converter/maixcam.md
#     的 model_type 列表；只是有版本门槛，YOLOv8≥4.3.0 / YOLO11 ≥4.7.0。）
_SLOT_CLASSES = {
    "yolov5": maix.nn.YOLOv5,
    "yolov8": getattr(maix.nn, "YOLOv8", None),
    "yolo11": getattr(maix.nn, "YOLO11", None),
    "classifier": getattr(maix.nn, "Classifier", None),
    "nn": getattr(maix.nn, "NN", None),
}
# "auto" 在 CAM1 上按 yolov5 解析（当前唯一的实际模型类型）
_DEFAULT_TYPE = "yolov5"


class MaixCam1AI:
    """MaixCAM / MaixCAM-Pro NPU 推理（人质形状识别）。

    与 MaixCam2AI 的差异（2026-10-07 回移植）:

    1. **模型格式**: CAM2 用 Axera 的 `.axmodel`（mud 键`model_npu`），
       CAM1 只认 CVITEK 的 `.cvimodel`（mud 键 `model`）。当前部署
       `model_239147.cvimodel`（YOLOv5，三类 yao_gu/yuan_zhu/yuan_tai）。
    2. **模型类**: `nn.YOLOv5`（CAM2 是 `nn.YOLO11`）。老固件可能没有
       YOLOv8/YOLO11 —— 缺失时显式失败而非跨类型兜底，见 _SLOT_CLASSES 注释。
    3. **构造参数**: YOLOv5 用 `model=<mud路径>` 关键字。⚠ `dual_buff`
       **CAM1 是支持的**（官方 docs/doc/zh/vision/dual_buff.md 就是 MaixCAM 文档），
       且默认开启；但它会让 `detect()` 返回**上一次**调用的图结果 → 人质坑位
       `first_seen` 定序引入一帧延迟（CAM1 约 30fps，即~33ms 机械位移）。
       故显式关掉，见 switch()。
    4. **性能**: CAM1 NPU 算力与内存都显著低于 CAM2，人质任务 320x240
       输入是可接受的既有配置，勿盲目上640。
    """

    def __init__(self) -> None:
        self._registry: dict[str, dict] = {}   # nick_name -> {path, type, kwargs}
        self._active_name: str = ""
        self._model = None
        self._model_type: str = ""
        self._model_path: str = ""

    # ------------------------------------------------------------------ #
    #  Registry API
    # ------------------------------------------------------------------ #

    @property
    def models(self) -> list[str]:
        return list(self._registry.keys())

    @property
    def active_model(self) -> str:
        return self._active_name

    @property
    def model_type(self) -> str:
        return self._model_type

    def add(self, nick_name: str, model_path: str, model_type: str = "auto", **kwargs) -> bool:
        if nick_name in self._registry:
            logger.warning("Model '%s' already registered, overwriting", nick_name)
        self._registry[nick_name] = {
            "path": model_path,
            "type": model_type,
            "kwargs": kwargs,
        }
        # 关键状态走 warning 而非 info: app/main.py 的 basicConfig 把 root
        # logger 钉在 WARNING，logger.info 在设备上**根本不会打印**，台架
        # 就grep 不到"模型是否注册/切换成功"这类最需要的信息。
        logger.warning("[CAM1] Registered model '%s' -> %s (type=%s)",
                       nick_name, model_path, model_type)
        return True

    def remove(self, nick_name: str) -> None:
        if nick_name not in self._registry:
            logger.warning("Attempt to remove unknown model '%s'", nick_name)
            return
        if self._active_name == nick_name:
            self.unload()
        del self._registry[nick_name]
        logger.info("Removed model '%s'", nick_name)

    def switch(self, nick_name: str) -> bool:
        """加载并激活一个已注册模型。

        CAM1 每次 switch 都从 flash 重新加载 .cvimodel 到 NPU，耗时与
        CAM2 同量级（~200-500ms）。当前只有人质一个模型，不构成问题；
        若将来加第二个模型，注意 RESCUE 订阅期内的首次 switch 会阻塞
        推流一拍。
        """
        if nick_name not in self._registry:
            logger.error("Cannot switch to unknown model '%s'", nick_name)
            return False
        if self._active_name == nick_name and self._model is not None:
            return True  # already active

        self.unload()
        info = self._registry[nick_name]
        path = info["path"]
        model_type = info["type"]
        kwargs = dict(info["kwargs"])

        slot_cls = self._resolve_class(model_type)
        if slot_cls is None:
            logger.error("Unsupported model type '%s' for '%s'", model_type, nick_name)
            return False

        # CAM1: YOLOv5(model=<mud>, dual_buff=False)
        # 显式 dual_buff=False：CAM1 支持该参数且**默认为 True**，而它会让
        # detect() 返回上一次调用的图结果 —— 人质坑位 first_seen 定序与
        # x_crop 像素判断会因此吃到一帧延迟（约 33ms @30fps）。关掉换实时性。
        # 旧固件若不认该 kwarg，用 TypeError 消息特征区分"签名不兼容"与
        # "模型本身坏了"（后者常以 TypeError/ValueError 冒泡），避免无谓重试
        # 一次 NPU 加载（200-500ms）并掩盖真实错误。
        kwargs.setdefault("dual_buff", False)
        try:
            self._model = slot_cls(model=path, **kwargs)
        except TypeError as e:
            msg = str(e)
            if "dual_buff" in msg:
                # 老固件没有 dual_buff：去掉后重试一次
                kwargs.pop("dual_buff", None)
                try:
                    self._model = slot_cls(model=path, **kwargs)
                except Exception as e2:
                    logger.error("Failed to load model '%s' from %s: %s",
                                 nick_name, path, e2)
                    self._model = None
                    return False
            else:
                logger.error("Failed to load model '%s' from %s: %s",
                             nick_name, path, e)
                self._model = None
                return False
        except Exception as e:
            logger.error("Failed to load model '%s' from %s: %s", nick_name, path, e)
            self._model = None
            return False

        self._model_path = path
        self._model_type = model_type
        self._active_name = nick_name
        # 同上：level=warning 才不会被 root logger 的 WARNING 阈值吞掉
        logger.warning("[CAM1] Switched to model '%s' (%s) labels=%s",
                       nick_name, path, self.labels)
        return True

    # ------------------------------------------------------------------ #
    #  Convenience aliases
    # ------------------------------------------------------------------ #

    def load(self, model_path: str, model_type: str = "auto", **kwargs) -> bool:
        return self.add("default", model_path, model_type=model_type, **kwargs) and self.switch("default")

    def unload(self) -> None:
        if self._active_name:
            logger.info("Unloading model '%s'", self._active_name)
        self._model = None
        self._model_type = ""
        self._model_path = ""
        self._active_name = ""

    # ------------------------------------------------------------------ #
    #  Properties
    # ------------------------------------------------------------------ #

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def input_width(self) -> int:
        if self._model is None:
            return 0
        try:
            return self._model.input_width()
        except Exception as e:
            logger.warning("Failed to get input_width: %s", e)
            return 0

    @property
    def input_height(self) -> int:
        if self._model is None:
            return 0
        try:
            return self._model.input_height()
        except Exception as e:
            logger.warning("Failed to get input_height: %s", e)
            return 0

    @property
    def labels(self) -> list[str]:
        if self._model is None:
            return []
        try:
            return self._model.labels
        except Exception as e:
            logger.warning("Failed to get labels: %s", e)
            return []

    @property
    def model_path(self) -> str:
        return self._model_path

    # ------------------------------------------------------------------ #
    #  Inference
    # ------------------------------------------------------------------ #

    def detect(self, frame, **kwargs) -> list[Detection]:
        """Run object detection on frame.

        *frame* 可以是 BGR numpy（常规流水线路径）或 `maix.image.Image`
        （零拷贝路径，传`_raw=True`）。人质任务当前走 numpy 路径。

        无模型/推理失败时返回空 list —— 上层 gks_task_processor 据此
        优雅降级（零值心跳），不崩溃。
        """
        conf_th = kwargs.pop("conf_th", 0.48)
        iou_th = kwargs.pop("iou_th", 0.5)
        _is_raw = kwargs.pop("_raw", False)

        if self._model is None:
            logger.warning("detect() called but no model is loaded")
            return []

        if not hasattr(self._model, "detect"):
            logger.warning(
                "detect() requires a model with detect() method, "
                "current model type=%s (nick_name=%s)",
                type(self._model).__name__, self._active_name or "N/A",
            )
            return []

        try:
            if _is_raw:
                img = frame
            else:
                frame_rgb = frame[:, :, ::-1]
                img = maix.image.cv2image(frame_rgb, bgr=False, copy=True)
        except Exception as e:
            logger.error("cv2image conversion failed: %s", e)
            return []

        try:
            objects = self._model.detect(img, conf_th=conf_th, iou_th=iou_th, **kwargs)
        except Exception as e:
            # 不做"丢掉 iou_th 重试"：CAM1 官方 nn_yolov5.py / yolov5.md 都带
            # iou_th，TypeError 更可能来自 detect 内部（numpy 转换、pts 越界等），
            # 吞掉它再重试只会掩盖真实错误、并悄悄改掉检测语义。
            logger.error("Model detect() failed: %s", e)
            return []

        # ⚠ 必须在 try 之外做`or []`：部分固件无目标时 detect() 返回 None 而非
        #   空 list，`for o in None` 会抛 TypeError 逃出本函数直达
        #   gks_task_processor._process_rescue（该处无 try 包裹）→ 打断整帧处理，
        #   违反上面"返回空 list 供上层优雅降级"的承诺。
        objects = [o for o in (objects or []) if o.score >= conf_th]
        if not objects:
            logger.debug("detect() returned 0 objects")
        else:
            logger.debug("detect() returned %d objects", len(objects))

        results: list[Detection] = []
        for obj in objects:
            kps = self._convert_keypoints(obj, self._model_type)
            angle: Optional[float]
            if hasattr(obj, "angle") and obj.angle != -9999:
                angle = float(obj.angle)
            else:
                angle = None

            # CAM1 YOLOv5 无人质分割需求，seg_mask 分支保留占位以对齐接口
            seg_mask_np: Optional[np.ndarray] = None
            mask_stats: Optional[MaskStats] = None
            if hasattr(obj, "seg_mask") and obj.seg_mask is not None:
                seg_mask_np = self._extract_seg_mask(obj.seg_mask)
                if seg_mask_np is not None:
                    ys, xs = np.nonzero(seg_mask_np > 127)
                    if xs.size:
                        mask_stats = MaskStats(
                            center_x=float(obj.x + xs.mean()),
                            center_y=float(obj.y + ys.mean()),
                            area_px=int(xs.size),
                        )

            results.append(Detection(
                x=obj.x, y=obj.y, w=obj.w, h=obj.h,
                class_id=obj.class_id, score=obj.score, label="",
                angle=angle, keypoints=kps, mask_index=-1,
                seg_mask=seg_mask_np, mask_stats=mask_stats,
            ))

        return results

    def segment(self, frame: np.ndarray, **kwargs) -> list[SegmentResult]:
        """分割推理：CAM1 当前无分割模型，恒返回空（接口占位）。"""
        detections = self.detect(frame, **kwargs)
        results: list[SegmentResult] = []
        for d in detections:
            if d.mask_stats is not None and d.mask_stats.area_px > 0:
                results.append(SegmentResult(
                    class_id=d.class_id,
                    center_x=d.mask_stats.center_x,
                    center_y=d.mask_stats.center_y,
                    area_px=d.mask_stats.area_px,
                    bbox_x=d.x, bbox_y=d.y, bbox_w=d.w, bbox_h=d.h,
                    score=d.score, detection=d,
                ))
        return results

    def classify(self, frame: np.ndarray, **kwargs) -> list[tuple[int, float]]:
        if self._model is None:
            logger.warning("classify() called but no model is loaded")
            return []
        if not hasattr(self._model, "classify"):
            logger.warning("classify() requires a classify() method, current type=%s",
                           type(self._model).__name__)
            return []
        try:
            img = maix.image.cv2image(frame, bgr=True, copy=True)
        except Exception as e:
            logger.error("cv2image conversion failed: %s", e)
            return []
        try:
            results = self._model.classify(img)
        except Exception as e:
            logger.error("Model classify() failed: %s", e)
            return []
        top_k = kwargs.get("top_k", 1)
        out: list[tuple[int, float]] = []
        for item in list(results)[:top_k]:
            if isinstance(item, (list, tuple)):
                out.append((int(item[0]), float(item[1])))
            else:
                out.append((int(item), 0.0))
        return out

    def get_mask(self, index: int = 0) -> Optional[np.ndarray]:
        return None

    # ------------------------------------------------------------------ #
    #  Context manager
    # ------------------------------------------------------------------ #

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.unload()

    # ------------------------------------------------------------------ #
    #  Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _resolve_class(model_type):
        """返回 CAM1 上对应的 MaixPy NN 类；None = 不支持。

        "auto" 在 CAM1 上解析为 yolov5（当前唯一实际模型类型）。
        老固件缺 YOLOv8/YOLO11 时 getattr 已回退到 YOLOv5。
        """
        if model_type == "auto":
            return _SLOT_CLASSES[_DEFAULT_TYPE]
        if model_type in _SLOT_CLASSES:
            return _SLOT_CLASSES[model_type]
        if isinstance(model_type, type):
            return model_type
        logger.warning("Unknown model_type '%s', falling back to yolov5", model_type)
        return _SLOT_CLASSES[_DEFAULT_TYPE]

    @staticmethod
    def _convert_keypoints(obj, model_type: str = "") -> list[Keypoint]:
        kps: list[Keypoint] = []
        if not hasattr(obj, "points"):
            return kps
        pts = obj.points
        if not pts:
            return kps
        offset = 0
        step = 2
        if model_type == "hand_landmarks":
            offset = 8
            step = 3
        elif len(pts) % 3 == 0:
            step = 3
        for i in range(offset, len(pts), step):
            score = float(pts[i + 2]) if step == 3 else 0.0
            kps.append(Keypoint(
                x=float(pts[i]), y=float(pts[i + 1]),
                score=score, id=(i - offset) // step,
            ))
        return kps

    @staticmethod
    def _extract_seg_mask(mask_img) -> Optional[np.ndarray]:
        try:
            mask_np = maix.image.image2cv(mask_img, ensure_bgr=False, copy=True)
            if mask_np.ndim == 3:
                mask_np = mask_np[:, :, 0]
            return mask_np
        except Exception as e:
            logger.warning("Failed to convert seg_mask: %s", e)
            return None
