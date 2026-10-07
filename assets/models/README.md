# assets/models/ 模型目录说明

⚠ **两个平台的模型格式互不通用，拷错就是RESCUE 永久 NACK（`switch()` 失败）。**

| 目录 | 平台 | 模型文件 | 加载类 | mud 关键键 |
|---|---|---|---|---|
| `hostage_yolov5_cam1/` | **MaixCAM-Pro**(cv181x) | `model_239147.cvimodel` | `maix.nn.YOLOv5` | `model =` |
| `hostage_yolo11n/` | MaixCAM2(AX630C) | `eod_hostage_320.axmodel` | `maix.nn.YOLO11` | `model_npu =` |

- **CAM1 只认 `.cvimodel`**（CVITEK 格式），`.axmodel` 跑不了；
- **CAM2 只认 `.axmodel`**（Axera 格式），`.cvimodel` 跑不了；
- 两者`labels` 相同（`yao_gu, yuan_zhu, yuan_tai`），但**权重与解码器不同**，
  跨平台复用不会报错，只会是模型加载失败或输出错误。

`feat/maixcam1-port`（MaixCAM-Pro）分支只用 `hostage_yolov5_cam1/`，部署命令：

```bash
cp assets/models/hostage_yolov5_cam1/* /root/models/
# mud 与 cvimodel 必须同目录同名配套
```

`EOD2026-NearCam`（MaixCAM2）分支只用 `hostage_yolo11n/`，部署命令：

```bash
cp assets/models/hostage_yolo11n/* /root/models/
```

模型缺失时不会崩溃：RESCUE 订阅回 NACK NOT_READY，EXPLOSIVE/TARGET 不受影响。
