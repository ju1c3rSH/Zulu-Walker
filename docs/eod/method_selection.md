# EOD2026 视觉方法选型（调研结论）

> 2026-09-06 定稿。调研输入：MaixPy v4 官方文档（F:\MaixPy-main）、GitHub 历年赛题仓库、
> **2025 年同一赛题省一等奖开源仓**（任务码语义与本赛题完全一致）。

## 0. 最重要的发现

**SakuraHop/explorer-eod-rescue-robot**（2025 广东省工科大学生实验综合技能竞赛「反恐排爆救援机器人」省一）：
- 任务码语义与我们一致：QR 三位 = 炸弹色 / 靶色 / 人质形状（1圆柱 2圆锥·圆台 3腰鼓）。
- 其架构：前置 OpenMV 扫码 + LAB 色块，臂端 OpenMV 最终 X/Y，**K230 跑 3 类量化 NN 检测形状**，UART 固定帧给 STM32，MCU 命令字节切换视觉模式。
- 对我们的印证：**MCU 驱动视觉模式切换（订阅制）是被冠军验证的正确姿势**；LAB find_blobs + 现场重标阈值足够打颜色；形状识别冠军用的是 NN。

## 1. 各子任务选型

| 子任务 | 选定方案 | 兜底方案 | 依据 |
|---|---|---|---|
| 二维码 | CPU `cv2.QRCodeDetector` + CLAHE（处理器 `QrTaskProcessor`），连续 2 帧一致才锁存 | 换 NPU `image.QRCodeDetector`（60+fps@320，MaixPy≥4.7.9） | 冠军用 find_qrcodes + N 次一致锁存；CPU 解码避免与 YOLO 抢 NPU；QR 是独立任务阶段（单活跃订阅天然时分复用） |
| 颜色识别 | LAB `find_blobs` 同款流程（`EodColorDetector`），AE/AWB 锁死，**每色备两套阈值现场重标** | 小型 YOLO（MaixHub 在线训练） | 冠军即 LAB 方案；NN 仅在光照恶劣时才必要 |
| 形状识别 | **第一版：轮廓几何分类**（`eod_shape` 竖直宽度剖面：圆柱=等宽 / 圆锥=单调渐变(含圆台) / 腰鼓=中部收腰），零训练当天可用；**正式版：yolo11n 3 类自训模型**经 `MaixCam2AI` 热切换插槽替换 | MaixHub 在线训练 | 冠军用 NN（3 类 320×320）——NN 是被验证的正式答案；几何法做基线和 sanity check |
| 激光对准 | 视觉只出误差流（±5000 归一化），**闭环在 MCU**：px→deg→Emm tick，死区+N 帧锁存才准发射（见 `laser_aim.md`） | 视觉端自闭环发 LOCKED 事件 | 2023/2025 电赛 E 题全为此模式（每轴 PID on 像素误差） |
| 夹取就绪 | 中心窗 + blob 锚点（底边中点）入夹取区 + 尺寸阈值，N 帧锁存 = `READY_TO_PICK`；MCU 再叠加里程计距离 | blob 像素宽→针孔测距 `dist ≈ f·W_real/w_px` | 冠军 main.c 就是 `OpenMV_data4 ∈ [162,193]` 中心窗 + 编码器距离双条件 |

## 2. MaixCAM2 平台结论（全部经文档/社区核实）

1. **串口锁 115200**：官方 wiki 明确警告其他波特率"误码率可能很高"（uart_clk 200MHz 分频取整）。带宽核算：最大帧 27B×60Hz ≈ 16.2 kbps ≪ 115200，完全够用。设备用 **UART4 `/dev/ttyS4`（A21/A22）**，UART0 是系统控制台不能碰。
2. **YOLO11**：`nn.YOLO11` 需 MaixPy ≥ 4.7.0；模型转换：ultralytics → 固定尺寸 ONNX(640×480) → **裁剪输出节点 `/model.23/Concat_output_0`+`Concat_1`+`Concat_2`**（解码/NMS 留给 MaixPy CPU）→ onnxsim → pulsar2(AX620E, INT8, 同时出 NPU2/VNPU 两个 axmodel) → .mud。社区有网页版转换器（liangzhuohua/maix_converter_platform）免 Docker。注意 sipeed/MaixPy issue #208：按官方手册在 ultralytics 8.4.52 + pulsar2 6/7 下会报错，锁版本。
3. **相机**：OS04D10 1280×720@60 / 2560×1440@30；本项目 640×640@60。曝光锁定：`cam.exp_mode(Manual)`/设 exposure/gain 即转手动；`cam.awb_mode(AwbMode.Manual)` + `set_wb_gain([R,Gr,Gb,B])`。>720p 强制 30fps；`maix_npu_ai_isp=1` 占一半 NPU 换画质，色块方案不需要，建议 0。
4. **性能余量**：YOLO11n 640×640 前向 ≈113fps（NPU 3.2TOPS）；find_blobs 640×480 ≈11ms；`dual_buff=True` 提吞吐但引入一帧延迟——**对准/打靶阶段建议关**。

## 3. 本仓库的落地架构（EOD2026-Spacial 分支）

```
MCU(panzer) ──UART4 115200──  FrameParser(AA55+CRC16-CCITT, 高字节在前)
                                │ CmdRequestEvent(type, min_interval, reserved=期望id)
                                ▼
                       EodCoordinator (app/coordinator.py)
                    单活跃订阅：门控 qr_task / eod_align（经主循环队列）
                    min_interval 限速；QR 一次性事件(内容去重)；5s 主机静默自动怠速
                                │ VisionResult dict（不含像素解析）
        ┌───────────────────────┴───────────────────────┐
  QrTaskProcessor                          EodAlignProcessor（按订阅切模式）
  cv2.QRCodeDetector+CLAHE                 BOMB/TARGET/HOSTAGE/BUCKET/HOME
  连续2帧锁存+三位域校验                    EodColorDetector(LAB) + eod_shape 轮廓分类
                                           N帧锁存 FOUND/MATCH/READY；HIT 启发式(默认关)
```

- 协议数据类型表 0x01~0x06 与 MCU 侧 `panzer/docs/design.md` §2.4 一一对应；**REQUEST 的 reserved 字节承载任务码期望 id**（digit1/2/3），视觉端据此置 COLOR_MATCH/SHAPE_MATCH。
- 测试：`tests/test_protocol_eod.py`（金向量=两边对拍基准）、`test_eod_shape.py`、`test_eod_color.py`、`test_eod_align_latch.py`、`test_coordinator_eod.py`，PC 端 `python -m pytest tests/` 全绿。

## 4. 已知局限 / 下一步

1. 形状几何分类假设**目标正立、相机正视**——侧视会误判，正式方案必须走 YOLO（攒数据：现场拍 200~500 张/类即可起训）。
2. HIT_CONFIRMED 启发式（锁定后区域消失/塌缩）需现场验证后打开 `hit_detect_enabled`；稳妥路径是训"人像立/倒"两类模型。
3. ring_id 环号估算依赖 `ring_radii_px` 标定（随距离缩放），8/9/10 环分辨率受相机角分辨率限制，打靶判据以 MCU 侧 HIT_CONFIRMED 为准，ring_id 仅辅助。
4. QR 在运动中解码率下降：READ_QR 阶段 MCU 应低速/停车（20s 超时足够两次尝试）。

## 5. 参考来源

- 省一同赛题：https://github.com/SakuraHop/explorer-eod-rescue-robot
- MaixPy v4 官方：https://github.com/sipeed/MaixPy （docs/doc/zh；wiki.sipeed.com/maixpy）
- 电赛视觉教程（MaixPy v4）：https://github.com/heyu-233/maixcam-ees-vision-course
- 2023 电赛 E 题激光追踪（云台↔图像坐标映射）：https://github.com/DuRuofu/STM32_Target_Tracking_System
- MaixCAM YOLO 转换网页工具：https://github.com/liangzhuohua/maix_converter_platform
- MaixHub 模型库 YOLO11：https://maixhub.com/model/zoo/453 ；yolo11n→axmodel 讨论：https://maixhub.com/discussion/101244
- sipeed/MaixPy issues #208（pulsar2 转换报错）/ #206（yolo26 加载）
- 2025 电赛 E 题云台（分轴 PID）：https://zhuanlan.zhihu.com/p/1938186896556688218
- 23 年 E 题国一三层控制：https://www.dianyuan.com/eestar/article-13276.html
- OpenMV 形状识别（圆度/顶点法兜底思路）：https://blog.csdn.net/weixin_52385589/article/details/126443973
