# EOD2026 现场标定指南

> 所有阈值在 `modules/zw_opencv_module/config/eod_config.yaml`，代码默认值只是 MaixPy
> 参考起点。2025 省一经验：**每色备两套阈值**（场地不同区域光照不同），比赛日提前 30 分钟重标。
> 标定前先锁曝光：`cam.exp_mode(Manual)`（或设 exposure/gain 即转手动）+
> `cam.awb_mode(AwbMode.Manual)` + `set_wb_gain([R,Gr,Gb,B])`——AE/AWB 漂移会让阈值失效。

## 1. LAB 颜色阈值（colors / bucket / hostage_colors / home_colors）

格式：`[L_min, L_max, A_min, A_max, B_min, B_max]`，OpenMV 惯例（L 0~100，A/B 有符号
-128~127），检测器内部自动转 OpenCV 编码，**yaml 里填 MaixVision 阈值编辑器读数即可**。

流程：
1. MaixVision 连设备，用"阈值编辑器"对目标物框选，把 6 元组填入 yaml 对应色；
2. 用 `run.py debug`（PC+USB 相机）或设备上跑 `eod_align`，确认框选稳定；
3. 同一颜色在场地两个区域各测一次，不稳就配两套（改代码读两套的逻辑占位在
   `EodColorDetector.candidates`——直接放两个同 id 条目，检测器按期望色优先合并）。
4. `blob.min_area_px` 按最近工作距离的目标像素面积的一半设；`edge_margin_px` 防场地
   边界杂物，现场看画面边缘有没有误检调。

## 2. 夹取/放置区（zones.grab / zones.place）

归一化矩形 `[x0, y0, x1, y1]`（帧宽高比例）。标定：
1. 机械臂就位，把炸弹/人质摆到**夹爪恰好可达**的位置；
2. 看 `eod_align` 调试画面里 BLOB 底边中点（BOMB/HOSTAGE 用 bottom 锚点）的落点；
3. 以若干次摆放的落点包络设矩形。READY 判据 = 锚点入区 且 MATCH，N=5 帧锁存。

## 3. 靶标环半径（target.ref_target_width_px / ring_radii_px）

1. 相机架在打靶工作距离 D0，正对靶标拍照；
2. 量靶色区域 bbox 宽 → `ref_target_width_px`；量 8/9/10 三环半径（px）→ `ring_radii_px`；
3. ring_id 估算按 `bbox宽/ref宽` 随距离缩放，只作策略参考，命中判据以 HIT_CONFIRMED 为准。

## 4. HIT 启发式启用（target.hit_detect_enabled）

默认**关**。启用前验证：
1. 锁定靶标（FOUND 稳定 ≥ lock_frames=15 帧）；
2. 人工放倒人像：应在 `hit_lost_frames`=12 帧内触发 HIT（区域消失）或面积塌缩
   `< hit_area_ratio × 锁定面积`；
3. 人像倒下但不遮挡靶色区域的布置会导致漏检——出现该情况就停用启发式，尽快上
   "人像立/倒"两类 YOLO 模型（`_DATA_TYPE_MODEL` 里挂 nick，自动热切换）。

## 5. 形状分类阈值（shape.*）

对三类实物各拍 10 张轮廓，跑 `tests/test_eod_shape.py` 同款 features 输出
（top/bottom/mid_ratio、slope_ratio、rectangularity），按实测分布收放 gate：
- 圆柱：`cylinder_end_min`（两端宽比下限）
- 圆锥/圆台：`cone_slope_min`（剖面回归斜率下限）+ `cone_small_end_max`（小端上限）
- 腰鼓：`waist_ratio_max`（中腰上限）
边界样本（圆台 vs 矮圆柱）分不开就记下特征值，回桌面对比后微调；实在重叠就等 YOLO。

## 6. QR（qr.min_valid_frames 在 QrTaskProcessor 构造参数，默认 2）

实测最远可解码距离、最大倾角、最大平移速度；READ_QR 阶段 MCU 停车/低速。超过
20s 读不到 → MCU 会重订一次，检查是不是曝光反光（哑光贴膜可解）。

## 7. px_per_deg（激光对准，见 laser_aim.md §2）

两点标定法：D=1000mm、S=500mm 标志点 → `px_per_deg = P / (2*atan(S/2D))`。
标完做一次闭环验证：靶标偏 5°，误差换算应闭合到 ±5% 内。

## 8. 串口（project_config.yaml uart_defaults）

- 端口 `/dev/ttyS4`（A21=TX，A22=RX），**115200 8N1**（官方警告其他波特率不可靠，勿改）。
- 联调用 `tests/test_protocol_eod.py` 的金向量做逻辑分析仪/对端对拍。
- MCU 侧：panzer UART4 同波特率，接线 TX↔RX 交叉、共地。
