# 工科赛近端视觉 · 台架验证清单（EOD2026-NearCam）

> 设备: MaixCAM2（AX630C）· 对端: panzer_Brain USART1（主站）
> 部署形态: `run.py main` / MaixPy 应用包 `gks_nearcam`
> 协议: panzer_links_protocol.md v2.1（副本见本目录 `gks_panzer_links_protocol_v2.1.md`）

## 1. 部署

1. 拷贝模型到设备标准路径（二选一）:
   - 从仓库副本: `assets/models/hostage_yolo11n/{eod_hostage_320.mud, eod_hostage_320.axmodel}` → `/root/models/`（mud 与 axmodel 必须同目录同名配套）;
   - 或从已安装应用: `cp /maixapp/apps/gks_nearcam/assets/models/hostage_yolo11n/* /root/models/`。
   - 路径与 `project_config.yaml` `ai.models[0].model` 一致; 模型缺失时 RESCUE 订阅会回 NACK NOT_READY（日志可见），EXPLOSIVE/TARGET 不受影响。
2. UART 接线: MaixCAM2 UART4 **A21(TX)/A22(RX)** ↔ Brain USART1，交叉连接，共地; `/dev/ttyS4` @ **115200**（官方警告其他波特率不可靠，勿改）。
3. 阈值文件 `/my-threshold/my_64threshold.txt`（可选）: 组序 = 爆红/爆绿/爆蓝/靶红/靶绿/靶蓝; 现场用 `python run.py debug color` 触屏调参后 SAVE 直接生成。
4. LED 点灯 pinmap 已含 maixcam2 分支（IO1_A25），无需改动。

## 2. 上电自检（无 Brain，USB-TTL 环回/抓包）

| # | 检查项 | 预期 |
|---|---|---|
| 1 | 屏幕显示 HUD | 状态栏 `MODE:IDLE  LINK:--`，无黑屏（显示不再依赖模型加载） |
| 2 | 日志 `cam w:640 h:480` | 相机 640×480 开启成功 |
| 3 | 日志 AI 注册 | `Registered model 'eod_hostage' -> /root/models/eod_hostage_320.mud`；缺失时仅告警不崩溃 |
| 4 | 协议向量对拍（PC 侧已测） | 订阅 `AA 55 06 20 01 32 00 E7 49` → ACK `AA 55 06 21 01 3C 0A 13 B8`；数据帧/STOP 见 md §6 |

## 3. 订阅链路（接 Brain 或 USB-TTL 模拟主站）

| # | 发送 | 预期 |
|---|---|---|
| 1 | 5B 订阅 `[0x01, 50, 爆色, 靶色, 形状]` | ACK `[0x01,60,10]`；HUD `MODE:EXPLOSIVE LINK:OK`；**推流率≈相机帧率**（结果驱动），无新结果时按 50ms 推心跳（下限 30ms） |
| 2 | 同参重订阅（Brain 1s 看门狗） | 锁存不清（HUD ORDER 保持），日志无 reset |
| 3 | 换 data_id / 换任务码 | 锁存清空重新锁（HUD LOCK n/2 重新计数） |
| 4 | `CMD_STOP` | 停推，HUD 回 IDLE |
| 5 | 订阅 `0x10`（远端 QR 的 id） | NACK `[0x10, 0x01]` |
| 6 | 订阅 `0x03` 且模型缺失 | NACK `[0x03, 0x02]`（NOT_READY），不推流 |
| 7 | 主站长时间不发指令（订阅保持） | 持续推流不中断、锁存不清（Brain 看门狗只在收不到 CAM 数据时重订阅; 从站不按主站静默 idle） |
| 8 | 停止喂订阅但未 STOP | 订阅期零值心跳不断流（Brain 1s stale 不会触发）; 视觉结果陈旧 >1s 自动归零 |
| 9 | **推流率核对（必做）** | 订阅后对照日志 `[UART TX] DATA_STREAM <Hz>Hz` 与 `[FPS] pipeline: <fps>`：靶标阶段两者应基本相等（误差逐帧输出）; 明显低于帧率说明视觉线程被拖慢或回退到旧限速。DebugConsole `FPS`/`TX` 两行同义 |
| 10 | **停帧归零（必做）** | 订阅期间拔相机 / 遮住镜头不再产生新帧：**1s 内** err/坑位归零（`found=0`），不得持续推冻结的 `found=1`; 恢复出图后立即恢复逐帧输出 |
| 11 | 退路开关 | `project_config.yaml` `gks_stream.push_on_new_result: false` 重启后：推流率回落到 `1/min_interval_ms`（丢中间帧），但心跳与锁存语义不变; `min_push_interval_ms: 16` 可压制突发连推 |

## 4. 三任务现场标定

| 任务 | 检查 | 工具 |
|---|---|---|
| 0x01 爆炸物 | 三色按**首次检出先后**映射坑位（同帧齐见时帧内按左右定序）；连续 2 帧同序 → HUD `ORDER:[...]` + `PIT:n`；`pit_left_is_1` 镜像开关核对左右；**默认 `infer_missing: tail`（两色即锁，2026-09-24 台架决策）** | `run.py debug color` |
| 0x01 补位核对 | **必做（用了推断就必须做）**：让三色按顺序入镜，肉眼确认"最先看见的颜色"报 `PIT:1`；再故意挡住一色，确认日志 `latch … inferred=True` 时 `PIT:n` 仍与实物一致。不一致 → `explosive.infer_missing` 改 `none`（三色齐见才锁，锁不上会超时 FAULT） | 日志 `latch …` + 实物 |
| 0x01 扫描分辨率 | `gks.explosive.scan_scale`：当前 **0.5（320×240 扫描）**，降不了帧率问题时再试 1.0 对比；判据是排爆阶段 `[FPS] pipeline` 要高于推流节奏（50ms/20Hz，实测全分辨率只有 12~19fps）。降采样后 LAB 阈值与面积门槛语义不变（门槛按 scale² 折算、cx 还原到全帧） | 日志 `[FPS] pipeline` |
| 0x01 通道序自检 | **上电必做一次**：把**蓝色**爆炸物放镜头前，HUD 框上的标签必须是 `C3`（蓝）；若显示 `C1` 说明红蓝互换 —— 先看日志 `explosive LAB slots: C1=red[…]`（行序对不对），再看 `blob scan engine` 是否 `maix(C++ find_blobs)`（降采样小图必须 RGB 字节序，见 `_downscale`/`maix_accel._as_rgb888`）。同理红色必须是 `C1`。**注意 `scan_scale<1.0` 才会走这条小图路径** | HUD 标签 + 日志一行 |
| 0x01 时间序核对 | **必做**：按比赛流程让三色逐个入镜，核对"最先看见的颜色"报 `PIT:1`（日志 `seen += n`）；再确认三色齐见时锁存只需 2 帧 | 日志 + 实物 |
| 0x02 靶标 | 靶标进 ROI → HUD `ERR:±xxxx`（=±1 对应 ±10000）；y 轴机械固定只闭环 X；误差**逐帧**推送（推流率≈相机帧率，见 §3 #9）。从站不做平滑/滤波/死区，主站若需滤波自行处理 | 同上 + `gks.target.roi` |
| 0x03 人质 | 形状按**首次检出先后**映射坑位（最先看见=pit1），连续 2 帧同序 → `POS:n`；只见到 2 个时未见者补到顺序尾部（日志 `latch … inferred=True`）；缺模型走零值心跳不崩溃 | 日志 + HUD |
| 0x03 人质顺序核对 | **必做**：按比赛流程走一遍，核对"最先看见的形状"确实落在 `POS:1`；再挡住最后那个坑，确认 `inferred=True` 时 `POS:n` 仍与实物一致。不一致 → `gks.rescue.order_by` / `infer_missing` 退回保守值（`x` / `none`） | 实物 + 日志 `latched_order` |

## 5. 回归基线

- PC: `python -m pytest tests/` 全绿（116 用例: C 库向量移植 + md §6 + Brain 对拍 + 门控/锁存语义 + 通道序回归）。
- 台架: 本清单 3.1~3.8 全部通过后再连整机联调。

## 6. 四页调试 UI 与告警横幅（2026-09 新增, 设备端人工验证）

| # | 检查 | 预期 |
|---|---|---|
| 1 | KEY_OK 从 MAIN 释放沿按三下 | 依序进 EXPLOSIVE → TARGET → RESCUE, 再按回 MAIN |
| 2 | 进任一调试页 | 画面为该调参页独占（视觉暂停推流零值心跳兜底） |
| 3 | 爆炸物页 3 组 LAB 滑条 + 点取色 + MIRROR 开关 | 阈值即时生效到识别预览; MIR 切换立即影响 left_is_1 |
| 4 | 靶标页 ROI 两黄横线 | 画在 target.roi 对应高度, 与 HUD 一致 |
| 5 | 靶标页中线刻度（HUD） | 中线 + ±20/40/80% 刻度, found 时变绿 |
| 6 | SAVE | 写 /my-threshold 热重载, 主链路锁存不清 |
| 7 | 人质页 conf/iou/xcrop 滑条 + RESET | 立即生效; RESET 回代码默认 |
| 8 | 告警横幅 | RESCUE 无模型/LAB 阈值文件缺失/相机停摆 => 红条文案; NACK 后 5s 内显示 reason |
