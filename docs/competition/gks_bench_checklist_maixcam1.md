# 工科赛近端视觉 · MaixCAM-Pro 回移植台架清单（feat/maixcam1-port）

> 设备: **MaixCAM-Pro**（CVI cv181x，有触屏）· 对端: panzer_Brain USART1（主站）
> 分支: `feat/maixcam1-port`（自 `EOD2026-NearCam` 的 `0f4db9b` 切出）
> 协议: panzer_links_protocol.md v2.1 —— **与 CAM2 逐字节一致，Brain 侧无需任何改动**

## 0. 与 CAM2 的差异速查

| 项 | MaixCAM2（基线） | MaixCAM-Pro（本分支） |
|---|---|---|
| HAL 平台 | `hal/platforms/maixcam2/` | `hal/platforms/maixcam1/` |
| UART 模块 | `maix.peripheral.uart` | `maix.uart`（`from maix import uart`） |
| 串口/引脚 | `/dev/ttyS4` A21(TX)/A22(RX) | **`/dev/ttyS0` A16(TX)/A17(RX)** |
| 人质模型 | `eod_hostage_320.axmodel` + `nn.YOLO11` | **`model_239147.cvimodel` + `nn.YOLOv5`** |
| 实测帧率上限 | 推流率≈相机帧率 | **约 30fps**（固件至少双缓存，取图延时 30+ms） |
| 镜头 | 固定焦距 | M12 手动调焦（靶标/爆炸物几何需重标） |
| 屏幕/触屏 | 有 | 有（调参页与四页路由原样保留） |

⚠ **模型格式互不通用**：`.axmodel`（Axera）在 CAM1 上跑不了，`.cvimodel`（CVITEK）在 CAM2 上跑不了。
CAM1 的 mud 用 `model =` 键（CAM2 是 `model_npu =`），两者同名不同义，别照抄。

## 1. 部署

1. 模型（CAM1 专用，与 mud **同目录同名**配套）:
   ```
   cp assets/models/hostage_yolov5_cam1/* /root/models/
   # -> /root/models/model_239147.mud + model_239147.cvimodel
   ```
   路径须与 `project_config.yaml` `ai.models[0].model`（`/root/models/model_239147.mud`）一致。
   缺失时 RESCUE 订阅回 NACK NOT_READY（日志可见），EXPLOSIVE/TARGET 不受影响。
2. UART 接线: **UART0 A16(TX)/A17(RX)** ↔ Brain USART1，交叉连接，共地; `/dev/ttyS0` @ **115200**。
   - A16/A17 **出厂默认即 UART0_TX/UART0_RX**（`pinmap.md:42`），代码会**跳过** pinmap
     复用。启动日志出现 `[CAM1] UART /dev/ttyS0 使用出厂默认引脚功能(A16/A17)，跳过 pinmap`
     属正常，不是告警。
   - ⚠⚠ **TX(A16) 是启动模式检测引脚，拉低 = 无法开机**。做 3.3V↔5V 电平转换时
     **务必确保 TX 默认不拉低**（保持浮空或用电平转换芯片）。
     **上电起不来时第一件事：查 A16 是否被拉低。**
   - ⚠ **开机日志干扰**：UART0 开机打印日志直到出现 `serial ready`。Brain 侧应丢弃
     开机段数据（协议有 `AA 55` 帧头，一般能自动丢弃，但建议显式丢弃前若干秒）。
   - ⚠ **与 maix protocol 抢线**：UART0 同时是系统默认 maix protocol 串口，占用后
     MaixVision IDE 调试通道会与本链路冲突。台架调试改用 **SSH**（`ssh root@<设备IP>`）
     或断开 IDE。`app.need_exit()` 走 maix通道，占用 UART0 时应用**可能无法从 IDE
     正常停止** —— 急停直接断电。
   - 若日后改回 UART1（A19/A18）：A19 出厂功能是 GPIOA19/PWM7，**必须** pinmap 复用
     （代码已内置 `A19→UART1_TX` / `A18→UART1_RX`，TX/RX 都要设），否则串口打开但
     收不到任何字节。
3. 阈值文件 `/my-threshold/my_64threshold.txt`:组序 = 爆红/爆绿/爆蓝/靶红/靶绿/靶蓝。
   **CAM1 传感器与 CAM2 不同，LAB 阈值必须现场重标，不可直接拷 CAM2 的值。**
   现场用 KEY_OK 切调参页（爆/靶页）或 `python run.py debug color` 触屏调参后 SAVE 生成。
   - ⚠ **调参页已默认降采样分析**（`gks.tuner.scan_scale: 0.5`）以解决 CAM1 卡顿，
     信息栏末尾显示 `s:0.5x`。`max:{area}px` 已是**全帧口径**（已还原，可照常按
     40~50% 反推 `explosive.area_threshold`）。要退回全帧对照改成 `1.0`。
4. `project_config.yaml` 已切为 `platform: maixcam1`; 打包清单 `app.yaml` 已含 maixcam1 五个文件与 cvimodel。

## 2. 上电自检（无 Brain，USB-TTL 环回/抓包）

| # | 检查项 | 预期 |
|---|---|---|
| 1 | 屏幕显示 HUD | 状态栏 `MODE:IDLE  LINK:--`，无黑屏（显示不依赖模型加载） |
| 2 | 日志 `cam w:640 h:480` | 相机 640×480 开启成功 |
| 3 | **无降级告警** | 启动日志**不得**出现 `camera.read(block=...) unsupported` —— 出现即表示固件老到不认非阻塞读，停帧保护已降级为阻塞读（见 §5） |
| 4 | 相机构造级别 | 日志 `[CAM1] camera open level=1` 表示 fps/buff_num 生效；若出现 `DEGRADED level=3/4` 则 fps 被丢弃（MaixPy 对 ≤1280x720 默认 80fps），实际帧率以实测为准 |
| 5 | 引脚复用 | 不得出现 `pinmap ... 失败` / `未能确认引脚复用`；出现则 UART1 不通，优先查接线与固件 |
| 6 | 硬件看门狗 | **`project_config.yaml: wdt_enabled: false`（默认）** —— 启动日志出现 `[WDT] 未武装`。原因：MaixCDK 的 `WDT::stop()` 是**空实现**（只打 `this operation is not supported` 却 `return 0`），而 `WDT(0, 10000)` 构造即启动、进程退出不会让它停 ⇒ 武装了就**关不掉**，退出后 ~10s 设备硬复位。台架判据：**重启发生在退出后约 10 秒、日志最后一行却是干净的 `program exited`** —— 这是看门狗到点的签名，别误判成供电/固件问题。要死锁兜底才置 true，并接受"退出后 →10s 重启"；判别本机能否真正关掉用 `python scripts/wdt_probe.py info`（看 nowayout）。回归测试 `tests/test_wdt_disarm.py` |
| 7 | 人质模型 | 日志 `[CAM1] Switched to model 'eod_hostage' (/root/models/model_239147.mud) labels=['yao_gu', 'yuan_zhu', 'yuan_tai']`；缺失仅告警不崩溃，RESCUE 订阅会回 NACK NOT_READY |
| 8 | 协议向量对拍（PC 侧已测） | 订阅 `AA 55 06 20 01 32 00 E7 49` → ACK `AA 55 06 21 01 3C 0A 13 B8` |

> **关于日志级别**：`app/main.py` 把 root logger 钉在 `logging.WARNING`，
> `logger.info` 在设备上**不会打印**。因此上述 AI 注册/切换、相机构造级别、
> pinmap 结果都刻意用 `warning` 级别输出—— 台架 grep 不到时先确认日志级别。

## 3. 订阅链路（接 Brain 或 USB-TTL 模拟主站）

| # | 发送 | 预期 |
|---|---|---|
| 1 | 5B 订阅 `[0x01, 50, 爆色, 靶色, 形状]` | ACK `[0x01,60,10]`；HUD `MODE:EXPLOSIVE LINK:OK`；**推流率≈相机帧率（CAM1 上限约 30fps）**，无新结果时按 50ms 推心跳（下限 30ms） |
| 2 | 切`0x02` TARGET | 归一化误差定点 `err_fixed` = (cx-w/2)/(w/2) ×10000，clamp ±1 |
| 3 | 切 `0x03` RESCUE | 三形状坑位 `pos` 1..3；模型缺失则 NACK NOT_READY |
| 4 | STOP `0x23` | 停推流，HUD 回 `MODE:IDLE` |
| 5 | 拔线/主站停发 | 1s 内停止推流 |

⚠ **帧率预期必须与 Brain 侧对齐**：CAM1 推流上限约 30fps，Brain 的 1s stale 判定与看门狗
不受影响，但任何"按 50fps 节奏等数据"的假设都要下调。

## 4. 排爆/人质硬要求（沿用CAM2 规范）

- 排爆与人质都启用**未见者补位**（`infer_missing: tail`）⇒ 每次出现日志 `latch … inferred=True`
  **必须核对 PIT 与实物一致**。补位只推出"是什么"、推不出"在第几坑"，前提不成立会
  **静默锁出错误坑位**（转台打错坑→抓错），比"锁不上"更危险。
- CAM1 排爆帧率低于 CAM2，若帧率明显不足：先把 `gks.explosive.scan_scale` 从 0.5 降到 0.35
  再评估（面积门槛按 scale² 折算、cx按 1/scale 还原，识别语义不变）。
- 靶标 ROI/误差中心按 **640 宽**标定，与旧上端一致；CAM1 镜头不同，**几何需重标**。

## 5. 已知降级（必读）

`MaixCam1Camera.read_raw` 依赖 `cam.read(block=False)` 做停帧判定（契约见
`framework/hal/interface/camera.py::read_raw`）。若 CAM1 固件不认该 kwarg，代码会：

1. 首次调用探测到 `TypeError` → 置 `_block_kwarg_ok = False`，打 warning；
2. 该拍**补一次阻塞读**把帧拿回来（否则首帧会凭空丢成 None）；
3. 之后每拍走阻塞读 —— **相机真挂死时线程会卡在 read()，1s 陈旧归零保护不会触发**。

**验收判据**：启动日志无该 warning = 固件支持非阻塞读，保护完整。
若出现 warning，说明固件太老，**升级固件**（官方建议升级到最新版本，否则部分 API 不可用），
不要带着降级上台架。

## 6. 回归测试（PC 端，改平台后必跑）

```bash
python -m pytest tests/          # 期望 131 passed
```

协议层/任务处理器/协调器用例与平台无关，**换平台后应全部继续通过** —— 这是"只改了
HAL、没碰协议与视觉语义"的最快信号。CAM1 专属用例见
`tests/test_gks_review_fixes.py::test_maixcam1_*`：

| 用例 | 守住什么 |
|---|---|
| `test_maixcam1_read_raw_returns_none_on_stall` | 停帧不得回退旧帧（CAM2 的 2026-09-28 教训） |
| `test_maixcam1_read_raw_converts_new_frame` | RGB→BGR 翻转 + `last_raw` 与 numpy 同帧 |
| `test_maixcam1_falls_back_when_firmware_lacks_block_kwarg` | 固件降级标记 + 当拍补阻塞读（否则首帧丢） |
| `test_maixcam1_open_degrades_to_minimal_signature` | 构造逐级降级能走通 |
| `test_maixcam1_uart_ttyS0_skips_pinmap` | 本项目用 UART0：A16/A17 出厂即UART0 功能，**不应**调 pinmap |
| `test_maixcam1_uart_ttyS1_still_sets_pinmap` | 防将来误删 UART1 复用（A19/A18 出厂**不是** UART1 功能） |
| `test_maixcam1_detect_handles_none_objects` | `detect()` 返回 None 不得崩视觉线程 |
| `test_maixcam1_missing_model_class_fails_loudly` | 缺 YOLO11 时显式失败，不跨类型兜底吐错误坐标 |
| `test_tuner_analyze_downscaled_matches_fullframe` | 调参页降采样与全帧语义等价（mask 尺寸/命中数/bbox/面积口径） |
| `test_tuner_analyze_area_uses_inv_squared` | 面积还原方向：0.5 档须≈全帧值（防误乘 scale² 差 16 倍） |
