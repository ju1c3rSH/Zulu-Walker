# 工科赛三链路应用协议（panzer_Brain 视角）

> 适用项目：工科赛排爆机器人（panzer_Brain 中控 + panzer 底盘 + 近/远端 MaixCAM）
> 帧结构、CRC16 定义沿用 [`protocol.md`](protocol.md)（AA 55 + Length + Type + Payload + CRC16-CCITT），
> MCU 侧解析/发送由本目录 `Middlewares/serial_protocol` 框架承载。
> 本文只定义**三条链路各自的应用层语义**：订阅码、payload 布局、时序。

| 链路 | 对端 | 交互模式 |
|---|---|---|
| USART1 | 近端 MaixCAM（爆炸物/靶标/人质识别） | **订阅制**：订阅后持续推送，直到 STOP |
| USART2 | 远端 MaixCAM（二维码扫描） | **一次性（常扫缓存）**：远端时刻扫码画框并缓存，请求到达即回最新有效 QR 任务码一次 |
| UART4 | panzer 底盘下位机 | **纯 request**：中控下发运动请求，底盘执行段表并回报完成事件 |

---

## 1. 通用定义

### 1.1 帧类型（沿用 Master-Slave v3.0）

| Type | 名称 | 方向 | Payload |
|---|---|---|---|
| 0x20 | CMD_REQUEST | 主→从 | [data_id, min_interval_ms, reserved] |
| 0x21 | CMD_ACK | 从→主 | [data_id, max_freq_hz, payload_size]（底盘链路含子命令回显：[0x21, sub_cmd_echo, status]） |
| 0x22 | CMD_NACK | 从→主 | [data_id, reason]（相机链路 2B，v2.2 起不变）；底盘链路 v2.2 升级为 [0x21, sub_cmd_echo, reason]（3B，legacy 2B 仍须被接受，按 payload 长度判别） |
| 0x23 | CMD_STOP | 主→从 | 空 |
| 0x24 | DATA_STREAM | 从→主 | [seq, data_id, sub_payload(8B)] |

DATA_STREAM 的 payload **统一 10 字节**：`seq(1) + data_id(1) + sub_payload(8)`，
多字节字段小端序。

### 1.2 data_id 全局表（三条链路共用一个编号空间）

| data_id | 名称 | 链路 | 说明 |
|---|---|---|---|
| 0x01 | DATA_EXPLOSIVE | USART1 | 爆炸物识别结果 |
| 0x02 | DATA_TARGET | USART1 | 靶标识别结果 |
| 0x03 | DATA_RESCUE | USART1 | 人质识别结果 |
| 0x10 | DATA_QR_CODE | USART2 | 二维码任务码（一次性） |
| 0x21 | CMD_CAR_CTRL | UART4 | 运动请求 REQ_NAV/REQ_STOP/REQ_TURNTABLE（下行，子命令式） |
| 0x22 | EVT_CAR_EVENT | UART4 | 底盘事件上报（NAV_DONE / NAV_FAULT / TURN_DONE） |

### 1.3 CRC16

CRC16-CCITT：poly `0x1021`，init `0xFFFF`，无反射、无终异或，
计算范围 `Type..Payload 末字节`，高字节在前。与 [`protocol.md`](protocol.md) 一致。

### 1.4 NACK 原因码枚举（v2.2，三条链路统一编号空间）

| 值 | 底盘/ZW 宏名 | 相机端宏名（语义别名） | 适用链路 | 语义 |
|---|---|---|---|---|
| 0x01 | NACK_FORMAT | NACK_UNSUPPORTED_TYPE | 相机+底盘 | 格式不受支持：payload 长度不足 / 帧类型不是 CMD_REQUEST / data_id 未知或不支持（相机侧 0x01 语义维持不变） |
| 0x02 | NACK_BUSY | NACK_NOT_READY | 相机+底盘 | 从端忙或状态未就绪（对齐相机与 Zulu-Walker 现有 0x02=NOT_READY） |
| 0x03 | NACK_PARAM | —（底盘专属） | 底盘 | 参数越界：子命令存在但 arg 非法（nav_id≥8、gear>3、turn_code>7、tracking_speed_level/stop_mode>1 等） |
| 0x04 | NACK_UNKNOWNSUB | —（底盘专属） | 底盘 | data_id=0x21 合法但 sub_cmd 不在 §4.2 定义表（底盘回显收到的原 sub 值供定位） |

- 相机链路 2B 帧 `[data_id, reason]` 只占 0x01/0x02 两档，沿用 v1.x 语义，不改格式。
- 底盘链路 v2.2 起 3B 帧 `[0x21, sub_cmd_echo, reason]`：`sub_cmd_echo` =
  被拒请求的 sub_cmd 原值（帧类型/data_id 层面被拒时填 0x00），中控据此精确定位是哪条请求失败。
- master（panzer_Brain）解析规则：NACK payload 3B → 取 `sub_cmd_echo` 联动；
  2B（legacy/相机）→ 只计数观测，不联动。

---

## 2. 链路一：USART1 ↔ 近端 MaixCAM（订阅制持续推送）

### 2.1 订阅与退订

```
中控 → CAM:  CMD_REQUEST [data_id, min_interval_ms, reserved]
             data_id ∈ {0x01 爆炸物, 0x02 靶标, 0x03 人质}
             min_interval_ms: 心跳周期（无新结果时的推送间隔），建议 30~50ms
CAM → 中控:  CMD_ACK [data_id, max_freq_hz, payload_size=10]
             或 CMD_NACK [data_id, reason]（未知 data_id / 忙）
CAM → 中控:  DATA_STREAM [seq, data_id, sub_payload] 持续推送
中控 → CAM:  CMD_STOP 停止当前订阅（CAM 停止推送）
```

- **切换任务 = 直接发新的 CMD_REQUEST**，无需先 STOP；CAM 切换识别分支后回 ACK。
- 中控侧参考 26-electric 摆杆订阅逻辑：WAIT_ACK → ACTIVE，1s 无数据判掉线重订阅。
- **推送节奏 = 结果驱动（2026-09-28，近端 CAM 侧实现）**：视觉每产出一帧结果就
  立即推一帧，**不受 `min_interval_ms` 限制** —— 打靶误差/坑位"有一帧就实时输出"，
  推流率因此≈相机帧率（上限即 `CMD_ACK` 的 `max_freq_hz`）。`min_interval_ms`
  的语义相应收窄为**心跳周期**：仅在"没有新结果"时按它补帧，用于满足中控 1s
  帧龄 stale 判定与 1s 看门狗（下限 30ms，即心跳最快 ~33Hz）。
  - 中控**不需要**为拿到逐帧结果而下调 `min_interval_ms`；维持 30~50ms 即可。
  - 中控若吃不下这个速率，只能由 CAM 侧改配置
    （`project_config.yaml` → `gks_stream.push_on_new_result: false` 回到纯
    interval 限速），协议层无协商手段。
  - 无新结果 / 结果陈旧 >1s / 模式归属不符 → 零值帧，**心跳不断流**。
- **订阅帧扩展（v1.3）**：USART1 的 CMD_REQUEST 完整 payload 为 5 字节
  `[data_id, min_interval_ms, 爆炸物色, 靶标色, 人质形状]`——三个任务码随
  （重）订阅一并下发（原点长焦扫码后主控已持有任务参数）。只带
  `[data_id(, min_interval_ms)]` 的短帧视为重订阅，CAM 沿用上次任务码。

### 2.2 sub_payload 定义（8B）

**0x01 爆炸物 DATA_EXPLOSIVE**

| 偏移 | 类型 | 字段 | 说明 |
|---|---|---|---|
| 0-1 | u16 LE | cx | 爆炸物像素 x 坐标（640 画面中心 ≈320） |
| 2 | u8 | pit | 位置编号 1~3；0 = 尚未定位 |
| 3 | u8 | flags | bit0: TARGET_FOUND；其余 0 |
| 4-7 | u8×4 | reserved | 恒 0 |

**0x02 靶标 DATA_TARGET（视觉端直接推归一化横向误差）**

| 偏移 | 类型 | 字段全称 | 说明 |
|---|---|---|---|
| 0-1 | i16 LE | horizontal_error_normalized_fixed | 横向误差归一化定点数 = int((pixel_horizontal_coordinate-320)/320*10000)，约 -10000~+10000；y 轴机械侧已调好，程序只闭环 X |
| 2 | u8 | target_found_flag | bit0: TARGET_FOUND；0 = 未发现目标（此时误差字节为 0 占位，不可信） |
| 3-7 | u8×5 | reserved | 恒 0 |

**0x03 人质 DATA_RESCUE**

| 偏移 | 类型 | 字段 | 说明 |
|---|---|---|---|
| 0 | u8 | pos | 人质位置编号 1~3；0 = 未定位 |
| 1 | u8 | flags | bit0: TARGET_FOUND |
| 2-7 | u8×6 | reserved | 恒 0 |

### 2.3 与旧格式对照

旧格式（0x55..0x45，6 字节，无长度无校验）：

| 旧字节 | 含义 | 新去向 |
|---|---|---|
| 0x55 / 0x45 | 首尾标识 | 由 AA 55 + LEN + CRC 取代 |
| [1][2] | cx 高/低字节 | EXPLOSIVE/TARGET cx u16 LE |
| [3] | pos/pit | EXPLOSIVE pit / RESCUE pos |
| [4] | MixCam_request_Stop | 语义并入 CMD_STOP |

中控→CAM 旧格式（任务码 + 摄像头开关）由 CMD_REQUEST(data_id) 取代；
"摄像头开/关"（Open_or_Close）作为 data_id 的订阅/退订天然表达，不再需要独立开关字节。

---

## 3. 链路二：USART2 ↔ 远端 MaixCAM（QR，一次性）

### 3.1 语义

```
中控 → CAM:  CMD_REQUEST [0x10, 0, 0]          ← 请求实时监测结果
CAM → 中控:  CMD_ACK [0x10, 0, 10]             ← 可选，表示已收到请求
CAM → 中控:  DATA_STREAM [seq, 0x10, sub_payload]   ← 有缓存立刻发这一帧；
             无缓存则继续扫，扫出有效 QR 后立刻补发这一帧（不回空帧）
```

- **常扫 + 缓存**：远端上电即每帧扫码，屏幕实时画 bbox，有效三位数字码进缓存
  （最新覆盖），与有无 REQUEST 无关。
- **请求即回**：收到 REQUEST → 回可选 ACK → 主循环第一时间发送缓存内容
  （实时监测结果），本轮结束，但继续扫码刷新缓存。
- **无码等待**：请求到达时无缓存 → 不回 NACK、不回 `valid=0` 空帧，保持扫码，
  扫出有效码立刻补发一帧。
- **重试**：中控对 ACK+DATA 等待超时（建议 3~5s）后可重发 REQUEST；重发是幂等的
  （有缓存即回，无缓存继续等）。

### 3.2 sub_payload 定义（8B）

| 偏移 | 类型 | 字段 | 说明 |
|---|---|---|---|
| 0 | u8 | qr_explosive | 爆炸物颜色：1 红 / 2 绿 / 3 蓝 |
| 1 | u8 | qr_target | 靶标颜色：1 红 / 2 绿 / 3 蓝 |
| 2 | u8 | qr_rescue | 人质形状：1 圆柱 / 2 圆台 / 3 腰鼓 |
| 3 | u8 | valid | 0x01 = 本帧为有效任务码；0x00 = 预留（当前不发） |
| 4-7 | u8×4 | reserved | 恒 0 |

### 3.3 与旧格式对照

| 旧 | 新 |
|---|---|
| CAM→中控 6B `[55, m0, m1, m2, 0x00, 45]`，中控轮询扫描缓冲 | DATA_STREAM 一次性推送，CRC 校验 |
| 中控→CAM 3B `[55, 0x01, 45]`（无意义应答） | 取消；ACK/重发 REQUEST 承担确认职责 |

---

## 4. 链路三：UART4 ↔ panzer 底盘（命令 + 事件）

### 4.1 现状结论：原来中控是如何控制底盘速度与转台的

通读下位机代码（`panzer/GetData/Control_Car.c`、`interurp.c`）后的结论：

- **中控不发送速度值。** UART4 旧下行只有 4 字节
  `[0x55, Car_start_flag, WC_PIT, 0x45]`：`Car_start_flag` → 下位机
  `U4_R_Data.Boss_State`（0x01 放行 / 0x00 停止），它是底盘路线状态机
  `Car_Drive_route()` 的**放行闸门**——在排爆区/打靶区/人质区的等待分支里，
  底盘看到 `Boss_State==0x01` 才把 `Line_speed` 从 0 恢复为内部常量继续走。
- **速度档是底盘内部常量**：`Middle_speed=2.0`、`Low_speed=1.5`（编码器速度单位）、
  到位即 `0`；经 `speed_control()` 加速度限幅 → 双轮速度环 PID（叠加 HWT101 航向环
  差动，`panzer` 的 USART3）→ PWM。中控对"走多快"没有发言权，只决定"何时走/停"。
- **转台在底盘侧**：Emm_V5 步进电机接 `panzer` 自己的 USART6（`Emm_V5.c` 中
  `#define Uart_Emm huart6`），TIM5 周期下发 `Emm_Pos_Control(Motor_Angle)`。
  中控只发 `WC_PIT`（坑位 1/2/3），底盘 `Control_Emm_Angle()` 自行映射
  1→90+33°、2→90°、3→90-33°；路线节点还会自行转 90°（转弯后）和 270°（排爆区转身）。
  注意区分：中控 Brain 自己 USART6 上还挂了一个张大头电机（机械臂俯仰），
  与底盘转台无关。
- **底盘上行**只有 3 字节 `[0x55, 0x01, 0x45]`，永远是 0x01——一个"阶段到位"事件，
  但**不带事件类型**，中控只能靠自身状态机时序去猜是"转弯成功"还是"停稳"。

### 4.2 新协议设计（v1.2：纯 request，段表存底盘）

中控是唯一决策者，底盘是纯执行器：**底盘不感知主状态机**，只认识三条运动
请求 + 完成事件。两端共享的契约只有一张 8 项小表（`NavTarget_t`，§4.3），
主控状态机如何增删重组与底盘固件无关。

**中控 → 底盘：CMD_REQUEST 携带 `[0x21, sub_cmd, arg0, arg1]`（4B payload）**

| sub_cmd | 语义 | arg0 | arg1 |
|---|---|---|---|
| 0x01 REQ_NAV | 执行 nav_id 对应段表（§4.4） | nav_id（`NavTarget_t`，0~5 任务路线 / 6~7 航向重锚伪路线） | gear：0=段表自带常量（默认）1=Low 2=Middle 3=High |
| 0x02 REQ_STOP | 急停安全停车，清执行状态 | 0 | 0 |
| 0x03 REQ_TURNTABLE | 转台定点角：1~3 对坑（1→123° 2→90° 3→57°）；4~6 对桶（4/5→270° 6→90°反向 CW，入桶摆位沿用实车验证值）；7 回正 90°（CCW） | 转台码 0~7（0=不动作） | 保留恒 0 |
| 0x04 GYRO_ZERO | 陀螺仪置零 + 底盘里程计航向基准复位；**须车静止**（上电流程：扫码→置零→起步） | 0 | 0 |
| 0x05 TRACKING_START | 打靶段内跟瞄开启（段表不断，靶标前一段开启，边走边修航向；履带动，转盘不动） | tracking_speed_level：0=段表速度 / 1=低速跟瞄 | 0 |
| 0x06 TRACKING_ERROR | 跟瞄误差流（中控 30~50ms 透传视觉归一化误差，闭环在底盘本地） | error_low_byte，error_high_byte（小端 i16 = horizontal_error_normalized×10000），target_found_flag | —（5B payload） |
| 0x07 TRACKING_STOP | 跟瞄停止 | tracking_stop_mode：0=就地刹停等发射 / 1=发射后继续走完当前段 | 0 |

- REQ_NAV 与 REQ_TURNTABLE 相互独立、任意先后；转台动作不再绑定路线节点，
  由主控按 ALIGN/EXE 需要自行调度。入桶流程：ALIGN_BUCKET 发 坑位+3（对桶码
  4~6），投放完成后发 7 回正——回正与下一 NAV（打靶位）同拍并行，不等 TURN_DONE
- 前向兼容：若新赛题需要任意转台角度，把 arg0 扩展为角度定点数（×2，
  0.5° 分辨率）即可，sub_cmd 不变

**底盘 → 中控：DATA_STREAM `[seq, 0x22, event, echo, reason, rsv×5]`（事件触发单帧）**

| event | 含义 | echo | reason |
|---|---|---|---|
| 0x02 NAV_DONE | 段序列全部完成 | 完成的 nav_id | 0x00 |
| 0x03 NAV_FAULT | 段执行异常 | 异常的 nav_id | 0x01 堵转 0x02 里程超时 0x03 航向发散 0x04 进度看门狗 |
| 0x04 TURN_DONE | 转台到位（可选回报，主控可忽略） | arg0 回显（坑/桶/回正码） | 0x00 |
| 0x05 TRACKING_DONE | 段内跟瞄收敛已刹停（|horizontal_error_normalized|<0.05 维持 500ms），可发射 | 0 | 0x00 |
| 0x06 TRACKING_FAULT | 跟瞄异常（预留未实现：当前丢帧/未收敛为悬停等待，不上报） | 0 | — |

底盘对每条命令回 `CMD_ACK [0x21, sub_cmd_echo, status=0x01]`（3B payload）；
参数/格式不合规回 `CMD_NACK [0x21, sub_cmd_echo, reason]`（3B，原因码见 §1.4），
legacy v1.x 的 2B `[data_id, reason]` NACK 帧 master 仍须接受（按长度判别，仅观测不联动）。

**底盘逐请求拒绝对照表（v2.2）**

| 被拒对象 | sub_cmd_echo | reason |
|---|---|---|
| 帧类型不是 CMD_REQUEST / payload<2B | 0x00 | 0x01 FORMAT |
| CMD_REQUEST 但 data_id ≠ 0x21 | payload[0] 原值 | 0x01 FORMAT |
| 0x01 REQ_NAV | 0x02 | payload<4B → 0x01；nav_id≥8 或 gear>3 → 0x03 PARAM |
| 0x02 REQ_STOP | 0x02 | 无（参数忽略，命中即收） |
| 0x03 REQ_TURNTABLE | 0x03 | payload<3B → 0x01；turn_code>7 → 0x03 |
| 0x04 GYRO_ZERO | 0x00（payload<2B 时在通用检查先行拦截，sub 值不可知）、0x04 正常收发无 NACK | —（其 2B 子帧不足由第一行覆盖） |
| 0x05 TRACKING_START | 0x05 | payload<3B → 0x01；speed_level>1 → 0x03 |
| 0x06 TRACKING_ERROR | — | 高频误差流不回 ACK/NACK；payload<5B 静默丢弃 |
| 0x07 TRACKING_STOP | 0x07 | payload<3B → 0x01；stop_mode>1 → 0x03 |
| 未知 sub_cmd | 原 sub 值回显 | 0x04 UNKNOWNSUB |

### 4.3 nav_id 共享小表（两端唯一契约）

`NavTarget_t`（panzer_Brain 侧定义于 `Engine/mission_sm.h`，底盘侧复制同一张）：

| nav_id | 目的地 |
|---|---|
| 0 | NAV_TO_QR 二维码区 |
| 1 | NAV_TO_BOMB 排爆作业点 |
| 2 | NAV_TO_BUCKET 排爆桶 |
| 3 | NAV_TO_TARGET 打靶位 |
| 4 | NAV_TO_HOSTAGE 人质台 |
| 5 | NAV_TO_RETURN 返回区 |
| 6 | NAV_REBASE_BOMB 重锚伪路线：排爆区原地转至绝对航向 −90.3°（台架标定角） |
| 7 | NAV_REBASE_RETURN 重锚伪路线：携人质返程前原地转至绝对航向 −179.8°（台架标定角） |

与 v1.1 的区别：底盘不再复制/理解 `RobotState_t`，主控状态机全表对底盘不可见。

### 4.4 段表（存底盘，原 if-else 路线完整复刻）

底盘固件内按 nav_id 存 8 张段序列表（6 张任务路线 + 2 张重锚伪路线，
`panzer/GetData/Control_Car.c`，段类型即 Motion 库 `mf_seg_t`），收到 REQ_NAV
后从段 0 顺序执行；
段执行器 = Motion 库 `mf_path`（进段快照 + 里程/航向增量判完成），
`mf_tick()` 挂在 TIM4 10ms 中断里（只消费 `Car_Sense` 采样快照，
见 §7），输出经逆解 + 轮速环 + 执行层直驱 PWM。

段表不是新设计的切分，而是**原 `Car_Drive_route()` flags 路线的逐段复刻**
（原版基线：git `4f3ceb2` true_BOOM_Finish）：直行距离用原常量原值，
速度档沿用原 Middle/Low，加速度沿用原 `speed_control Acc=45`（轮转/s²），
转弯沿用原 `Car_turn_corner` 机制（90° 弧，中心线走 `Diameter×π` 单位
= π 驱动轮转）。原常量以旧 Car_Pos 单位原值存进段表，经宏换算：

| 原参数 | 原值 | 换算（`Control_Car.h`） |
|---|---|---|
| 距离 43/12/46/35/48.5/45.5/175/37.5/60/170 | 旧 Car_Pos 单位，1 单位 = 1/15 驱动轮转 | `MF_M_DIST(u) = u/15 × π×WHEEL_DIA_M` |
| 速度 Middle=2.0 / Low=1.5 | 驱动轮转/s | `MF_M_SPEED(rps) = rps × π×WHEEL_DIA_M` |
| 加速度 45 | 轮转/s² | `MF_M_ACC(45)` |
| 转弯 | 90°，中心线走 π 轮转 | `TURN_RADIUS_M = 2π×WHEEL_DIA_M` |

`WHEEL_DIA_M` 取同车重建工程记录值 0.150m（驱动轮径 150mm，
Engineering-competition-code-2026 `GetData/motion.h`），**台架 5m 直线复标**
后全表距离/速度/半径自动跟随。转台编排（原 flag9 转台转 90°、flag100
对坑/270°舞、flag11 回正）不进段表，由主控 REQ_TURNTABLE 按节点调度
（对坑 1~3 / 对桶 4~6 / 回正 7）；原 flag17 的 5s 兜底转向属主控逻辑，
同样不进段表。旧 flag10/flag17 的两处航向重锚（Tar_Yaw=-90.3/-179.8）
复刻为重锚伪路线 nav_id 6/7（`MF_SEG_TURN_TO` 原地转绝对航向），仍由
主控按节点 REQ_NAV 调度，重锚角为台架标定项。

| nav_id | 段序列（原 flags 路线复刻，数值=原常量原值） |
|---|---|
| NAV_TO_QR(0) | [MOVE 43]（原 flag0 出发首段，台架单段用） |
| NAV_TO_BOMB(1) | [43, ARC L90, 12, ARC R90, 46, ARC L90, 35, ARC R90, 48.5, ARC R90, 45.5(低速)]（原 flag0~10） |
| NAV_TO_BUCKET(2) | [空表，立即 DONE]（桶与排爆作业点在车道两侧同一点位，到位后仅转台转向，无底盘位移） |
| NAV_TO_TARGET(3) | [MOVE 175]（原 flag11~12） |
| NAV_TO_HOSTAGE(4) | [37.5, ARC R90, 60]（原 flag13~16） |
| NAV_TO_RETURN(5) | [MOVE 170]（原 flag17~18） |
| NAV_REBASE_BOMB(6) | [TURN_TO −90.3°]（原 flag10 排爆区重锚，对坑前执行） |
| NAV_REBASE_RETURN(7) | [TURN_TO −179.8°]（原 flag17 返程重锚，携人质返程前执行） |

REQ_NAV 的 gear 覆盖在启动前把段速度补丁进 RAM 副本后启动。

标定原则：**NAV 是粗定位**——段终点只需落在"目标进入视觉视野"的捕获半径内，
残差由 ALIGN 的视觉伺服吃掉（这正是 NAV→ALIGN→EXE 三段式的分工）。
纵向里程无重定位，转弯段航向残差由下一段航向环自愈，横向滑移同样归
ALIGN 兜底。

### 4.5 重发与段内续跑（防事件丢失，硬增量位移的核心规则）

| 场景 | 底盘行为 |
|---|---|
| REQ_NAV(nav_id) 重发且该序列**进行中** | **段内续跑**：保留当前段里程锚点与段索引，不重跑（增量位移对重发天然幂等，前提是不清锚点） |
| REQ_NAV(nav_id) 重发且该序列**已完成** | 不产生运动，只补发 NAV_DONE |
| REQ_NAV(新 nav_id) | 停当前段，新段表从段 0 开始（唯一允许"重来"的情况） |
| REQ_STOP | 任何时刻立即安全停车并清执行状态；之后的 REQ_NAV 从段 0 开始 |

进度看门狗：段执行中连续 3s（建议值）`Car_Pos` 无变化 → 上报
NAV_FAULT(reason=0x04)。增量位移对"轮子打滑车没动"是盲的，用里程变化率兜底。

主控侧 NAV 超时（`cfg.nav_ms`）未收到 NAV_DONE → 重发 REQ_NAV（按上表幂等）
→ 重试耗尽 → `ST_FAULT` → Hook 发 REQ_STOP。

### 4.6 与旧格式对照

| 旧 | 新 |
|---|---|
| 下行 `[55, start, pit, 45]`，启停+坑位，速度全是底盘内部常量 | 三条请求 REQ_NAV/REQ_STOP/REQ_TURNTABLE，路线切分权上移主控，段表仍在底盘 |
| 上行 `[55, 0x01, 45]`（无类型，靠中控时序猜） | 事件帧带 event 码 + nav_id 回显 + 故障原因，完成主动通知 |
| 主控不感知底盘进度，事件丢失即错拍 | 重发幂等 + 段内续跑 + 进度看门狗，链路丢包不破坏运动学 |

---

## 5. 时序示例

### 5.1 USART1 订阅靶标

```
中控(Brain)                    近端CAM(USART1)
    │── CMD_REQUEST(0x02,50,0) ─→│
    │←─ CMD_ACK(0x02,60,10) ─────│
    │←─ DATA_STREAM(seq,0x02,cx,flags) ─┐ 持续推送
    │←─ DATA_STREAM(...) ───────────────┘
    │── CMD_STOP ────────────────→│  任务完成，停止推送
```

### 5.2 USART2 请求 QR（一次性）

```
中控(Brain)                    远端CAM(USART2, 时刻扫码画框+缓存)
    │── CMD_REQUEST(0x10,0,0) ──→│
    │←─ CMD_ACK(0x10,0,10) ──────│  (可选)
    │←─ DATA_STREAM(seq,0x10,qr) │  有缓存第一时间发; 无缓存则扫出后立刻补发
    │   (超时未收到 → 重发 REQUEST, 幂等) │
```

### 5.3 UART4 纯 request 运动控制

```
中控(Brain)                        底盘(panzer, UART4)
    │── REQ_NAV(BOMB,gear=0) ──────→│
    │←─ ACK[0x21,01,01] ────────────│  (可选,联调期开)
    │     (底盘按段表增量位移)         │
    │←─ DATA[seq,NAV_DONE,BOMB] ────│  段序列完成,主动回报
    │   (超时未收到 → 重发REQ_NAV,    │
    │    段内续跑,幂等)               │
    │── REQ_NAV(REBASE_BOMB) ──────→│  排爆区航向重锚(原地转−90.3°,复用NAV机制)
    │←─ DATA[seq,NAV_DONE,REBASE_BOMB]│
    │── REQ_TURNTABLE(pit=2) ──────→│  转台对坑(与NAV独立调度)
    │── REQ_TURNTABLE(pit+3=5) ────→│  入桶: 转台对桶(摆位沿旧实测值)
    │── REQ_TURNTABLE(7=回正) ──────→│  投放后回正,与 REQ_NAV(TARGET) 同拍并行
    │── REQ_NAV(REBASE_RETURN) ────→│  携人质返程前重锚(原地转−179.8°)
    │←─ DATA[seq,NAV_DONE,REBASE_RETURN]│
    │── REQ_STOP ──────────────────→│  任何时刻急停
```

---

## 6. 测试向量

CRC 按 §1.3 计算。可用串口助手直接发送验证（含 v2.2 底盘 3B NACK 与相机 legacy 2B NACK 向量）。

```
USART1 中控→CAM 订阅爆炸物(50ms):  AA 55 06 20 01 32 00 E7 49
USART1 中控→CAM 订阅靶标(50ms):    AA 55 06 20 02 32 00 BE 19
USART1 中控→CAM 订阅人质(50ms):    AA 55 06 20 03 32 00 89 29
USART1 CAM→中控 ACK(爆炸物):       AA 55 06 21 01 3C 0A 13 B8
USART1 CAM→中控 爆炸物数据帧:       AA 55 0D 24 2A 01 D2 04 02 01 00 00 00 00 30 49
USART1 中控→CAM 停止订阅:          AA 55 03 23 F5 F1

USART2 中控→CAM 请求QR:            AA 55 06 20 10 00 00 F0 ED
USART2 CAM→中控 ACK:               AA 55 06 21 10 00 0A 27 13
USART2 CAM→中控 QR数据帧:           AA 55 0D 24 01 10 01 02 01 01 00 00 00 00 DB 48

UART4 中控→底盘 REQ_NAV(二维码区):   AA 55 07 20 21 01 00 00 6F 72
UART4 中控→底盘 REQ_NAV(排爆点):     AA 55 07 20 21 01 01 00 5C 43
UART4 中控→底盘 REQ_NAV(返回区,gear=2):
                                    AA 55 07 20 21 01 05 02 B0 C5
UART4 中控→底盘 REQ_STOP 急停:       AA 55 07 20 21 02 00 00 36 22
UART4 中控→底盘 REQ_TURNTABLE(pit_position=2):AA 55 07 20 21 03 02 00 67 70
UART4 中控→底盘 REQ_TURNTABLE(对桶坑2=码5):  AA 55 07 20 21 03 05 00 FE E7
UART4 中控→底盘 REQ_TURNTABLE(对桶坑3=码6):  AA 55 07 20 21 03 06 00 AB B4
UART4 中控→底盘 REQ_TURNTABLE(回正=码7):     AA 55 07 20 21 03 07 00 98 85
UART4 中控→底盘 REQ_NAV(重锚排爆区=6):       AA 55 07 20 21 01 06 00 C5 D4
UART4 中控→底盘 REQ_NAV(重锚返程前=7):       AA 55 07 20 21 01 07 00 F6 E5
UART4 中控→底盘 TRACKING_START(低速跟瞄):AA 55 07 20 21 05 01 00 80 83
UART4 中控→底盘 TRACKING_ERROR(e=+0.25,found):AA 55 08 20 21 06 C4 09 01 EE EA
UART4 中控→底盘 TRACKING_ERROR(e=-0.25,found):AA 55 08 20 21 06 3C F6 01 A7 E6
UART4 中控→底盘 TRACKING_STOP(就地刹停):AA 55 07 20 21 07 00 00 DD D2
UART4 中控→底盘 TRACKING_STOP(继续走完当前段):AA 55 07 20 21 07 01 00 EE E3
UART4 底盘→中控 ACK:                AA 55 06 21 21 01 01 57 DC
UART4 底盘→中控 NACK(REQ_NAV 参数越界, sub_echo=0x02, PARAM):
                                    AA 55 06 22 21 02 03 B9 11
UART4 底盘→中控 NACK(帧类型/data_id 不支持, sub_echo=0, FORMAT):
                                    AA 55 06 22 21 00 01 FF 31
UART4 底盘→中控 NACK(未知子命令0x09, sub_echo=0x09, UNKNOWNSUB):
                                    AA 55 06 22 21 09 04 15 0C
USART1 相机→中控 legacy 2B NACK(未知 data_id=0x01): AA 55 05 22 01 01 07 2A
UART4 底盘→中控 NAV_DONE(排爆点):    AA 55 0D 24 01 22 02 01 00 00 00 00 00 00 16 2B
UART4 底盘→中控 NAV_FAULT(排爆点,堵转):
                                    AA 55 0D 24 02 22 03 01 01 00 00 00 00 00 A5 97
UART4 底盘→中控 TURN_DONE(pit_position=2):AA 55 0D 24 03 22 04 02 00 00 00 00 00 00 90 E8
UART4 底盘→中控 TURN_DONE(对桶码5):  AA 55 0D 24 03 22 04 05 00 00 00 00 00 00 89 AC
UART4 底盘→中控 TRACKING_DONE:AA 55 0D 24 04 22 05 00 00 00 00 00 00 00 AB 22
UART4 底盘→中控 TRACKING_FAULT(预留, 未实现):AA 55 0D 24 05 22 06 00 01 00 00 00 00 00 49 B2
```

样例数据帧解读（USART1 爆炸物）：`seq=0x2A`、`cx=0x04D2=1234`、`pit=2`、
`flags=0x01(已找到)`。

---

## 7. 三端改动点清单

| 端 | 改动 |
|---|---|
| panzer_Brain | `Middlewares/serial_protocol` 已入工程；三个串口各建一个 `Protocol_Parser_t` + 链路 cfg；按 §2/§3/§4 注册处理函数；订阅重试参考 26-electric `Protocol_Task` 模式；**v2.2：三链路 NACK 解析（§1.4）+ sent 标志作废联动 + USART3 调试回显 NACK 事件** |
| 近端 MaixCAM | `main.py` 收发改为帧编解码（参考 `protocol.md` 的 Python CRC 实现）；三个任务分支按 data_id 切换，推送循环受订阅状态控制；未知 data_id 回 NACK [did,0x01]（已迁 Zulu-Walker，见 v2.2） |
| 远端 MaixCAM | `main.py` 时刻扫码画 bbox 并缓存有效码，收到 REQUEST 第一时间发缓存（无码则扫到再补发一帧） |
| panzer 底盘 | `interurp.c` 换成 protocol 框架帧解析（sub_cmd/args，`protocol.c/h` 已入 `GetData`）；删 `Car_Drive_route()`/`Sport_Car()`/`Car_turn_corner()` 旧路线，复刻为 6 张任务 `mf_seg_t` 段表 + 2 张重锚伪路线（`Control_Car.c` 按 nav_id 索引，§4.4），执行交给 Motion 库 `mf_tick()`（TIM4 10ms：纯消费 `Car_Sense.c` 采样快照→里程计+段执行+逆解+轮速环，采集/存储/执行在 sense 层）；完成回 NAV_DONE，重发走段内续跑（§4.5）；转台改 REQ_TURNTABLE 驱动（对坑/对桶/回正码）；GYRO_ZERO 支持上电全自动（扫码→置零→起步） |

### 7.1 调试口约定（v2.2，两块板各一条，均 ISR 零发送）

| 端 | 口 | 事件行（主循环回发，事件驱动） | 查询输入 | 查询回行 |
|---|---|---|---|---|
| panzer_Brain | USART3, 115200 | `NACK:[C/Q/R]<sub>/<reason> #<n>`（C=底盘链/Q=USART2 扫码/R=USART1 近相机） | `'?'` | `BRAIN:<状态名> ACK:<n> NACK:<n> <sub>/<reason>` |
| panzer | USART2, 115200 | `ACK:<sub>/<status> #<n>`、`NACK:<sub>/<reason> #<n>`（自报无链路前缀，与 Brain `[C]` 行同构对照） | `'?'`（USART2 RX DMA1_S5/CH4，IDLE+0x8000 突发扫描） | `CHASSIS:NAV=<I/R/D> ID=<nav_id> SEG=<seg>/<done> TRK=<I/R/L> E=<误差x10000>` + `POSE:<x_mm> <y_mm> <yaw_x10deg>` |

联调试法：同一参数错误在 Brain 侧打 `NACK:[C]02/03`、底盘侧打 `NACK:02/03`，两边 `#n` 应同步递增；底盘 ORE/错误重装沿用 UART4 ErrorCallback 同款。USART2 RX DMA 用 HAL 旧式 Stop+重装骨架（同 UART4 实车验证路径），IDLE→重装窗口丢字节仅影响 '?' 类人工单字节输入，可忽略。

---

## 8. 版本历史

| 版本 | 日期 | 变更 |
|---|---|---|
| v2.2 | 2026-09-14 | NACK 全链路语义落地（§1.4 新章节）：原因码统一枚举 0x01 FORMAT/0x02 BUSY/0x03 PARAM/0x04 UNKNOWNSUB（相机 0x01/0x02 语义与格式不变）；底盘 NACK 升级 3B `[0x21, sub_cmd_echo, reason]`（legacy 2B 仍接受，master 按 payload 长度判别），逐请求拒绝对照表 + 样例帧；Brain 新增三链路 NACK 解析与 volatile 计数 / fresh 事件标志，按 sub_cmd_echo 立即作废对应 sent 标志触发下拍重发，USART3 调试口主循环异步回显 NACK 事件与计数；底盘新增 USART2 调试口（RX DMA1_S5/CH4）ACK/NACK 事件与 '?' 状态查询（§7.1）；Zulu-Walker SDK NACK_REASON_NAMES 补 0x03/0x04 |
| v2.1 | 2026-09-12 | 入桶方向补全 + 航向重锚恢复：REQ_TURNTABLE arg0 扩展 4~6 对桶（270°/270°/90°反向CW，摆位沿用旧实测值）与 7 回正90°，TURN_DONE echo 泛化为 arg0 回显，投放后回正与下一 NAV 同拍并行；nav_id 新增 6/7 重锚伪路线（MF_SEG_TURN_TO 原地转绝对角 −90.3°/−179.8°，复用 NAV 幂等/看门狗/重发全套机制），主控在排爆区对坑前与携人质返程前各调度一次 |
| v2.0 | 2026-09-11 | 打靶段内跟瞄（段表不断）：视觉 0x02 直推归一化误差 e=(cx-320)/320；新增 TRACKING_START/ERROR/STOP + TRACKING_DONE/FAULT（全称语义变量）；底盘段内叠加视觉航向修正、收敛段内刹停、发射后继续走完当前段；REQ_TURNTABLE 恢复固定角（删 err）；激光 PA4 拉低发射 |
| v1.5 | 2026-09-10 | 远端改常扫+缓存+请求即回（时刻扫码画 bbox，无码等扫到再回，不回空帧）；近端修任务码切换清 latch、无阈值空帧心跳、阈值缺失保护；帧格式/Brain 零改动 |
| v1.4 | 2026-09-09 | 段表改为原 flags 路线完整复刻（基线 git true_BOOM_Finish：距离/速度/加速度/转弯机制全用原值，单位=旧 Car_Pos，R=2π×轮径，WHEEL_DIA_M 采 150mm 记录值）；Motion 库去硬件化：mf_tick 改纯消费 `Car_Sense` 采样快照（seq 去重），采集/存储/执行移入 sense 层，删 mf_hal；新增 sub_cmd 0x04 GYRO_ZERO（置零+里程计航向基准复位）；上电全自动流程：IDLE 静置→原点扫码（周期重发直到读出）→陀螺仪置零静置→NAV 起步，删调试口 'g' 手动触发；§4.2/§4.4/§7 随实现刷新 |
| v1.3 | 2026-09-09 | 底盘侧落地（线上协议不变）：段表实现为 Motion 库 mf_seg_t（Control_Car.c 按 nav_id 索引），mf_tick 融合进 TIM4 10ms 循环；UART4 换 protocol 框架解析；Brain 三链路收发回调 + mission_sm 接通 NAV/QR/急停；任务流改为原点长焦扫码（EXE_QR 提前，NAV_TO_BOMB 前插 43cm 首段直达排爆）；USART1 订阅帧扩展携带任务码（5B payload）；近/远端 MaixCAM 改帧编解码；§2.1/§4.4/§7 随实现刷新 |
| v1.2 | 2026-09-08 | UART4 改纯 request：删状态镜像与 RobotState_t 共享表，改为 REQ_NAV/REQ_STOP/REQ_TURNTABLE 三原语 + 6 项 NavTarget_t 契约；段内续跑规则修正（§4.5）并新增进度看门狗 |
| v1.1 | 2026-09-08 | UART4 升级为主状态机镜像同步：CMD_CAR_CTRL 携带 state_id（§4.3 共享枚举表），NAV 状态触发段表增量位移（§4.4），NAV_DONE/NAV_FAULT 主动回报与幂等重发（§4.5） |
| v1.0 | 2026-09-08 | 初版：三条链路应用语义、data_id 全局表、payload 布局、测试向量 |
