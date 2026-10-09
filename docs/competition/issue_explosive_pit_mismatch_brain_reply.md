# 排爆 pit 错位问题 — Brain 侧核对回报（闭环）

> 本文是 `issue_explosive_pit_mismatch.md`（视觉侧简报）的 **Brain 侧回执 + 闭环结论**。
> 由 **panzer_Brain** 侧逐条核对后回报，CAM 侧记录归档。
>
> 日期：2026-10-09
> 关联文件：
> - 现象简报（视觉侧）：`docs/competition/issue_explosive_pit_mismatch.md`
> - 转交简报（精简版）：`docs/competition/issue_explosive_pit_mismatch_brief.md`
> - 协议：`docs/competition/gks_panzer_links_protocol_v2.1.md`

---

## 0. 结论摘要（一句话）

**Brain 侧 5 项待确认点全部核对通过，未发现可导致「转 pit3 抓红色」的缺陷。**
全链路为 `CAM 定序算 pit → 原样透传 → 原样下发转台码`，Brain **不做任何颜色↔坑位重映射**。
**根因确认在 CAM 侧**：`infer_missing: tail` 补位在「目标色（蓝）位于 pit1 且首帧漏检」时，
把未见者补到顺序尾部 → 上报 `pit=3` → Brain 忠实执行 → 转到实体 pit3（红球）。

---

## 1. 逐条核对结果

### 1.1 发送侧任务码顺序 —— ✅ 顺序正确，可排除

`Interrupt/interrupt.c:144-164`：

```c
void Cam1_Subscribe(uint8_t data_id, uint8_t interval_ms,
                    uint8_t bomb, uint8_t target, uint8_t shape)
{
    uint8_t request_payload[5] = {data_id, interval_ms, bomb, target, shape};
```

- payload 顺序 = `[data_id, interval, 爆, 靶, 形]`，严格按协议。
- 调用点（`mission_sm.c:1199`）实参顺序亦为 `(bomb_color, target_color, hostage_shape)`，一致。
- 参数来源链：
  QR 相机主动推 `send_data_stream(0x10, [d0,d1,d2,0x01])`
  （`maixcam/工科赛下端/main.py:157`，三位 ASCII 码**按输入顺序拆位，不做语义重排**）
  → Brain `Hook_QrDecode` 原样落 `params->bomb_color=qr_payload[0]`（=digit1）、
  `target_color=qr_payload[1]`、`hostage_shape=qr_payload[2]`（`mission_sm.c:76-79`）。

**⇒ `321` → 爆色=3=蓝，正确。顺序问题排除。**

### 1.2 `found=0 / pit=0` 的处理 —— ✅ 严格忽略，无回退漏洞

`Camera_GetPitPosition`（`interrupt.c:196-234`）：

- 帧超时 >1000ms → 返回 0（stale）。
- **`found_local==0 || pit_position_local==0 || pit>3` → 返回 0**（221 行，三重判据）。
- 返回 0 时**不写** `*pit_position`。

消费侧 `Align_LockPit`（`mission_sm.c:84-105`）：
仅在返回非 0 **且** `target_found_flag != 0` **且** `pit ∈ [1,3]` 时才写
`msm.align_target_pit`；返回 0 时直接 `return 0` 等下一拍。

**⇒ 不存在「把 0 当合法值」或「沿用旧 pit」的路径。**
（注：`align_target_pit` 一旦非 0 即锁死不再重读相机 —— 见 §3 补充提醒。）

### 1.3 `pit` 消费偏移 —— ✅ 正确，未与 cx 混字节

`Interrupt/interrupt.c:211-214`（排爆 `0x01`）：

```c
pit_position_local = raw_data[2];                    // sub_payload 偏移 2
found_local = (uint8_t)(raw_data[3] & 0x01u);        // 偏移 3, bit0
```

`raw_data` 在 `on_cam1`（138 行）由 `memcpy(&payload[2], SUB_PAYLOAD_SIZE)` 拷入 ——
即**已跳过 seq(1)+data_id(1)**，故 `raw_data[2]` 正好落在「cx u16LE 之上」。

**⇒ 偏移全对，cx 高字节没有被当作 pit。**

### 1.4 `pit` ↔ 转台角度映射 —— ✅ 非镜像，映射正确

Brain 侧 `Align_TurnSend(msm.align_target_pit)` / `Align_TurnWait(pit)`
（`mission_sm.c:143, 1239`）—— **pit 数值直接当转台码原样下发**。

底盘侧 `nav_turntable.c` 码表：

| 转台码 | 物理角 |
|---|---|
| 1 | 123° |
| 2 | 90° |
| 3 | 57° |

映射 1:1、非镜像，方向一致（1→123° 最左，2→90° 中，3→57° 最右）。

**⇒ Brain 侧无镜像 bug。收到 pit=3 → 底盘忠实转到 57°（pit3 物理位）。链条本身无错。**

### 1.5 原始 `pit` 序列日志 —— 决定性对照证据

Brain 侧调试口（USART3，`?` 查询）：

| 读数 | 位置 | 含义 |
|---|---|---|
| `SM:... PIT=<n> ...` | `interrupt.c:1095` | `msm.align_target_pit` = 锁存后**实际拿去转台的坑号** |
| `CAM1:01=<ms> 02=.. 03=..` | `interrupt.c:1150` | 排爆流新鲜度 |
| `TT:x<code>#<cnt>` | `interrupt.c:1048` | 最近下发的转台码 |

**判读法**：若现场 `SM:PIT=3` 且 `TT:x03`，
则 Brain 只是老实执行了 CAM 推的 `pit=3` —— **锅在 CAM 的定序/补位**。

---

## 2. 综合判断（闭环）

**Brain 侧 5 项全部通过，无自身缺陷。**
全链路为 `CAM 定序算 pit → 原样透传 → 原样下发转台码`，Brain 不做颜色→坑位重映射。

⇒ 简报 §3.3 的根因成立并被确认为唯一根因：

> **CAM 侧 `infer_missing: tail` 补位**
> 在「目标色（蓝）位于 pit1 且首帧漏检」时，`seen=[2,1]`（绿、红）→
> 补位成 `ordered=[2,1,3]`（蓝被塞尾部）→ 上报 `pit=3`。
> Brain 忠实执行 → 转到实体 pit3（红球）→ 抓错。

---

## 3. ⚠ 补充提醒（简报未覆盖，CAM 侧须一并排查）

结合 `Align_LockPit` 的**锁存**语义：

- Brain 在 `align_target_pit` **首次非 0 后即停止重读相机**
  （直到 `SM_AlignBomb` 重试重入且 `retry_cnt==0`，或 STOP/换码）。
- 这意味着：**CAM 若在扫视早期就把 pit 锁成 3，即使后续修正成 1，Brain 也收不到修正。**

这与简报 §4「复现 B：锁存后不再纠正」是**同一性质**，但责任侧在两边叠加：

| 侧 | 行为 | 后果 |
|---|---|---|
| CAM | 错误锁存后，若不再产出修正值 | 错误永久固化 |
| Brain | 已锁存 pit 且 `tt.busy` 占用时 | 修正帧被**静默丢弃** |

**⇒ CAM 侧须确认：锁定错误后是否还有「重推修正帧」的机制。**
若 CAM 会重推 `pit=1`，但 Brain 已锁存 `pit=3`（且转台忙），
则修正帧会被 Brain 静默丢弃 —— 需要明确的**重锁协议**（如 STOP→重订阅清锁存），
否则「CAM 修了也白修」。

---

## 4. CAM 侧后续待办

1. **确认根因**：抓 `[GKS] latch pit=… seen=… ordered=… inferred=True` 日志，
   核对 `seen` 中是否缺目标色（蓝=3）。若缺且 `inferred=True` ⇒ 根因坐实。
2. **修法定夺**（见简报 §7）：
   - **A（推荐）** `gks.explosive.infer_missing: none` —— 三色齐见才锁，
     锁不上超时 FAULT（**可见可诊断**），不静默抓错。
   - B 治漏检：降 `pixels_threshold` 或放宽蓝盒 `L/A` 上界（治本）。
   - C 改定序：`order_by: x`（仅三坑同屏可靠）。
3. **排查 §3 的锁存-修正问题**：确认锁定错误后 CAM 有无重推机制、
   以及是否需与 Brain 约定「STOP→重订阅」来清锁存。

---

## 5. Brain 侧回执原文（存档）

> Brain 侧 5 点全部核对完毕：①任务码 payload[2..4]=[爆,靶,形] 顺序正确，321 解析为爆色=3蓝；
> ②`Camera_GetPitPosition` 对 found=0/pit=0/pit>3 一律返回 0 并严格忽略，无回退漏洞；
> ③pit 取 `sub_payload[2]`（已跳过 seq+data_id），cx 按 u16LE 读，偏移无误；
> ④pit 直接当转台码原样下发，1→123°/2→90°/3→57° 非镜像，映射正确；
> ⑤调试口 `?` 的 `SM:PIT=` 与 `TT:x<code>` 行即为 Brain 收到的 pit 与发出的转台码。
> **结论：Brain 无自身缺陷，是忠实执行被算错的 pit=3，根因确认为 CAM 端 `infer_missing:tail` 补位**。
> 另请 CAM 侧确认：锁存到错误 pit 后是否还有重推修正机制 —— Brain 在 `align_target_pit` 非 0 后
> 停止重读相机（锁存语义），若 CAM 后续修正了 pit 也须确认 Brain 是否还会接受。
