# 排爆阶段问题简报：任务码 321（目标蓝球）→ 转台转 pit3 抓了红色

> 提交给 **电控侧（panzer_Brain）** 调查用。
> 视觉侧（Zulu-Walker / MaixCAM-Pro）已完成代码走查与 PC 端复现，
> 本文件给出**现象、协议事实、复现证据、待电控侧确认的点**。
>
> 日期：2026-10-09　分支：`EOD2026-NearCam` / `feat/maixcam1-port`（当前工作分支）
>
> **📌 状态：已闭环（2026-10-09）**
> Brain 侧 5 项已逐条核对并回报，**无自身缺陷**；根因确认 = CAM 侧
> `infer_missing: tail` 补位。回执见
> `docs/competition/issue_explosive_pit_mismatch_brain_reply.md`。
> 该回执另提出一个**锁存-修正**新问题（Brain 锁存后停止重读相机），
> 见回执 §3。

---

## 1. 现象（现场观测）

- **任务码输入 `321`**，按协议语义 = `(爆色=3蓝, 靶色=2绿, 形状=1圆柱)`
  ⇒ 排爆阶段目标应为**蓝色球**。
- 实物坑位：**蓝球在 pit1**，红球在 pit3（绿在 pit2）。
- 相机扫过顺序：**正常**（先看 pit1 → 再看 pit2，pit3 由推导得出）。
- **实际动作**：转台**自动转到 pit3**，**抓了红色**；蓝球（pit1）**没有抓**。

即：视觉侧上报的 `pit = 3`，但实物 pit3 是红球，导致抓错。

---

## 2. 协议事实（v2.1，两平台逐字节一致）

来源：`docs/competition/gks_panzer_links_protocol_v2.1.md`

### 2.1 订阅帧（Brain → CAM，USART1）

```
CMD_REQUEST (0x20) 完整 payload 5B = [data_id, min_interval_ms, 爆炸物色, 靶标色, 人质形状]
```

- `data_id = 0x01` = EXPLOSIVE（排爆）
- 任务码三位：**digit1 = 爆炸物色，digit2 = 靶标色，digit3 = 人质形状**（1红/2绿/3蓝；形状 1圆柱/2圆台/3腰鼓）
- 短帧（1~4B）= 重订阅，沿用上次任务码

**关键待确认**：`321` 是按 `[爆,靶,形] = [3,2,1]` 填入 payload 的
（即 payload[2]=3, payload[3]=2, payload[4]=1），**请电控侧确认发送侧的顺序**。
若 Brain 侧按 `[形,靶,爆]` 或其它顺序填，则 CAM 收到的爆色会变成 1（红）而不是 3（蓝）
—— 这是首先要排除的一种可能。

### 2.2 结果帧（CAM → Brain，USART1）

```
DATA_STREAM (0x24)  payload 10B = [seq, data_id, sub_payload 8B]
```

**0x01 爆炸物 sub_payload：**

| 偏移 | 类型 | 字段 | 说明 |
|---|---|---|---|
| 0-1 | u16 LE | cx | 爆炸物像素 x（640 画面中心≈320） |
| 2 | u8 | **pit** | 位置编号 1~3；**0 = 尚未定位** |
| 3 | u8 | flags | bit0: TARGET_FOUND |
| 4-7 | u8×4 | reserved | 恒 0 |

- 有效性判据（协议 §5）：`found=1 且 pit∈[1,3]` 才被采纳。
- **待确认点**：Brain 是否在 `found=0 / pit=0` 时**正确回退**（不动作），
  而不是把上一次的 `pit` 或某个默认值拿去转台。见 §4 第 2 条。

---

## 3. 视觉侧代码走查结论（CAM 端）

### 3.1 红蓝通道序：**无问题**

已逐环节核对，红蓝不会互换：

| 环节 | 处理 | 结论 |
|---|---|---|
| 相机取帧 | `image2cv(ensure_bgr=False)` → `[::-1]` | `last_raw`=RGB888，numpy=BGR ✅ |
| maix C++ 快路径 | `_as_rgb888()` 检测 BGR 标记并转 RGB | ✅ |
| 降采样小图 | `cvtColor(BGR2RGB) + bgr=False` | ✅（2026-09-26 事故已修，有回归测试） |
| cv2 回退 | `COLOR_BGR2LAB`，输入 BGR | ✅ |
| 色号映射 | `COLOR_KEYS={1:red,2:green,3:blue}` | ✅ |

**排爆三色阈值在当前设备值下两两互斥**（红盒 B>9，蓝盒 B<-25，中间真空带），
物理上红蓝不可能同时命同一个像素。

> 注意：HUD 上的 `C1/C2/C3` 是**色号**（C1=红/C2=绿/C3=蓝），**不是坑位号**。
> 蓝球框显示 `C3` 是正常的，勿误读为"蓝在 pit3"。

### 3.2 任务码解析：**无问题**

`parse_cmd_request_payload`：`codes = (payload[2], payload[3], payload[4])` = `(爆色,靶色,形状)`。
排爆用 `_codes[0]`（爆色），`321` → `_codes[0]=3`（蓝）✅。

### 3.3 坑位定序：**这是根因所在**

排爆定序规则（`eod_config.yaml` → `gks.explosive`）：

```yaml
order_by: first_seen     # 坑位顺序 = 首次检出先后（谁先被看到谁就是 pit1）
infer_missing: tail      # 三色互不相同 → 见到 2 色即锁，未见者补到顺序「尾部」
```

**逻辑铁前提**：*第一个被看到的颜色，物理上就在 pit1*。

---

## 4. PC 端复现证据（已用 CV 复现）

用与设备一致的阈值，合成"实物 pit1=蓝、pit2=绿、pit3=红"，任务码爆色=3（蓝）：

### 复现 A：蓝球在 pit1 首帧漏检 → 补位塞到尾部

```
帧1 只见 绿(pit2)              -> seen=[2]
帧2 见   绿+红                 -> seen=[2,1]   （已够2色 → 补位）
帧3 见   绿+红（蓝仍漏检）     -> seen=[2,1] → ordered=[2,1,3]

结果:  _position = [2,1,3]  即 pit1=绿 pit2=红 pit3=蓝
       pit = 3   inferred = True
       ↑ 实物 pit1=蓝 pit2=绿 pit3=红  → 完全错位
       → Brain 转 pit3 → 抓到红球  ← 与现场现象完全一致
```

### 复现 B：一旦锁存不再纠正

```
锁存后再让三色齐现, _pos_flag 已 =1 → 不再重锁
错误映射被永久固化, 直到 set_task 换码 / STOP
```

### 3.3 结论

**不是红蓝搞反，是「补位（infer_missing: tail）在目标色恰为 pit1 且首帧漏检时必然算错」。**

`tail` 补位只推出「未见者**是什么**」，并**假设未见者 = 最后一个坑**。
当未见者其实是**第一个坑**（因为 pit1 漏检）时，就静默把目标判到 pit3。

---

## 5. 请电控侧（Brain 侧）确认的点

请在 Brain 侧代码内逐条核对并回报：

1. **发送侧任务码顺序**：`321` 填进 CMD_REQUEST payload 时，
   是否严格为 `[data_id, min_interval_ms, 爆=3, 靶=2, 形=1]`？
   （即 payload[2..4] = 3,2,1）。若顺序不同 ⇒ CAM 收到的爆色不是蓝。

2. **`found=0 / pit=0` 时的行为**：CAM 在未锁存时推 `found=0, pit=0`。
   Brain 是否**严格忽略**这种帧（不转台、沿用上一有效值）？
   还是会把 `pit=0` 当合法值 / 保留上一次 `pit` 去动作？

3. **`pit` 的消费偏移**：消费 DATA_STREAM 时，
   `pit` 是否取在 sub_payload 偏移 **2**（`[seq(1) + data_id(1) + cx(2)]` 之后）？
   是否正确按 u16 LE 读 cx 而没把 cx 高字节当 pit？

4. **转台 pit↔角度映射**：pit1/2/3 到转台角度的标定方向是否与实物一致？
   （若 Brain 侧 pit 映射本身镜像，也会表现为"转到反向的坑"）

5. **日志**：Brain 侧能否提供本次任务下**收到的 `pit` 序列**（或 ACK/DATA 原始帧）？
   与 CAM 侧日志对照（见 §6）可直接定位是 CAM 推错还是 Brain 读错。

---

## 6. CAM 侧需要电控侧配合抓的日志

下次复现时，CAM 侧控制台会打出（`log_print`，WARNING 级可见）：

```
[GKS] seen += 2 -> [2]
[GKS] seen += 1 -> [2, 1]
[GKS] latch pit=3 seen=[2,1] ordered=[2,1,3] codes=[3,2,1] inferred=True
[UART TX] DATA_STREAM ... pit=3 ...
```

**判读**：若 `seen` 里**没有 3（蓝）** 而 `inferred=True`，
即 100% 确认是「蓝球漏检 → 补位塞尾部」；
若 `seen` 里有 3 但 `ordered` 仍错，则是定序方向问题（另一类）。

---

## 7. 视觉侧可选修法（供参考，待现场确认后择一）

| 方案 | 改动 | 代价 |
|---|---|---|
| **A（推荐）** 关补位 | `gks.explosive.infer_missing: none` | 三色齐见才锁；锁不上会超时 FAULT（**可见可诊断**，不会静默抓错） |
| B 治漏检 | 降 `pixels_threshold`（现 4000）或放宽蓝盒 `L/A` 上界 | 需现场标定 |
| C 改定序 | `order_by: x`（按画面左右，不依赖先见） | 仅当三坑同屏可靠，跨帧行为待核 |

---

## 附：关键代码位置索引（供对照）

| 内容 | 路径 |
|---|---|
| 排爆处理 + 定序/补位/锁存 | `modules/zw_opencv_module/processors/gks_task_processor.py`
  `_process_explosive` (行 743+)、`_ordered_pits` (行 649)、`_latch_debounce` (行 663) |
| 任务码解析 | `modules/zw_uart_module/protocol.py` `parse_cmd_request_payload` (行 204) |
| 订阅门控 / 锁存清理 | `app/coordinator.py` `_on_cmd_request` (行 ~340)、`_latch_reset_needed` (行 399) |
| 阈值文件 | `/my-threshold/my_64threshold.txt`（设备）/ 仓库根同名文件 |
| 协议文档 | `docs/competition/gks_panzer_links_protocol_v2.1.md` |
| 字段核对记录 | `docs/competition/gks_protocol_crosscheck.md` |
