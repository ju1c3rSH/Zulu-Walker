# 人质坑位身份判定：单帧锁存缺陷与修复方案

> 2026-11-04 定稿。适用分支 `feat/explosive-dock-trim`（platform: maixcam1）。
> 触发事件：2026-11-04 03:37 现场日志，圆台被锁成圆柱，`pos` 报错一坑。
> 状态（2026-11-06 更新）：
> - **已实施最小版**：`shape_hits` 连续帧门槛（§0.1，commit 见 git log）。不动测试、不动协议。
> - **下文 §4 的 v3（run + 跨类去重 + 入画锚点）仍待实施**，保留作为下一步方案。
> - 已提交 `ebbe1f8`（conf_th 0.55→0.65）保持不动。

---

## 0. 结论

`pos` 错位的直接原因不是模型认不出圆台，而是**代码不允许它纠正**：

- 圆台在入画首帧（x≈570，画面宽 640，物体只有半截 + 运动模糊）被判成圆柱 0.59；
- `_shape_seen[1] = cx` 单帧即写入、永久占位；
- 下一帧起模型连续 6 帧以 0.80~0.92 判对圆台，只能新增键 2，**键 1 撤销不了**；
- 集合变成 `{3,1,2}` → 三形状"齐见" → `inferred=False` → 静默锁死 `pos2=[3,1,2]`，pit2/pit3 双错。

修复思路：**入表单位从"形状"改成"一次连续出现（run）"**，并给 run 加确认门槛；定序用 run 的首次出现时刻，身份用该 run 的持续证据。单帧误判不再能占位。

---

## 0.1 已实施的最小版（2026-11-06）

只做了一件事：**给 `_shape_seen` 加入表门槛 `shape_hits`（连续在场帧数，缺席即清零）**。
v3 的三条改动（跨类去重 / 入画锚点 / `_shape_order`）全部未上。

| 项 | 值 |
|---|---|
| 代码默认 | `_DEFAULT_CONFIG["rescue"]["shape_hits"] = 1`（= 旧"首帧定案"，单测基线） |
| 现场配置 | `eod_config.yaml` → `shape_hits: 3` |
| 锁存最早帧 | `shape_hits + pit_latch_frames - 1` = 4 |
| 观测 | latch 日志新增 `hits=3 peak={1: 1}`；`peak` = 见过但没攒够门槛的形状及其**历史最长**连续帧数（幽灵的指纹） |

**为什么默认值与 YAML 故意不同步**：`tests/` 里有 9 条用例编码了"一形状一帧即定案"的时序
（2 条必须重写、7 条需补帧）。默认值留 1 就能让 418 条基线全绿，现场靠 YAML 生效。
代价是"改代码默认值 = 改 9 条用例"，这是刻意的：谁动门槛谁必须重跑那 9 条。
唯一被迫改的一行是 `tests/test_gks_review_fixes.py` 的 `range(2)` → `range(4)`——
该用例无 config 构造、直接读生产 YAML。

**与 §3.1 的关系（重要，别误杀）**：§3.1 否掉的是**累计帧数**
（幽灵 1 帧 + 真圆柱 12 帧 → `n=13` 照样入表）。本版是**连续帧数 + 缺席清零**：
幽灵 1 帧后 streak 归零，不会被 30 帧后的同名真目标救活。两者不是同一个判据，
§3.1 的反例对本版不成立。

**本版挡不住的情形**（都是已声明的上限，不是 bug）：

- 幽灵**连续** ≥ `shape_hits` 帧 → 照样穿透（`tests/test_rescue_shape_streak.py::test_ghost_three_frames_still_passes` 钉住了这个边界）。
- 某坑可见帧数 < `shape_hits` → 永不入表 → 被 `tail` 补到顺序尾部 → 静默错。`peak=` 是它唯一的可见信号。
- 失败模式 B（同帧跨类双框）与 C（中途重订阅锚点）未处理。

**为什么不做 `max_pits=2` 截断**（"只扫两个 pit"这条物理约束的直接用法）：
03:37 日志里幽灵是**第 2 个键**（`yao_gu → yuan_zhu(幽灵) → yuan_tai(真)`），
`seq[:2]` 会留下 `[3, 1]`，照样错。判断"谁是真"只能靠证据强度，不能靠键的位次。

**为什么不动 `iou_th`**：上游 MaixCDK `_nms()` 的抑制条件是
`a.class_id == b.class_id && iou > iou_th`，**per-class**，且 `detect()` 没有 `agnostic`
参数。所以同一物体被判成两个**不同类别**的框，NMS 永远不会互相抑制——调 `iou_th` 对此无效。
rescue 三形状互不相同、坑间距 ~400px、框宽 ~80px，同类之间 IoU≈0，调它收益也≈0。
真要治"同帧双框"只能加应用层跨类去重（§4.3，8 行、无状态、独立提交），
且注意它的边界：**保留高分框 ≠ 保留正确类别**（幽灵 0.72 / 真目标 0.68 时去重会保住幽灵）。

---

## 1. 现场证据

### 1.1 03:37 段（故障段，conf_th=0.55）

```
[03:37:22] rescue det: yao_gu=0.75(572,325)
[03:37:22] rescue det: yao_gu=0.86(383,331)
[03:37:23] rescue det: yao_gu=0.86(129,344) yuan_zhu=0.59(570,336)   ← 幽灵
[03:37:23] rescue det: yao_gu=0.91(81,344)  yuan_tai=0.80(539,331)   ← 同一物体，31px 后
[03:37:23] latch pos2=[3, 1, 2] seen=[3, 1, 2] order_by=first_seen codes=[1, 3, 2] inferred=False
[03:37:23] rescue det: yao_gu=0.80(71,345) yuan_tai=0.87(516,339)
[03:37:23] rescue det: yao_gu=0.69(55,338) yuan_tai=0.92(467,338)
[03:37:23] rescue det: yao_gu=0.59(35,335) yuan_tai=0.88(438,336)
[03:37:23] rescue det: yuan_tai=0.90(403,338)
```

同一轨迹 570→539→516→467→438→403，只有首帧判错，后续 6 帧全对。系统在**第 2 帧**就完成锁存。

### 1.2 04:14 段（conf_th=0.65 之后）

```
[04:14:21] rescue det: yao_gu=0.66(555,319)
[04:14:21] rescue det: yao_gu=0.76(536,325)
[04:14:21] rescue det: yao_gu=0.88(445,330)
[04:14:21] rescue det: yao_gu=0.69(58,338)  yuan_tai=0.83(492,335)
[04:14:22] rescue det: yuan_tai=0.85(447,338)
[04:14:22] latch pos2=[3, 2, 1] seen=[3, 2] order_by=first_seen codes=[1, 3, 2] inferred=True
```

幽灵消失，`seen=[3,2]`、`inferred=True`、`pos=2` 正确。证明 0.65 能挡住 0.59 这一档；但挡不住 0.65~0.75 之间的幽灵（见 §7 实测）。

### 1.3 同场景两次订阅结果不一致（03:37）

```
第一段: latch pos2=[3, 1, 2] seen=[3, 1, 2] inferred=False
第二段: latch pos2=[2, 1, 3] seen=[2, 1]     inferred=True
```

第二段物理上不可能正确：`yao_gu` 已在 x=35，`yuan_tai` 还在 403~539，前者必然更早入画。根因是 STOP 后重订阅清空了 `_shape_seen`，`first_seen` 的锚点不再是 pit1，`tail` 把已出画的 pit1 补到了 pit3——与 2026-10-09 排爆事故同一机理（`docs/competition/issue_explosive_pit_mismatch.md`）。

---

## 2. 现状算法（改前）

代码位置：`modules/zw_opencv_module/processors/gks_task_processor.py`

| 环节 | 函数 | 行为 |
|---|---|---|
| 定序状态 | `_shape_seen`（`_reset_latch` 初始化） | dict：形状号 → 最近一次画面中心 x；**键的插入顺序 = 坑位顺序** |
| 入表 | `_process_rescue` 检测循环末 | `_shape_seen[shape] = cx`，首帧即写、每帧刷新 |
| 组序 | `_process_rescue` | `seq = list(_shape_seen)`（`order_by:x` 时按 value 排序 + 镜像翻转） |
| 补位 | `_ordered_pits` | 少于 2 个不锁；正好 2 个时把缺的形状补到尾部，`inferred=True` |
| 去抖 | `_latch_debounce` | 连续 `pit_latch_frames`(2) 帧同序才锁 |
| 锁后 | `_process_rescue` | `if self._position2[0] == 0` 守卫，永不重算 |

同帧多目标排序：`objs.sort(key=lambda o: o.x, reverse=not pit_left_is_1)`，保证不受 YOLO 输出顺序影响。

---

## 3. 失效模式

| 代号 | 触发条件 | 机制 | 后果 |
|---|---|---|---|
| **A** 首帧误判 | 入画首帧判成邻类，分数 0.65~0.75，仅 1 帧 | 单帧即写 `_shape_seen`，键永驻；后续正确帧只能新增另一个键 | 集合凭空多一个形状 → "齐见" → `inferred=False` → 两坑错 |
| **B** 同帧跨类重复框 | 同一物体在同一帧吐出两个类别的框（同 x 或相差 1~2px） | 模型 NMS 是类内的，跨类重复框得以保留；`sorted(key=x)` 让 x 最小者抢先入表 | 重复框抢占一个坑位 |
| **C** 中途重订阅 | 重置后第一个目标已在画面中部 | `first_seen` 锚点不是 pit1，`tail` 把缺的补到尾部 | 顺序整体反向 |

三者共性：**用"一帧事件"决定"整段身份"**。

### 3.1 关键负面结论：按"形状"过滤无效

设想用"形状出现帧数 ≥2"过滤幽灵——实测无效。幽灵圆柱与 pit3 的真实圆柱是**同一个形状号**，计数被真实目标救活：

```
幽灵 1 帧(0.72) + 真圆柱 12 帧(0.86)  →  形状 1 的 n=13，照样进集合，结果仍错
```

因此**入表判据不能是"累计帧数"，必须是"当前连续段（run）的长度"**。
§0.1 已实施的 `shape_hits` 就是这条的最简形式：只维护"当前段的连续长度"（缺席清零），
不做 run 的历史追踪。它能挡住"幽灵单帧/短段后接真目标"，挡不住"幽灵连续 ≥hits 帧"；
完整的 run 追踪（`_shape_runs` + `_shape_order` 按 `first_seq` 排序）留给 §4。

---

## 4. 新算法

### 4.1 数据结构（`_reset_latch` 初始化）

```python
self._rescue_seq  = 0      # 帧序号，单调递增，仅用于定序比较
self._shape_runs  = {}     # shape -> {first_seq, last_seq, first_cx, last_cx, n, confirmed}
                           #   ⚠ 必须有 last_seq: 判"上一帧也出现"只能用它 == seq-1,
                           #   不能用 _shape_seen（后者每帧刷新、永不清空, 会把幽灵
                           #   和 30 帧后的真目标续成同一段 = §3.1 否掉的累计式）
self._shape_order = []     # 已确认形状，按 first_seq 升序（order_by:x 时按 last_cx）
self._shape_seen  = {}     # 仅喂 HUD 与 order_by:x 兜底, **不参与 first_seen 定序**
```

### 4.2 逐帧流程（`_process_rescue`）

```
1  self._rescue_seq += 1
2  跨类去重：按 score 降序，与已保留框 IoU > dedup_iou 的丢弃
3  聚合成帧：frame_shapes = {shape: (cx, 本帧该形状最高分)}
4  更新 run：对每个 shape
       if last_seq == seq - 1 且 |cx - last_cx| <= run_window:
            n += 1; last_cx = cx
       else:
            新 run：first_seq = seq; first_cx = last_cx = cx; n = 1; confirmed = False
       if n >= shape_hits 且 not confirmed:
            confirmed = True
            把 (shape, first_seq, last_cx) 追加进 _shape_order
5  组序：seq_list = [shape for (shape, ...) in sorted(_shape_order, key=first_seq)]
        （order_by:x 时改为按 last_cx 升序，镜像翻转规则与现状一致）
6  补位：ordered, inferred = _ordered_pits("rescue", seq_list)
        ★ 锚点门：len(seq_list) == 2 且 首个确认段 first_cx < entry_anchor_x * W
          → 不补位、不锁、保持 pos=0（等第三个形状）
7  锁存：_latch_debounce(ordered) 不变；锁后 _position2 不变
```

### 4.3 子规则①：跨类去重

```python
keep = []
for obj in sorted(objs, key=lambda o: -o.score):
    if all(iou(obj_box, k_box) <= dedup_iou for k in keep):
        keep.append(obj)
```

依据：类内 NMS 允许同一物体跨类共存；不做去重，重复框会参与 `sorted(key=x)` 并抢位。

### 4.4 子规则②：连续段确认

- 同形状 + **相邻帧都出现** + `|Δx| ≤ run_window` 视为同一段；
- 段长 ≥ `shape_hits` 才确认并进入 `_shape_order`；
- **确认后黏住**：出画、漏检都不撤销（避免已出画的真实 pit1 被事后抹掉）；
- 已确认的形状**不再重复入 `_shape_order`**：断段后重新攒够会 append 第二次，
  让 `len(seen)` 虚高、补位 `[s for s in (1,2,3) if s not in seen]` 重复或漏；
- `_shape_seen` 只喂 HUD / `order_by:x` 兜底；未确认的段不写它（HUD 少闪一个幽灵，
  可接受）。first_seen 定序只读 `_shape_order`。

参数依据：日志实测帧间位移 23~49px，坑间距 ~400px，`run_window=140` 不会串坑；每坑在画 30~50 帧，`shape_hits=2` 的安全边际约 15 倍。

### 4.5 子规则③：入画锚点

车前进时目标从右边缘入画（x≈0.87W~0.97W）。若重置后第一个被确认的段首次出现位置远离右边缘，说明扫描已在中途，`tail` 的"未见者 = 最后一个坑"前提不成立：

- **只禁补位**，不禁三形状齐见的正常锁存（三个都看见了，顺序本就完整）；
- 两形状 + 锚点不成立 → 不锁、`pos=0`，等第三个形状。

---

## 5. 参数

| 参数 | 默认 | 含义 | 现场调整建议 |
|---|---|---|---|
| `shape_hits` | 2 | run 确认所需帧数 | 掉帧严重时可放宽到 2；不建议 >3 |
| `run_window` | 140 | 同段允许的最大帧间位移（px） | 车速加快时上调 |
| `dedup_iou` | 0.5 | 同帧跨类去重阈值 | 一般不动 |
| `entry_anchor_x` | 0.85 | 入画锚点（帧宽比例），0 = 关闭 | 相机视野变化时重标 |

配置位置：`eod_config.yaml` `gks.rescue` 段 + `_DEFAULT_CONFIG["rescue"]`（两处必须同步，测试 fixture 只读默认值）。

---

## 6. 行为矩阵（离线重放实测）

重放方式：用 `StubAI`/`Det` 桩把检测序列喂进真 `GksTaskProcessor`，脚本见 §11。

| 场景 | 改前 | v3 |
|---|---|---|
| 三形状先后单独出现 | ✅ pos=2 | ✅ pos=2 |
| 前两个同帧出现、之后分开 | ✅ | ✅ |
| 三形状同帧、模型输出乱序 | ✅ | ✅ |
| 先 1 个、下帧变 2 个（正常入画） | ✅ | ✅（晚 1 帧锁） |
| 首帧幽灵 0.72 仅 1 帧 | ❌ pos=3 | ✅ pos=2，`inferred=True` |
| 同帧同物体双判（三类同 x） | ❌ pos=3 | ✅ pos=2 |
| 中途重订阅，只看到后两个 | ❌ pos=1 | ✅ pos=0（不猜） |
| 只看到一个形状 | ✅ pos=0 | ✅ pos=0 |

耗时：新增逻辑 < 0.08ms/帧（当前 loop 37~48ms、task CPU 6~9ms），可忽略。

---

## 7. 失败模式与兜底

| 情形 | 行为 | 现场可见性 |
|---|---|---|
| 某坑只出现 1 帧 | 该段不确认 → 两形状补位（锚点成立）或 pos=0 | 日志 `drop=[...]` |
| 锚点不成立（扫描中途起步） | 不补位、pos=0 | 日志 `anchor=mid` |
| 三形状齐见 | 直接锁，`inferred=False` | 无歧义 |
| 模型整体低分（< 0.65） | 全部不入集合 → pos=0 | 一眼看出 |

新增日志（沿用"变化才打"，禁止逐帧打印）：

```
[GKS] shape ok=2 n=3 first=(536) drop=[1] anchor=edge
```

`inferred` 在全仓**没有任何 .py 消费方**（只有 HUD/调参页显示），因此它是"给人看的"信号，不能当作系统级兜底——真正的兜底是"该锁不上就不锁"。

---

## 8. 测试计划

1. 新增 `tests/test_rescue_shape_identity.py`，7 条场景各一条，断言 `latched_order / inferred / pos`（对应 §6 表格）。
2. 现有基线：`pytest tests/ -q` → **409 passed**。预计 9 条 rescue 用例需按"每形状 ≥2 帧"调整帧数（2→3）；其中 `test_rescue_order_is_first_seen_across_frames`、`test_rescue_first_seen_multi_frame_ignores_mirror` 编码的正是"一形状一帧即定案"的旧语义，应**重写**而非放宽断言。
3. 台架：前进扫三坑 20 次，统计 `pos` 与实物一致率、`inferred` 出现率；同时记录每坑在画帧数，确认 `shape_hits` 的安全边际。

---

## 9. 提交拆分与回滚

| 提交 | 内容 | 独立回滚 |
|---|---|---|
| 1 | 跨类去重（8 行，无状态） | ✅ |
| 2 | 连续段确认 + `_shape_order` 定序 | ✅ |
| 3 | 入画锚点 | ✅ |
| 4 | 测试用例 + 文档 | ✅ |

已提交的 `ebbe1f8`（conf_th 0.65）保持不动，四条改动都不依赖它。

---

## 10. 明确不改的边界

`_ordered_pits`、`_latch_debounce`、`_order_by_x`、`_infer_missing`、UART 协议与 `pos` 语义、HUD 接口、`framework/hal/*` 全部不动；`_shape_seen` 保留供 HUD 与 `order_by: x` 使用。

不做的事：
- **不做轨迹跟踪器（track voting）**：每坑 30~50 帧，run 确认已覆盖单帧误判；引入 track 需改 HAL 推理返回值，风险与收益不成比例。
- **不做"锁存后重算"**：Brain 首次收到非零 `pos` 即锁死 `align_target_pit`（`docs/competition/issue_explosive_pit_mismatch_brain_reply.md`），事后修正会被丢弃。

---

## 11. 附录：离线重放方法

重放脚本（临时目录，不入库）：`%TEMP%\rescue_replay.py`、`%TEMP%\rescue_two.py`

```python
class StubAI:                     # labels 顺序 = model.mud: yao_gu, yuan_zhu, yuan_tai
    loaded = True
    labels = ["yao_gu", "yuan_zhu", "yuan_tai"]
    def detect(self, frame, conf_th=0.5, iou_th=0.5, **kw):
        return self.dets           # list[Det(x, y, w, h, class_id, score)]

proc = GksTaskProcessor(config=cfg)   # cfg 用 load_gks_config("/nonexistent")
proc.set_ai(ai); proc.set_task(MODE_RESCUE, (0, 0, 2))
r = proc.process(np.zeros((480, 640, 3), np.uint8))
r.result_data["latched_order"], r.result_data["pos"], proc.pit_status["shapes_seen"]
```

类别映射（易错）：`class_id 0 = yao_gu(形状3)`、`1 = yuan_zhu(形状1)`、`2 = yuan_tai(形状2)`。

---

## 12. 参考

- `modules/zw_opencv_module/processors/gks_task_processor.py`：`_process_rescue` / `_ordered_pits` / `_latch_debounce` / `_reset_latch`
- `modules/zw_opencv_module/config/eod_config.yaml`：`gks.rescue`
- `docs/competition/issue_explosive_pit_mismatch.md`：同类补位事故的复现记录
- `docs/eod/method_selection.md`：形状识别方案选型
