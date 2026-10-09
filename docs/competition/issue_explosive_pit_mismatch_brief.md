# 给电控侧 Agent 的一段话（可直接复制粘贴）

---

**背景**：反恐排爆救援机器人，视觉从站 = MaixCAM-Pro（Zulu-Walker），主站 = panzer_Brain。
链路：CAM → USART1 @115200 → Brain，协议 `panzer_links_protocol v2.1`。

**现场现象**：
任务码输入 `321`（协议语义 = 爆色3蓝 / 靶色2绿 / 形状1圆柱 ⇒ 排爆阶段目标应为**蓝色球**）。
实物：**蓝球在 pit1**、绿 pit2、红 pit3。相机扫描顺序正常（先 pit1 再看 pit2）。
**实际动作：转台自动转到 pit3，抓了红色；蓝球（pit1）没抓。**

**视觉侧已完成的走查结论**：
1. 红蓝通道序已逐环节核对，**不会互换**（含 `_as_rgb888` 兜底 + 降采样 `bgr=False`，有回归测试）；当前设备阈值下排爆三色 `red/green/blue` 三个 LAB 盒**两两互斥**。
2. 任务码解析正确：`321` → `codes=(爆色3, 靶色2, 形状1)`，排爆取 `_codes[0]=3`（蓝）。
3. **根因在坑位定序**：排爆用 `order_by=first_seen`（谁先被看到谁就是 pit1）+ `infer_missing=tail`（见 2 色即锁，未见者补到**尾部**）。
   前提是"第一个被看到的颜色物理上在 pit1"。PC 端已复现：**若 pit1 的蓝球首帧漏检**，则 `seen=[2,1]`（绿、红）→ 补位成 `ordered=[2,1,3]`（蓝被塞到 pit3）→ 上报 `pit=3` → Brain 转 pit3 → 抓到红球。**与现场现象逐字一致**，且锁存后不再纠正。

**请 Brain 侧协助核对这几点**（对应 `interrupt.c` / 相关发送与解析代码）：

> **📌 已核对完毕（2026-10-09，Brain 侧回报）**：5 点**全部通过，Brain 无自身缺陷**。
> 结论：Brain 是**忠实执行被算错的 pit=3**，根因确认为 CAM 端 `infer_missing: tail` 补位。
> 完整回执见 `docs/competition/issue_explosive_pit_mismatch_brain_reply.md`。
> 其中新发现一个**锁存-修正**问题：Brain 在 `align_target_pit` 非 0 后停止重读相机，
> CAM 后续即使修正 pit 也可能被静默丢弃 —— 见回执 §3。

1. **任务码发送顺序**：`321` 是否按 `[data_id, min_interval_ms, 爆=3, 靶=2, 形=1]` 填入 CMD_REQUEST 的 payload[2..4]？顺序错会导致 CAM 收到的爆色不是蓝。
2. **`found=0 / pit=0` 处理**：CAM 未锁存时推 `found=0, pit=0`。Brain 是否严格忽略这类帧（不动作、沿用上一有效值），而不是把 0 当合法值或保留旧 pit 去动作？
3. **`pit` 消费偏移**：读 DATA_STREAM 时，`pit` 是否取 sub_payload 偏移 2（即 `seq(1)+data_id(1)+cx u16LE(2)` 之后）？别把 cx 高字节误当 pit。
4. **pit↔转台角度映射**：pit1/2/3 到转台角度的标定方向是否与实物一致？（Brain 侧映射镜像也会表现为"转到反向的坑"）
5. **能否提供本次收到的 `pit` 序列 / 原始 DATA 帧？** 与 CAM 侧日志对照，可立刻区分"CAM 推错"还是"Brain 读错"。

**CAM 侧复现时会打这些日志**（可用于对照）：
```
[GKS] seen += 2 -> [2]
[GKS] seen += 1 -> [2, 1]
[GKS] latch pit=3 seen=[2,1] ordered=[2,1,3] codes=[3,2,1] inferred=True
```
若 `seen` 里没有 3（蓝）且 `inferred=True` ⇒ 确认"蓝球漏检→补位塞尾部"。

**协议字段速查**（v2.1）：
- CMD_REQUEST(0x20) 完整 5B payload = `[data_id, min_interval_ms, 爆炸物色, 靶标色, 人质形状]`
- DATA_STREAM(0x24) 10B = `[seq, data_id, sub_payload 8B]`
- 0x01 爆炸物 sub_payload = `[cx u16LE@0-1, pit u8@2, flags u8@3, rsv×4]`，`pit∈[1,3]`，0=未定位
- 有效判据：`found=1 且 pit∈[1,3]` 才采纳

完整简报见：`docs/competition/issue_explosive_pit_mismatch.md`
