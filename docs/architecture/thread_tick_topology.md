# 线程与节拍拓扑(Thread-Tick Topology)

> 状态: 2026-08-22 随 P2 定契约成文 · 依据: 复盘 D3/D4/D6 决策 + main 分支实测线程清单
> 配套原语: `framework/slot.py`、`framework/cmd_queue.py`、`framework/event_bus.py`

## 1. 通信机制选型表(三原语,不再新增)

| 数据/动作性质 | 原语 | 典型数据 | 丢失语义 |
|---|---|---|---|
| 连续状态,中间值可丢 | **Slot**(`framework/slot.py`) | 相机帧、AI 检测结果、fps/link 状态、待呈现帧 | 只保留最新;读方按代数号(gen)判新 |
| 有序动作,不可丢但可合并 | **CmdQueue**(`framework/cmd_queue.py`) | 标定请求、录制开关、模型切换、EXIT、按键 | FIFO 保证执行;同 kind 待处理时合并 |
| 稀疏异步广播,多方关注 | **EventBus**(`framework/event_bus.py`) | 急停、链路断开、PC 上/下线 | 不保证送达;订阅者自行容错 |

选型问句:这条数据"丢一帧要不要紧?"——不要紧→Slot;要紧且会重复按→CmdQueue;
多方要同时知道→EventBus。三者都不匹配说明需求本身该重新审视。

## 2. 硬规则

1. **主循环 tick 是唯一消费者枢纽**:生产者(RX 线程/UI 回调)只准入队/发布,
   禁止直接调用其它线程的方法。
2. **lambda 队列废除**:命令必须是 `Cmd(kind, payload)`,消费处天然获得审计日志。
3. **显示两段式**(D3/D4):合成在视觉线程内经 composer hook 完成(不开线程);
   呈现由主循环每 tick `SinkGroup.flush()` 泵出,sink 自持 fps 限流,**不自带线程**。
4. **稳态常驻线程预算 ≤ 4**;新增常驻线程必须在 PR 描述里给出豁免理由。
5. **锁内禁止 IO**:publish/drain 只做内存操作;慢操作(UART 写、socket send、
   模型加载)一律移出临界区或改为入队(CONC-01/02/03 的根因即违反此条)。

## 3. 线程清单(实测于 main@92d0b49)

| # | 线程 | 启动点 | 职责 | P4 处置 |
|---|------|--------|------|---------|
| 1 | main tick | `module_manager.run()` | 主循环节拍器 | **保留**,吸收 beacon/persist/flush 泵 |
| 2 | vision loop | `vision_manager._vision_loop` | 读帧→推理→合成 | **保留**(合成在此线程) |
| 3 | display loop | `main._display_loop` | 轮询取帧→display.show | **删除**(由 main tick flush 泵替代) |
| 4 | log-writer | `log_util` 启动 | 异步落盘日志 | **保留**(磁盘 IO 隔离) |
| 5 | beacon | `main._beacon_loop` | UDP 发现广播 | **并入 main tick**(纯发送无阻塞) |
| 6 | fill-light persist | `main._writer_loop` | 补光状态写盘 | **并入 main tick**(队列改 CmdQueue 式 drain) |
| 7 | pc-heartbeat listener | `pc_heartbeat._listen` | UDP 心跳接收 | **保留**(阻塞 recvfrom 无法并入,豁免理由:纯等待无 CPU) |
| - | record-notify | `coordinator:211` 瞬态 | 录制信令 UDP×3+sleep | 改为 CmdQueue 动作,消灭瞬态线程 |
| - | streamer | `zw_wifi_stream` | MJPEG 推流 | 按需存在(唯一允许自带写线程的 sink),关闭即无 |

目标稳态:**main / vision / log-writer / pc-heartbeat = 4** ✓ 预算达标。

## 4. 反模式清单(评审时逐条对照)

- `show()` 返回值携带退出语义(DIS P-04 已修方向:输入走 InputSource→CmdQueue)
- 锁内做 sleep/sendto/model.switch(CONC-02/03)
- 对象身份比较判新鲜(`frame is not _last_seen_frame`)→ 一律 Slot.gen
- 为每条数据流临时发明机制(手写 deque drain、布尔旗标+轮询)
- 在 RX 回调线程做重活(CONC-01 ai.switch)→ 入队,主循环执行

## 5. 迁移映射(P4 执行)

| 现状(位置) | 目标 |
|---|---|
| `main.py:193` 帧身份比较 | `Slot.load(seen_gen)` |
| coordinator `_pending_results` deque drain | `Slot`(最新结果语义) |
| 触摸按钮直调(main.py:141-176) | InputSource.poll → CmdQueue(CALIB/LED/EXIT) |
| `coordinator:211` 瞬态通知线程 | Cmd("RECORD_NOTIFY") 由 main tick 执行 |
| RX 线程 `ai.switch`(CONC-01) | Cmd("SWITCH_MODEL") 入队,主循环切换 |
| `_display_loop` + `get_display_frame()` | vm.last_frame Slot → SinkGroup.push/flush |

## 6. 护栏

- fitness function(BRANCH-03,进 pre-commit):
  `grep -rnE "import (maix|cv2|serial)" framework/ modules/` 必须为空(app/ 豁免)
- 单元测试:`python -m unittest discover -s tests`(Slot/CmdQueue 行为契约)
