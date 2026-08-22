# Code Review Report — Zulu-Walker 2026H 赛后框架审查

> 日期: 2026-08-22 · 审查对象: main@92d0b49
> 详细问题清单见同目录 `framework_issues.md`(本报告为其执行摘要与健康项附录)

## 1. 审查方法

两阶段:

1. **人工架构走查** — HAL 接口完备性、显示链路类型契约、调度机制盘点、模块职责边界、配置一致性、分支拓扑。产出结构性问题(ARCH/DISP/SCHED/MOD/DEAD/CFG/BRANCH 系列)。
2. **四路并行子代理深审** — 每路限定文件范围 + 排除已知问题清单 + 强制 file:line 证据:

| 路 | 范围 | 产出 |
|----|------|------|
| A | framework/(event_bus, module_manager, state_machine)+ utils/ + main.py 资源管理面 | 15 条 |
| B | zw_uart_module/ + coordinator.py(729行)+ pc_heartbeat | 11 条 |
| C | 运维·安全·打包(仓库卫生/app.yaml/依赖/socket) | 3 条(范围截断) |
| D | pipeline_camera/task_manager/maixcam2 相机层/YOLO handler/pendulum_calibrator | 12 条 |

已知问题(此前复盘已识别的 HAL 缺口、上帝类、调度三机制、配置漂移、死代码等)在各路 prompt 中明确排除,子代理只找增量。

## 2. 统计总览

- 问题总数:**~70 条**(2026-08-22 复审后:SCHED-02 证据失效已改写降级,其余全部核实)
- 分布:[高] 12 · [中] ~38 · [低] ~20
- 高危构成:协议矛盾 1(PROTO-01)、并发竞态 1(CONC-01)、数据正确性 1(VIS-01 陈旧帧)、资产 1(OPS-01 凭据)、持久化 1(非原子 YAML 写)、日志雪崩 1(LOG-01)、结构性 6(ARCH-01/06、DISP-01/02/03、MOD-01)

子系统热力(条目数):显示/调度/HAL 结构性债务最深(跨平台主因);UART 协议与视觉管道藏着全部"当场翻车型"正确性问题;日志系统被三路独立命中(交叉印证)。

## 3. Top 10 风险速览

| # | ID | 一句话 | 为什么排这里 |
|---|-----|--------|--------------|
| 1 | PROTO-01 | PENDULUM 帧布局代码/注册表/文档三方矛盾 | MSPM0 按文档实现则主控链路必不通;协议复用前的头号地雷 |
| 2 | CONC-01 | ai.switch 在 RX 线程与推理线程裸奔竞态 | 秒级窗口的 NPU 并发访问,板端可致段错误 |
| 3 | VIS-01 | 相机故障期冻结帧持续流入控制路径 | 向主机上报"看似正常的假数据",比崩溃更危险 |
| 4 | OPS-01 | WiFi 凭据在公网仓库历史 | 删文件无效,需历史重写 |
| 5 | —(main.py:453) | 标定落盘非原子写,断电即配置全毁且静默 | 比赛现场高频断电场景,已有正确范式在同文件未复用 |
| 6 | LOG-01 | 主循环异常日志无限速 ~500Hz 同步写盘 | 次要 bug 可拖垮主循环节拍+写满 SD |
| 7 | CTRL-01 | 丢帧计数节拍不对称,容限缩水几十倍 | found 位高频抖动的直接根因 |
| 8 | CONC-03 | 录制信令 150ms sleep 在 cmd_lock 内 | 演示录制开始瞬间 DATA_STREAM 断流 |
| 9 | DISP-01/02 | 显示类型契约破裂 + 11 处静默失效 | 跨平台第一暗雷:不报错、功能无声消失 |
| 10 | CTRL-02 | 主机链路超时是死字段 | 链路死亡不可感知,link_active 恒真误导排障 |

## 4. 已确认健康项(避免重复审查)

各路代理明确核实无问题的方面——后续 review 可直接引用本节跳过:

- **EventBus**:history deque(maxlen=1000)内存有界;unsubscribe 无泄漏;KeyboardInterrupt 传播干净(break→stop_all 资源正常释放)
- **UART**:CRC 计算构建/解析对称正确(CCITT 表驱动);send_raw 多线程经 _write_lock 串行化,锁序恒为 cmd_lock→write_lock 无反序;发送异常不改订阅状态无泄漏;PENDULUM 构建器三字段有 clamp 无 int16 溢出;缓冲区上限 258B、粘包半包按字节处理正确
- **α-β 滤波**:LOCKED 状态无永久死区(snap+bias_sum 双通道解锁完备);UNLOCK_JUMP 后速度尖峰为 g-h 固有特性非缺陷;pixels_per_cm 配错属配置风险(门限随之缩放)非代码缺陷
- **视觉管道**:ROI 亮度统计开销可忽略;Phase1 独占相机先于视觉线程启动、与 AEC 无接管竞态;camera_id 跨重连稳定(CameraHub 引用不变);重连退避(3s stall+5s 固定)够用;queue_size=3 丢最旧帧对控制延迟无害(上界≈2 帧周期);TaskManager 串行设计本身无并发缺陷;profiler 已全局禁用
- **utils**:point.py acos 截断/零向量守卫完备;focal_distance_util 除零有防护;cpu_affinity 的 bind_current_thread 实现正确;console_capture 机制本身正确(print→队列,writer 回写走 __stdout__ 无递归)

## 5. 待办中的未决事项

- **VIS-02(image2cv 零拷贝视图 vs 驱动缓冲复用)**:离线无法证实,需板端实验(推理中帧内容 hash 比对)。安排在下一次上电窗口。
- **CONC-01 底层耐受性**:maix nn C++ 层是否耐受并发访问,需板端压测。
- **C 路审查范围截断**:依赖钉扎核查、subprocess 注入面、docs 死链抽查未完成,下届开工前补一轮。

## 6. 建议行动顺序(详见 issues 清单附图)

1. **本周可做的热修**(不动架构):PROTO-01/02 文档对齐✅(已完成 2026-08-22) → main.py:453 原子写 → CTRL-01 门控 → VIS-01 帧序列号贯穿 → OPS-01 凭据处置
2. **减重半天**:DEAD 系列 + app.yaml 清单修正
3. **定契约一周**:Slot/CmdQueue/FrameSink/Canvas 四原语落地 + thread_tick_topology.md 选型表(像素容器契约已决策:平台原生类型 + Canvas 协议,不走 canonical cv2——AX630C 硬件 VO/JPEG 路径优先)
4. **平台收编**:五个 capability + PLATFORMS 元数据 + exit_check 注入;fitness function(grep 判据)进 pre-commit
5. **拆 VisionManager**:依赖 3、4 完成
6. **治理收尾**:配置单一事实源、分支归档、LOG/SM/LIFE 中低项批量清

## 7. 给下一届的一句话

这套框架的 HAL 骨架是对的,比赛期欠的是"把平台依赖赶进笼子"的纪律——`grep -rnE "import (maix|cv2|serial)" framework/` 输出为零的那天,换板子就是新建一个 platforms/<name>/ 目录的事。

## 8. 准确性复审记录(2026-08-22 第二轮)

对两份文档的全部 file:line 引用逐条对照 main@92d0b49 实测复核,结论:**~95% 引用精确命中**(含 vision_manager 1065 行整、分支落后数 57/373、OPS-01 提交祖先可达等硬断言)。修正项:

| 项 | 原文 | 实测 | 处置 |
|----|------|------|------|
| SCHED-02 | coordinator `_enqueue_sm(lambda:)` 模式 [高] | HEAD 已无此机制(git log -S:随 d11207a 移除);现状为显示线程内联直调(main.py:141-176) | 改写为[中],CmdQueue 建议保留为 D6 前瞻规则 |
| DISP-02 | "14 处" inline import maix.image | 实数 **11 处**(:462-975) | 已改 |
| MOD-01 | "8 个 set_xxx(main.py:650-682)" | main.py 实调 **9 个**,散布 :161/:266/:655-659/:672-680/:718-719;vm 定义 13 个 set_* | 已改 |
| 非原子写引用 | main.py:443 | 直写在 **:453-454**(441 为函数 docstring) | 已改(roadmap/Top10/§6) |
| ARCH-01/04、MOD-03 行号 | :85 / __init__:50 / cpu_affinity:8-13 | :86 / :57 / :11-15(±1~7 行漂移) | 已改 |
| LOG-02 | "logs/app.log 被 git 跟踪" | git 仅跟踪 performance.log;两者均打进包(app.yaml:70-71) | 已改口径 |
| CONC-02/SCHED-01 | 裸写 event_bus.py、_enqueue_sm 残留 | 补全路径 framework/event_bus.py;清除残留引用 | 已改 |

未改动但值得知道的核实结论:CONC-03 的 150ms 属实(for 循环 3×sleep(0.05));CALIB-01 的 `except: pass` 在 main.py:430-431 外层;CALIB-03 降级链缺陷原文在 calibrator `__init__.py:196-200`;streamer.py:38 为 `from maix import http`(措辞略异,实质一致);camera_hub shim 经查**零引用方**,可直接删。
