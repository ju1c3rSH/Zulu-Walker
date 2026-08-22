# Zulu-Walker 框架问题总清单(2026 赛后复盘)

> 生成日期: 2026-08-22
> 来源: 人工架构走查(HAL/调度/显示/模块边界) + 4 路子代理深审(framework·utils / uart·coordinator / 运维·安全·打包 / 视觉管道·相机层)
> 配套文档: `code_review_report.md`(审查方法、统计、健康项);本文件为完整问题清单
>
> 严重度定义:
> - **[高]** 数据正确性错误 / 资产暴露 / 阻碍跨平台的核心结构性问题
> - **[中]** 显著增加维护成本或影响稳定性、可观测性
> - **[低]** 卫生问题,不影响功能
>
> ID 约定: `ARCH`架构HAL `DISP`显示 `SCHED`调度 `MOD`模块边界 `DEAD`死代码打包 `CFG`配置 `PROTO`协议 `CONC`并发 `CTRL`控制逻辑 `VIS`视觉管道 `CALIB`标定 `LOG`日志 `SM`状态机 `LIFE`生命周期 `UI`调试台 `OPS`运维安全 `BRANCH`分支护栏

---

## 一、架构与 HAL 层

### ARCH-01 [高] 触摸屏无 HAL 抽象,app 层直连 maix 并使用专有坐标换算
- **位置**: `app/main.py:86`(直接构造 `touchscreen.TouchScreen()`)、`app/main.py:116`(`maix.image.resize_map_pos_reverse` 做帧→屏坐标映射)
- **描述**: HAL 定义了 Camera/Display/Uart/AI 四个 Protocol,但没有 Touchscreen。触摸读取与"显示帧坐标→屏幕坐标"的换算全部写在 app 层 main_callback 内,且换算依赖 maix 专有 API。
- **后果**: 换任何带触摸的平台(或桌面用鼠标模拟)都必须重写 app 主文件;触摸命中测试逻辑无法复用。
- **建议**: 新增 `Touch(Protocol)`(read + map_to),maixcam2/linux(mock 可 None)各给实现;键盘作为桌面的等价输入源。

### ARCH-02 [中] WDT 无抽象:构造、喂狗策略、禁用逻辑 45 行全在 app 层
- **位置**: `app/main.py:214-259`(`_make_wdt_feed`,直连 `maix.peripheral.wdt.WDT(0,10000)`)
- **描述**: 项目定位是"Maix 平台保留 WDT,通用 Linux 不需要",但现状是 app 直接认识 maix 的 WDT 类型。真正有价值的部分——启动阶段多点喂狗(WiFi 后、模型加载后、Phase-1 循环内)——是纯调度策略,却和平台实现耦合在一起。
- **后果**: 无法按平台替换看门狗来源(MaixCAM2 硬件狗 / Linux systemd watchdog / 无);app 层存在平台依赖。
- **建议**: `Watchdog(Protocol)`(feed/disable)+ 各平台 `create_watchdog()`(不支持返回 None);main 缩减为 `wdt_feed = machine.watchdog.feed if machine.watchdog else lambda: None`;多点喂狗策略原样保留。

### ARCH-03 [中] WiFi AP/STA 启动逻辑整体写在 app 层
- **位置**: `app/main.py:270-306`(`_init_wifi`,直连 `maix.network.wifi`)
- **描述**: 用户已定位 WiFi 为项目特化功能、框架不需要。问题不在"缺抽象",而在**归属未定义**:特化代码散在 app 主文件里,没有声明式的方式表达"此功能仅 maixcam2 有效"。
- **后果**: 换平台时要么 ImportError 要么静默半失效;app 主文件持续膨胀。
- **建议**: 移入独立模块并加元数据 `PLATFORMS = ("maixcam2",)`;ModuleManager.load 时平台不匹配则跳过并记日志。

### ARCH-04 [中] GPIO/补光灯半成品:平台函数存在但绕过 Machine 直取
- **位置**: `framework/hal/platforms/maixcam2/__init__.py:57`(`set_fill_light`)、`app/main.py:571-576`(try-import 该函数)
- **描述**: 补光灯实现放在了正确的目录,但获取方式是 app 层 try-import 平台包内部符号,而不是通过 Machine 注入。这使平台内部符号变成了公共接口,破坏了封装。
- **后果**: 与 ARCH-03 同类:平台耦合 + 归属不清(用户已定位为项目特化)。
- **建议**: 同 ARCH-03 的 PLATFORMS 元数据方案;或最小改动:Machine 增加可选属性由平台工厂填充。

### ARCH-05 [中] HTTP 图传绑死 maix.http.JpegStreamer
- **位置**: `modules/zw_wifi_stream/streamer.py:38`
- **描述**: JpegStreamer 类的线程循环内 import `maix.http`,整个模块只能在 MaixPy 固件上运行。用户已定位为项目特化。
- **后果**: 同 ARCH-03;且该模块与"FrameSink 应为通用协议"的方向冲突(见 DISP-03)。
- **建议**: 改造为 `JpegStreamSink`(实现 FrameSink 协议),maix http 部分保留在 maixcam2 平台路径内;模块加 PLATFORMS 元数据。

### ARCH-06 [高] 框架核心反向依赖平台:module_manager try-import maix.app
- **位置**: `framework/module_manager.py:67`(`from maix import app as _maix_app` 做 need_exit 轮询)
- **描述**: framework 是最应平台无关的一层,却为了"检测退出"感知了具体平台的模块。虽有 ImportError 兜底,但依赖方向已经反了:应该是平台告诉框架如何检查退出,而不是框架去探测平台。
- **后果**: 破坏分层契约的先例;每多一处此类代码,跨平台测试矩阵就多一个隐性分支。
- **建议**: 最小改法:`ModuleManager(exit_check=None)` 注入谓词,maixcam2 平台提供 `create_exit_check()`(返回 `app.need_exit`),其它平台 None。彻底改法:退出统一为 CmdQueue 的 `Cmd("EXIT")` 命令(见 DISP-04)。

### ARCH-07 [中] capability check("try-import 探活")重复散布
- **位置**: `app/main.py:30-35` 与 `app/coordinator.py:15-20` 各写一遍 `_HAVE_MAIX_SYS`(探测 `maix.sys.memory_info` 用于 CMM 内存压力检查)
- **描述**: 两处独立 try-import 同一平台能力做同一件事。原则应是:**平台能力不需要在远处重新发现——Machine.create 已经 import 过平台包,问 Machine 即可**。任何 platforms/ 目录之外的 `try: import X except ImportError` 都是一次 HAL 缺口现形。
- **后果**: 能力判定逻辑漂移风险;同一探活在多处维护。
- **建议**: `SysInfo(Protocol).memory_pressure() -> Optional[float]`(None=平台无此概念),maixcam2 读 CMM、linux 读 /proc/meminfo、mock 返回 None。

---

## 二、显示链路

### DISP-01 [高] Display Protocol 类型契约破裂:声明 ndarray,实际传 maix.Image
- **位置**: `framework/hal/interface/display.py`(`show(frame: np.ndarray) -> bool`) vs `modules/zw_opencv_module/vision_manager.py:420`(`disp = raw_img.copy()` 为 maix.image.Image)、`framework/hal/platforms/maixcam2/display.py`(`show(self, frame)` 不做类型转换直接透传)
- **描述**: 接口声明的像素容器与实际传递的不一致,且 Protocol 只是名义约束无运行时校验。maixcam2 收 maix.Image,mock/linux 收 ndarray——同一个接口两种语义。
- **后果**: 跨平台时所有依赖该签名的代码行为不可预测;静态检查形同虚设。
- **建议**: **【已决策 2026-08-22】不采用 canonical cv2**(cv2image 转换在 AX630C 上是实打实的全帧拷贝,而 maix.Image 直通硬件 VO/JPEG 编码路径)。契约定为:**canvas 像素容器 = 平台原生类型;跨平台代码只准经 Canvas 协议(见 DISP-03)触碰像素**。Canvas 与 Sink 由同一平台包提供,类型天然匹配,全链路零转换。

### DISP-02 [高] vision_manager 内 11 处 inline `import maix.image` 在非 maix 平台静默失效
- **位置**: `vision_manager.py:462-975`(轨道绘制、cm 刻度 sprite、按钮、横幅、检测列表等全部绘制函数)
- **描述**: 所有 overlay 绘制走 maix.image API,且用 `except ImportError: return` 兜底。这意味着在 linux/mock 平台上**不是报错,而是按钮、轨道、检测列表无声消失**。
- **后果**: 调试时 UI 功能莫名缺失极难排查;显示层成为纯 maix 专属代码。
- **建议**: 绘制逻辑迁入 `OverlayRenderer`,只依赖 `Canvas` 协议(7 个冻结原语:copy/line/rect/circle/string/blit/w-h);maixcam2 后端委托 maix 原生绘制并保留 sprite 缓存等硬件路径优化,desktop 后端委托 cv2。护栏:①原语集冻结,禁止后端暴露原生对象逃逸;②金帧一致性测试(两后端渲染同一场景断言结构等价);③sprite 缓存属后端私有优化不进协议。代价:每个新 overlay 效果写两份委托(~10-20 行),是选硬件加速路径的固定租金。

### DISP-03 [高] 合成与呈现未分离:一个 show() 背两个职责
- **位置**: Display 接口整体;`vision_manager._update_display_frame`(合成)、`app/main.py:184-209` `_display_loop`(呈现轮询)、`vm.set_capture_sink`(推流分发,已是隐式 Sink)
- **描述**: "把 overlay 画进帧"(内容生成)与"像素如何到达屏幕"(呈现)焊在一个接口。MaixCAM 是推完整帧模型,桌面 cv2.imshow 其实也是推帧模型,Qt 才是 GUI 事件循环模型——不拆开就无法同时支持。
- **后果**: 显示相关代码无法在任何第二平台上复用;VisionManager 因此长出 8 个 set_xxx 注入口。
- **建议**:【已决策 2026-08-22】拆为 `Canvas + OverlayRenderer`(合成)+ `FrameSink`(呈现)。合成走**钩子式**:模块暴露通用 `set_composer(fn)` 钩子,fn 由 app/display 提供、经 Canvas 协议绘制,仍在视觉线程内执行(**不开新线程**——双核平台上多线程=GIL 争用,框架需保持轻量以同时覆盖 MaixCAM2/RK3588)。呈现为 **flush 泵式**:Sink 不自持线程,由主循环 tick 调 `sinks.flush()`,各 sink 按代数号+自身节流决定是否上屏。现有代码归位:`_display_loop` 删除→MaixLcdSink.flush 替代(净减一个常驻线程)、`set_capture_sink/_maybe_push_capture`→Sinks 扇出、streamer→JpegStreamSink(唯一允许自带写线程的 sink,仅流媒体开启时存在)、新增 CvWindowSink(imshow+waitKey 在主循环泵中执行,恰好满足 HighGUI 窗口线程约束,键盘事件顺路进 CmdQueue)。

### DISP-04 [中] 退出语义混进呈现层:show() 返回 False 表示按 q
- **位置**: `framework/hal/platforms/linux/display.py:13-15`(waitKey 检测 'q'/ESC 返回 False)
- **描述**: HighGUI 把输出(imshow)和输入(waitKey 取键)焊在一起,LinuxDisplay 顺势把"用户要求退出"的决策藏进了显示调用的返回值。退出决策应属于命令通道。
- **后果**: 退出路径不可审计;触摸退出(main.py os._exit)、'q' 退出(debug_console os._exit)、maix need_exit 三条退出路径行为不一致(UI-02 有实际后果)。
- **建议**: sink 把键值交给 `InputSource → CmdQueue`,主循环读到 `Cmd("EXIT")` 才退,单一出口;show() 返回值删除。

### DISP-05 [中] 显示节流三处各自为政
- **位置**: `app/main.py:10,196`(`_DISPLAY_EVERY_N=2` 跳帧)、`vision_manager.py:408`(`_DISPLAY_REBUILD_INTERVAL_S` 重建节流)、`linux/display.py`(无任何节流)
- **描述**: 同一件事(控制显示频率)在三个层次各有一套且互不知晓。规则应当是:**节奏属于生产者(VisionManager 决定重建频率)和消费者端(sink 自持泵频+Slot),中间不许再插手**。
- **后果**: 调整显示性能时要改三处;不同平台组合下节流效果不可推理。
- **建议**: 删除 app 层 `_DISPLAY_EVERY_N`;每个 Sink 构造参数带 fps_limit,内部 Slot 存最新帧 + 代数号对比跳过未变帧。

### DISP-06 [低] 两套呈现栈并存,分层规则未成文
- **位置**: 生产栈(LinuxDisplay/MaixCam2Display 走 HAL) vs 调试栈(`BaseDebugWindow` trackbar 调参面板,detectors/\*/debug,直连 cv2 完全绕过 HAL)
- **描述**: 调试窗口当时刻意没花功夫走抽象——这个决定本身正确,但它现在是**未写下来的默认**,新成员无从判断"什么时候可以直连 cv2"。
- **后果**: 规则靠口口相传;有人可能把调试栈模式带进生产代码。
- **建议**: 成文三条:"debug UI 仅经 `run.py debug` 加载;允许直连 cv2;永不进入设备打包清单(app.yaml exclude debug 目录)"。

---

## 三、调度模型

### SCHED-01 [高] 三种通信机制并存,选型无规则
- **位置**: 全仓。EventBus 发布订阅(uart 事件)/ 显式队列 drain(_pending_results deque)/ 回调注入(wdt_feed、send_raw、capture_sink、rail_provider)(注:_enqueue_sm 已随重构移除,见 SCHED-02 复审说明)
- **描述**: 同一份数据走哪种机制取决于代码书写时的历史,而非数据性质。没有人能不看实现就回答"这条数据谁发给谁、在哪个线程、丢了会怎样"。
- **后果**: 新增一条数据流时每次都要重新纠结机制;并发 bug 的根因分析成本高。
- **建议**: 定选型表并写入 `docs/architecture/thread_tick_topology.md`:**Slot 最新值**(连续状态,丢失容忍:球位置/显示帧/fps/link 状态)、**CmdQueue 有序命令**(不可丢可合并:SM 触发/录制指令/标定请求)、**EventBus 广播**(稀疏异步通知:急停/链路断)。GIL 上 tick 轮询 + 三原语的可调试性优于精致事件驱动——这是本届比赛验证过的经验,不要倒退。

### SCHED-02 [高→已改写·中] UI 动作在显示线程内联直调:无队列、无去重、无审计(原"lambda 队列"证据失效)
- **位置**: `main.py:141-176`(display 回调线程内同步直调:Phase2 标定 :150-157、补光切换 :167-176)
- **复审说明【2026-08-22】**: 原清单引用的 `coordinator._enqueue_sm(lambda: ...)` 模式在 HEAD 已不存在——`git log -S "_enqueue_sm"` 确认其随 Ti2026StateMachine→VisionState 重构(d11207a)移除,"按 3 次标定按钮入队 3 个等价 lambda"的描述对 main@92d0b49 不成立。原[高]定级随之撤销。
- **描述(现行代码)**: 按钮动作不经任何队列,由显示回调**同步直调** vm/coordinator。机制变了,缺陷未消:连点=重复触发(Phase2 可重入)、动作历史不可观测、显示线程与视觉线程经直调产生无隔离的跨线程耦合——与 SCHED-01"选型无规则"同根。
- **后果**: 同原条目实质:重复命令无法防御;排障看不到动作历史。
- **建议**: 不变,作为前瞻设计规则保留(D6):typed command(`@dataclass class Cmd: kind:str; payload:dict`)入 CmdQueue,入队处合并同 kind(如队首已是 CALIB 则跳过),主循环消费处天然获得审计日志。

### SCHED-03 [中] "最新值槽"概念有 4 种手写实现
- **位置**: `main.py:193`(`frame is not _last_seen_frame` 对象身份检查)、`pipe.last_results` 字典、coordinator `_pending_results` deque drain、`frame_serial` 新鲜度门控(vision_manager.py:396-404)
- **描述**: 四处各自发明了"存最新值 + 判断有没有新的"这一同一概念,实现细节各异(身份比较、serial 比较、drain 清空)。
- **后果**: 语义不统一(有的丢旧值有的堆积);每处都要单独理解;bug 修一处漏三处。
- **建议**: 一个 `<50 行` 的 `Slot` 类(publish 覆盖并递增 gen;load(last_gen) 无新值返回 None)统一四处;显示线程的身份黑魔法变成显式的代数号对比。

### SCHED-04 [中] 一处 UI 功能跨三个文件
- **位置**: 按钮绘制在 vision_manager(`_draw_calib_button` 等)、rect 经 getter 再暴露给 app(`get_calib_button_rect`)、命中测试在 main.py 回调里(`_in_btn` ~80 行嵌套)
- **描述**: 标定/补光/退出三个按钮的状态机分散在视觉模块(画)、HAL 边界(量尺寸)、app 层(判命中、执行动作)。
- **后果**: 改一个按钮要动三处;触摸映射链路(屏→帧坐标→按钮 rect)难以单测。
- **建议**: 命中测试(rect contains point)下沉为共享纯函数;按钮动作统一走 CmdQueue;绘制归 OverlayRenderer。

---

## 四、模块边界

### MOD-01 [高] VisionManager 上帝类(1065 行,六种职责)
- **位置**: `vision_manager.py` 全文:管道管理 + 显示合成 + UI 按钮 + capture sink 分发 + 相机故障横幅 + cm 刻度 sprite 缓存
- **描述**: app 层用 9 个 set_xxx()(【2026-08-22 复审修正】散布于 main.py:161/:266/:655-659/:672-680/:718-719;vision_manager 共定义 13 个 set_*)向它注入回调,本质是在补 HAL 缺失的能力(DISP-03 的下游症状)。
- **后果**: 任何显示/UI 改动都在碰视觉管道核心;比赛期多次性能修复都不得不在这个文件高风险操作;无法单独测试。
- **建议**:【已决策 2026-08-22】合成逻辑整体迁出模块:vision_manager 只留纯管道(init/start/loop/stop、pipeline/task、AEC PI),暴露 `set_composer(fn)` 钩子 + last_frame/last_results 两个 Slot;绘制全家(_draw_rail/_draw_overlays/_draw_*_button/sprite 缓存)迁 `app/display/compositor.py`(经 Canvas 协议);按钮状态机与命中测试迁 `app/display/ui_state.py`;main.py 的 8 个 set_xxx 注入口与 get_display_frame() 删除。目标 ~400 行纯管道。

### MOD-02 [中] DI 容器硬编码平台名特判
- **位置**: `machine.py:68`(`if platform == "linux": resolve_camera_source(...)`)
- **描述**: Machine.create 作为平台无关的装配器,却认识"linux"这个具体名字并为其做摄像头源解析。
- **后果**: 每加一个平台就要往容器里加一个 if;容器不再是纯装配。
- **建议**: 平台包自暴露能力钩子(如 `platform_mod.resolve_camera_source`),容器只负责调用存在与否(getattr 探测)。

### MOD-03 [低] cpu_affinity 默认核心映射按 AX630C 双核写死
- **位置**: `utils/cpu_affinity.py:11-15`(`{0}`小核/`{1}`大核 默认值)
- **描述**: 角色到核心的默认映射是针对 AX630C 2×A53 的拓扑;有 `platform.system()=="Linux"` 门控不会崩,但在多核 PC/RK3588 上这些数字语义失效。
- **后果**: 配置不可移植;静默绑错核对性能可能有害。
- **建议**: 无配置时禁用而非用默认映射;或映射表进 project_config.yaml 按设备提供。

---

## 五、死代码与打包冗余

### DEAD-01 [低] utils/state_machine/ 整包是 framework 版死副本,仍被打包
- **位置**: `utils/state_machine/{base,visual_state_machine,__init__}.py` vs `framework/state_machine/`;`app.yaml:210-212` 打包清单包含
- **描述**: 两处都有 State/BaseStateMachine/VisualStateMachine;app 只 import framework 版(utils/state_machine 仅自引用)。
- **后果**: 修 bug 容易改错副本;包体无谓增大。
- **建议**: 整目录删除 + app.yaml 移除。

### DEAD-02 [低] VisualStateMachine 两处定义均无人引用
- **位置**: `framework/visual_state_machine.py` 与 `utils/state_machine/visual_state_machine.py`
- **描述**: 当前 coordinator 不使用视觉追踪状态机,两份定义都是孤儿。
- **后果**: 维护噪音;"到底哪份是真的"需要考古。
- **建议**: 若下届不用则删;要用则只保留 framework 版并在 coordinator 接线。

### DEAD-03 [低] camera_hub re-export shim
- **位置**: `modules/zw_opencv_module/camera_hub.py:1-3`(仅 `from framework.hal.camera_hub import CameraHub`)
- **描述**: 历史迁移留下的兼容转发。
- **建议**: 更新引用方后删除。

### DEAD-04 [低] 旧赛题遗留未被引用:mission_state_machine(771 行)、line_follow_sm
- **位置**: `app/mission_state_machine.py`、`app/line_follow_sm.py`(main.py 已不 import)
- **描述**: 物料块赛题的 34 状态机与循迹 SM 仍在仓库,AGENTS.md 已标注遗留,但代码本体还在且 mission 相关文档仍指向它们。
- **后果**: grep 命中的干扰源;新人误读为现行逻辑。
- **建议**: 移入 `archive/` 目录或打 tag 后删除;app.yaml 相应清理。

### DEAD-05 [中] app.yaml exclude 与 files 双向自相矛盾(~50 行死条目)
- **位置**: `app.yaml` — exclude 第 17 行 `requirements*` vs files 第 197/227/228 行又列出三个 requirements 文件;exclude 第 18-24 行排除 cargo/circle/ring 检测器与四个处理器 vs files 第 89-139/148-165/181/184/186 行逐一列出同一批文件
- **描述**: 打包结果完全取决于 maixpack 对 exclude/files 的优先级(当前 exclude 生效)。files 里近 50 行是被 exclude 否决的死条目。
- **后果**: 强烈误导维护者(以为在打包);工具升级改变优先级时会突然打进一堆废弃代码。
- **建议**: 以 exclude 为准重写 files 清单,删掉全部死条目。

### DEAD-06 [低] 游离脚本与 debug 目录打包进设备包
- **位置**: 根目录 `capture_exposure_set.py`(游离 maix 脚本,不属于任何结构);app.yaml files 包含 modules/zw_opencv_module/debug 与 detectors/*/debug 全部调试窗口
- **描述**: 设备上运行不到的调试 UI 与一次性脚本进入发布包。
- **后果**: 包体膨胀;设备端存在不可达代码面。
- **建议**: capture_exposure_set.py 移入 archive 或 tools/;debug 目录用通配符 exclude。

### DEAD-07 [低] dist/ zip 版本号与 app.yaml version 脱节
- **位置**: `dist/` 最高 v2.0.2 vs `app.yaml:4` version 1.9.16
- **描述**: 发布产物命名与清单版本字段不同步,无法从产物名推断对应源码版本。
- **建议**: 发版脚本统一从 app.yaml 读版本号。

---

## 六、配置漂移

### CFG-01 [中] UART 口三个版本并存
- **位置**: `machine.py:40,50` 默认 `/dev/ttyS1`@921600 / `project_config.yaml:37` 实际 `/dev/ttyS4`@921600 / `PROJECT_CONTEXT.md` 写 ttyS3@115200
- **描述**: 同一硬件事实有三个互相矛盾的记载。yaml 是真值,其余两处是历史残留。
- **后果**: 按 PROJECT_CONTEXT 或 machine 默认值接线的 MSPM0 将收不到任何数据。
- **建议**: yaml 单一事实源;machine.py 缺配置时 warning 而非静默默认;PROJECT_CONTEXT 降级为历史文档并在头部标注。

### CFG-02 [中] 摄像头分辨率三个版本
- **位置**: `machine.py` 默认 640×480 / yaml 实际 1280×352@60fps / PROJECT_CONTEXT 写 640×640
- **描述**: 同 CFG-01,三处记载不一致。**注意:1280×352 本身是刻意设计【2026-08-22 确认】,不是待修正项**——摆杆轨道为横向长条,纵向裁剪至 352 将 NPU 输入面积压至常规方形约 27%(省算力),同时保留 1280 横向像素密度(pixels_per_cm≈51 的测量精度依赖于此),模型 steelball_1280x352.mud 亦按此输入尺寸训练导出。分辨率属设备/任务级配置,归 yaml 管。
- **后果**: 按 PROJECT_CONTEXT(640×640)或 machine 默认(640×480)接线/排查会得到错误结论;"好心人"把它改成标准分辨率会同时破坏精度与算力预算。
- **建议**: 同 CFG-01(yaml 单一事实源);并在 AGENTS.md 或 yaml 注释中记录 1280×352 的设计理由,防止后续被当作异常值"修正"。

### CFG-03 [中] PROJECT_CONTEXT.md 大面积滞后
- **位置**: 全文——仍写 LineFollowCoordinator、协议 v2.1 心跳、34 状态机为现行、仓库内 MaixPy-main 目录(实际在外部 F:\MaixPy-main)
- **描述**: 该文件曾定位为"Agent 首先加载的完整上下文",但 v3.0 主从改版后未同步,现在一半内容误导读者。
- **建议**: 头部加"已过时,以 project_config.yaml 与 AGENTS.md 为准"声明,或直接归档。

### CFG-04 [低] 相机参数透传通道:扩展点合法,但转发方式欠稳且无文档
- **位置**: `machine.py:80-82`(无条件向 create_camera 转发 focal_length_mm/sensor_width_mm/sensor_height_mm 等,yaml 中暂未配置)
- **描述**:【2026-08-22 修正定性】该透传是**框架扩展点而非死代码,保留**(判据:扩展点="有通道待使用者",死代码="有实现无入口";框架应为未出现的需求预留接口)。剩余问题有二:①转发是无条件的——键缺失也传 None,要求每个平台工厂签名认识全部键,构成跨平台隐性耦合;②通道无任何文档,存在感为零等于不存在。
- **建议**: 改为只转发 yaml 中实际出现的键(平台工厂只需实现自己关心的参数);在 yaml 注释或 AGENTS.md 记录支持的相机键及平台支持矩阵(如 `aec` 仅 maixcam2 实现);约定平台对未知键的行为(忽略+warning 优于 TypeError)。

---

## 七、UART 协议正确性

### PROTO-01 [高→已解决] PENDULUM 数据流布局:代码、注册表、协议文档三方矛盾
- **位置**: 代码 `coordinator.py:694-697`(sub_payload= pe_x(2B)+ball_cm(2B)+**硬编码 `[0x01,0x00]`**+速度v(2B)=8字节;flags 恒 0x01,"未锁定整帧不发");注册表 `protocol.py:68`(`DATA_PENDULUM_POSITION: 10` 总长);文档 `protocol_pendulum.md:84-95`(sub_payload=6字节,flags 要求动态 bit0=TARGET_FOUND 且丢帧时仍发 flags=0x00 帧);ACK 尺寸三方矛盾(`master_slave_protocol.md:62,:124` payload_size=0 vs 其它处含 header)
- **描述**: 同一帧格式存在三个不一致的定义。代码用硬编码常量顶替了协议要求的动态 flags 语义;文档之间对 ACK 的 payload_size 口径(仅 sub_payload vs 含 header)也互相打架。
- **后果**: MSPM0 若按文档实现则**每帧长度/CRC 校验必然失败**——主控链路不通或靠双方巧合调通;文档承诺的"flags.bit0==0 保持上次值"策略完全失效;下届复用协议时是确定性地雷。
- **建议**:【已决策+已执行 2026-08-22】**以源码为准修正文档**(protocol_pendulum.md 已升 v1.1:sub_payload 6B→8B 增 ball_vel 字段、注册表 8→10、ACK 示例修正、flags 语义改为恒 0x01 + 接收侧按帧超时判丢失、§7 配置更新至 1280×352/ppc=51.0)。注:本协议为**每赛题重设物**,下届以 protocol.py 为起点重新定义即可,框架层复用的只有帧结构(SOF+CRC16)、FrameParser 与 build/parse 工具。遗留改进(可选):`[0x01,0x00]` 硬编码换语义常量。

### PROTO-02 [中→已解决] CMD_ACK.payload_size 语义在 master_slave_protocol.md 内部自相矛盾
- **位置**: `master_slave_protocol.md:570-572`("reports sub_payload size only, total = 2 + payload_size")vs 同文 :408/:432/:476("payload = 8 字节,CMD_ACK.payload_size = 8" 即含 header);`protocol_content.md:63` 也定义含 header;代码取含 header 口径(`coordinator.py:355-358`)
- **描述**: 文档两处口径差 2 字节,对 LINE/TARGET/DET_STAT/PENDULUM 全部定长类型生效。
- **后果**: MSPM0 按 :571 实现将接收缓冲算大 2 字节,逐帧解析错位。
- **建议**: 全文档统一为代码口径(含 header),修订 :570-572。

### PROTO-03 [中] FrameParser Length 字节被噪声污染时吞掉后续多个好帧,无重同步手段
- **位置**: `uart_driver.py:130-137`(GOT_LEN 态按 `3+_expected_length` 盲收至满才校验)、`:139-141,147-157`(CRC 失败整缓冲丢弃 reset,不逐字节回退重扫,也无字节间超时中止半帧)
- **描述**: Length 字节本身不受 CRC 保护。线路噪声把它改大(如 20→200)时,其后最多 197 字节内的完整合法帧会被当作当前帧的 body 吞掉,校验失败后又是整段丢弃。921600bps 连续指令流下单点误码可连锁丢弃多条 CMD_REQUEST/CMD_STOP。
- **后果**: 偶发性订阅响应丢失,现场表现为"主机发了请求但从机没反应",复现困难。
- **建议**: CRC 失败后从 SOF 后第 2 字节起重扫描(经典 HDLC 式逐字节回退);或增加帧间超时强制中止半帧。

### PROTO-04 [低] Length 上界校验是恒真条件
- **位置**: `uart_driver.py:120`(`if 3 <= byte <= 255` —— bytes 迭代恒 0-255)
- **描述**: 校验实际只拒绝 0/1/2,制造了"已做上界防护"的假象,与 MAX_PAYLOAD_SIZE=252 的注释呼应失效。
- **后果**: 无直接危害(CRC 兜底),纯误导性代码。
- **建议**: 明确写成 `if 3 <= byte <= MAX_PAYLOAD_SIZE`。

### PROTO-05 [低] 订阅参数 min_interval_ms 被完全忽略
- **位置**: `events.py:17-19`(事件携带该字段)、`coordinator.py:_on_cmd_request`(未使用)、loop 每 tick 无速率限制发送、ACK 恒报 freq=60(`:358,:364`)
- **描述**: 协议允许主机请求低速流,实现无条件 60fps 全速发。
- **后果**: 协议承诺与实现脱节;带宽与主循环时间被无条件占用。
- **建议**: 订阅激活时记录 interval,发送侧按墙钟门控;ACK 如实回报实际频率。

### PROTO-06 [低] CALIB 期间"成功 ACK 却静默无数据"且 ACK 重复构建
- **位置**: `coordinator.py:352-354`(先置 `_streaming_type`)、`:360-362`(CALIB 态发 ACK 后 return,跳过 change_state(STREAMING))、`:361 与 :364`(构建完全相同的 ACK 两次)
- **描述**: 标定时主机拿到成功 ACK 却收不到任何数据,也没有 NACK 解释;靠 loop() 的状态门静默不发。
- **后果**: 主机侧无法区分"标定中"与"从机死了";重复构建属维护噪音。
- **建议**: CALIB 态改发 NACK(BUSY 类理由码)或延后 ACK 到标定完成;删重复行。

### PROTO-07 [低] SEG_MASK 构建器无符号 to_bytes,负坐标将 OverflowError
- **位置**: `coordinator.py:526-527`(`int(s["center_x"]).to_bytes(2,'little')` 缺 signed=True)
- **描述**: 防御性缺陷。当前 plate_seg 模型未注册,该请求会走 NACK NOT_READY(:343-350),故暂不可达;一旦启用分割订阅且中心点为负(图像边缘外推),异常被 module_manager 吞掉、该 tick 流静默中断。
- **建议**: 加 signed=True 或先 clamp 到 [0, 65535]。

---

## 八、并发与线程安全

### CONC-01 [高] ai.switch() 在 UART 回调线程调用,与推理线程无锁竞争
- **位置**: 链路 `uart_driver.py:331-337`(RX 回调同步 publish)→ `event_bus.py:30-32`(订阅者在发布线程内联执行)→ `coordinator.py:346`(`self._ai.switch(nick)`);`maixcam2/ai.py:78-108` switch **无锁**,先 unload(`_model=None`)再从 flash 重载,文件头注释自述耗时 200-500ms(:75-77);同时视觉管线线程持 `machine.ai` 逐帧推理
- **描述**: 主机切换订阅类型的瞬间,RX 线程会把推理线程正在使用的模型置空/替换。竞态窗口确证;底层 C++ 是否耐受并发访问待板端证实。
- **后果**: 切换瞬间输出抖动/异常帧甚至段错误;RX 回调被阻塞数百 ms 致 UART 帧积压。
- **建议**: switch 移出 RX 线程(经 CmdQueue 由主循环执行);ai.py 内部加锁保护 _model 替换;切换期间对 detect() 返回"忙"而非崩溃。

### CONC-02 [中] EventBus publish 持 RLock 逐个调用订阅者,订阅者含慢操作
- **位置**: `event_bus.py:27-35`(即 `framework/event_bus.py`,publish 持 RLock 逐个内联调订阅者);订阅者 `_on_cmd_request`(coordinator.py:343-365)锁内执行 `ai.switch`(秒级)与 `_send()`(UART 写)
- **描述**: 发布期间任何线程的 publish/subscribe/history 全部阻塞。当前未构成锁环,属"长临界区+外部慢操作"结构。
- **后果**: 一次模型切换即冻结整条事件总线;未来新增订阅者若再碰锁序,死锁风险上升。
- **建议**: publish 改快照列表后锁外派发;慢操作订阅者自身改为入队(CONC-01 同解)。

### CONC-03 [中] 录制信令的 150ms sleep 在 _cmd_lock 锁内同步执行
- **位置**: `main.py:309-329`(`_send_record_cmd` 3 次 sendto 每次 sleep(0.05))+ `coordinator.py:225-241`(`_start_recording→_try_notify_pc` 持 `_cmd_lock` 调用它)
- **描述**: 录制启动/停止瞬间,主循环 coordinator.loop 在 :274 抢同一把锁被卡 ≥150ms。
- **后果**: DATA_STREAM 断流、500Hz tick 出现尖峰——恰发生在比赛演示录制开始的时刻。
- **建议**: UDP 信令异步化(专用小线程或队列),change_state 只做状态翻转。

### CONC-04 [低·潜伏] 相机 read/read_raw 共享状态无锁,跨线程 release 存在竞态
- **位置**: `camera.py:217-258`(_cam/_last_raw/_last_frame/_frame_serial 无锁读写)、`vision_manager.py:993-996`(release_pipeline 可能从其他线程触发 camera.release())
- **描述**: 当前接线为单消费者、Phase1 先于视觉线程启动(main.py:609-611),故未触发;结构上 `_cam` 置 None 与使用之间存在窗口。
- **后果**: 未来引入第二个消费者或动态释放管道时变成真实竞态。
- **建议**: release 路径加简单锁或在文档中固化"单消费者"约束。

---

## 九、控制链路逻辑

### CTRL-01 [中] 钢球 miss/hit 计数节拍不对称:丢检容限实际远小于设计值
- **位置**: `coordinator.py:659-666`(hit 只在 `is_new_frame=True` 即 AI 帧率节拍累计)vs `:640-644`(ball 为 None 分支**每个主循环 tick**都调 `_on_ball_invalid()`,无 is_new_frame 门控)
- **描述**: 主循环(~300-500Hz)远快于 AI 帧率(60fps),所以 `_BALL_DROP_FRAMES=3` 名义上是"3 个 AI 帧",实际约等于 3 个主循环 tick(几十 ms)。
- **后果**: 单次检测闪烁几乎立即 disarm 并触发滤波器整体复位(:546-549),found 位高频抖动,主机收到频繁"目标丢失";α-β 滤波反复重启失去平滑价值。
- **建议**: invalid 分支同样按 AI 帧节拍计数(用 is_new_frame 门控或记录 last_seen_frame_serial 对比)。

### CTRL-02 [中] 主机链路超时机制缺失:_last_cmd_time/_master_linked 是死字段
- **位置**: `coordinator.py:98-99`(初始化)、:302/:334/:374(三处写 `_last_cmd_time`,**全文从未读取比较**)、:335(置 True)与 :721(调试读取)——`_master_linked` 永不回退;PING/PONG 仅被动应答(`uart_driver.py:386-392`),从机不监测 PING 停发
- **描述**: v3.0 从机角色下没有对主机的活性监测。UART 断连/主机死机后,从机按原订阅无限流式发送。
- **后果**: 链路死亡不可感知,无法降级或告警;调试时 link_active 恒真误导排障。
- **建议**: loop() 中用已有的 _last_cmd_time 实现"N 秒无主机消息 → 置 _master_linked=False + EventBus 广播链路断"。

### CTRL-03 [中] pc_heartbeat 重连回调一次性失效
- **位置**: `pc_heartbeat.py:94`(`_was_connected` 只在收到心跳处置 True,超时路径从不清除,仅 stop():58 清除)、`is_connected`(:28-29)在 app 内无调用者
- **描述**: PC 断开 5 秒再重连时 `_on_connected` 不再触发,coordinator.py:201-207 的"PC reconnected, retry notify"录制通知重试逻辑首次断连后永久失效。
- **后果**: 第二次连接的 PC 收不到自动通知,需手动干预——比赛换电脑场景会踩。
- **建议**: 超时路径复位 `_was_connected=False`(带迟滞防抖)。

---

## 十、视觉管道与相机层

> 注【2026-08-22 复审】:下文裸写的 `camera.py` 均指 `framework/hal/platforms/maixcam2/camera.py`(平台实现);HAL 接口在 `framework/hal/interface/camera.py`。

### VIS-01 [高] 相机重连/停滞期间陈旧帧进入 AI 控制路径,新鲜度门控未覆盖
- **位置**: `pipeline_camera.py:96-104`(read_raw 返回 None 时回退 `self._last_frame` 且照常执行任务、无新鲜度标记)、`camera.py:237-240`(重连期 `_cam=None` 返回 None 但 `last_raw` :183-185 仍持旧 Image)、`ai_inference_processor.py:61-64`(last_raw 非 None 就推理并报 success=True)、`camera.py:252-254`(read_raw 失败 return 缓存帧)、`vision_manager.py:363-366`(`any_fresh = frame is not None` 把旧帧计为新鲜)
- **描述**: 显示路径有 frame_serial 门控(vision_manager.py:396-404),控制路径完全没有。相机故障/重连的数秒内,AI 持续消费同一冻结画面,coordinator 收到结果即置 `_ai_new=True`(:322-325),is_new_frame 门控被击穿,α-β 把陈旧检测当新量测。
- **后果**: 故障期向 MSPM0 持续上报"看似正常实则停滞"的钢球位置;armed 不掉;FPS 统计虚高。
- **建议**: 帧携带 serial/时间戳贯穿到 VisionResult;read_raw 失败返回 None 而非缓存(新鲜度语义交调用方);处理器拒绝 serial 未变的帧。

### VIS-02 [中·待确认] image2cv(copy=False) 视图生命周期 vs buff_num=3 驱动缓冲复用
- **位置**: `camera.py:227,248-249`(_last_frame 是指向 maix Image 内部缓冲的负步长零拷贝视图;仓库注释 :68-71 自认"driver may reuse Image objects")、`pipeline_camera.py:100-102`(视图长期存入 _last_frame 跨迭代持有)
- **描述**: 若驱动环形缓冲在处理耗时超过约 2 个帧周期(@60fps≈33ms)时回卷覆写像素,推理/AEC 统计会读到撕裂帧。MaixPy 克隆缺 C++ 相机驱动源码,**能否靠持引用阻止回收无法离线证实**。
- **后果**: 若成立:偶发无规律误检、AEC 亮度统计毛刺,极难复现排查。
- **建议**: 板端验证(推理中反复比对帧内容 hash);必要时对该帧 `copy=False`→深拷贝仅在超时时启用。

### VIS-03 [中] AEC PI 输出无步进限幅,增益大幅跳变造成画面泵动
- **位置**: `vision_manager.py:687-698`(delta=kp*err+ki*I 后仅 gain_min/max 饱和钳位,无斜率限制);参数 project_config.yaml:26-32(kp=0.5,target_mean=80,interval=60 帧)
- **描述**: 最坏 |err|≈175 时单步 Δgain≈92,约 1 秒一跳,50→600 全程仅需 ~6 步;稳态在 deadband 边缘也有每秒 ±4~5 的可见波动。
- **后果**: 画面亮度周期性突跳,YOLO 输入分布突变造成检测瞬时丢失,间接打击 α-β 滤波。
- **建议**: 加 max_step_per_adjust(如 ±16);或对 delta 做 EMA。

### VIS-04 [中] 重连后硬件 gain 被静默重置,AEC PI 状态却不复位
- **位置**: `camera.py:119-140`(_maybe_reconnect 成功后 `_apply_init_params()` 把 gain 打回初值 200,:103)vs `vision_manager.py:135-137`(_aec_err_i/_aec_ema/_aec_counter 仅 __init__ 初始化)
- **描述**: 重连后第一次 AEC 调整用陈旧 EMA/积分对着已被复位到 200 的执行器计算。
- **后果**: 一次错误跳变;EMA 约 10 个采样周期(~10s)内曝光不稳定。
- **建议**: 重连完成事件回调清零 AEC 状态(或将 AEC 状态移入 camera 对象随重连一起复位)。

### VIS-05 [低] 积分器抗饱和只防饱和不防失效;deadband 内冻结但不衰减
- **位置**: `vision_manager.py:673-675`(deadband 内 return,陈旧积分跨界携带 ki×max_i≈5 灰阶偏置)、`:691-698`(条件积分只在执行器饱和时停;set_gain 返回 False(camera.py:269-277)时积分照常累积到 max_i)
- **描述**: 执行器故障时控制器持续"空转"积分,恢复后首拍过冲。
- **后果**: 低频小幅亮度失准;故障恢复瞬间的过冲。
- **建议**: set_gain 失败时冻结积分;deadband 内积分乘衰减系数(如 0.98)。

### VIS-06 [低] 初始 gain 设置失败导致软件 AEC 永久瘫痪 + 日志刷屏
- **位置**: `camera.py:97-106`(_apply_init_params 吞掉 gain 设置异常,_last_gain 保持 None)、`vision_manager.py:681-684`(每 60 帧 ≈1s 打一条 "AEC skipped: last_gain unknown",永不自愈)
- **描述**: 开局一次瞬时失败即决定整场运行 AEC 不可用,只剩固定 exposure_us=3000/gain=200。
- **后果**: 现场只能靠翻日志发现曝光从未自适应。
- **建议**: _last_gain 未知时周期性重试 _apply_init_params;日志升级为一次性 ERROR。

### VIS-07 [低] 未知 task_type 静默跳过,无任何日志
- **位置**: `pipeline_camera.py:73-76`(`except ValueError: continue`)
- **描述**: vision_config.yaml 里处理器类名拼错时该任务直接消失。
- **后果**: 设备端表现为"功能没了"而非报错。
- **建议**: 至少 log.warning 一次(启动期一次性)。

---

## 十一、标定链路

### CALIB-01 [中] 标定异常被静默吞掉,诊断信息失败路径无人消费
- **位置**: `pendulum_calibrator/__init__.py:117-121`(仅 cvtColor 有 guard)、子方法写了 `_diag['fail_reason']` 但 `main.py:431-432` 用 `except Exception: pass` 整体吞、`:713` 重试 15 次后只打一行 "Phase1 FAILED"、`get_last_diagnostics()`(:106-107)失败路径从不打印
- **描述**: 诊断基础设施做了,生产调用方没接。现场失败时无法区分光照/阈值/拟合哪一级问题。
- **后果**: 只能盲调参数。
- **建议**: 失败路径打印最后一次 diagnostics;except 至少 log_print 异常摘要。

### CALIB-02 [中] Phase1 新角度直接覆盖持久化标定的 angle,无一致性防护
- **位置**: 合并逻辑 `main.py:692-707`(保留持久化 origin_x/y、替换为本次 angle_rad,无 |Δangle| 校验、origin 不对新轴重新投影);坏值来源如 `__init__.py:454-516` Hough 平均角可能是外值
- **描述**: 一次坏拟合会静默污染原本正确的持久化标定(origin 是旧轴上的点,angle 是新轴)。
- **后果**: 投影 pe_x 比例/偏移失真,控制精度劣化且无告警。
- **建议**: |Δangle| 超阈值(如 10°)时拒绝合并并告警;接受时对 origin 做轴变换重投影。

### CALIB-03 [低] 列中心线阈值降级链构造缺陷 + 降级不可观测
- **位置**: `__init__.py:197-200`(150/120 仅在 `< base_threshold` 时入链——配置 column_threshold≤150 则降级链为空,与 project_config.yaml:57 注释"失败自动降级 150->120"矛盾)、`:222`(实际用到的阈值记在 diag,但 main.py:424-427 成功日志只打 method/pts/angle)
- **描述**: 180→150→120 的降级在现场完全不可见,误以为始终用首选阈值。
- **建议**: 成功日志附 threshold;降级链改为固定 [base, 150, 120] 去重。

---

## 十二、日志系统

### LOG-01 [高] 主循环异常日志无限速:任一回调持续异常即以 ~500Hz 同步写盘
- **位置**: `module_manager.py:79-103,111-114`(loop/tick/display_callback 异常仅 coordinator.loop 有 1s 限速,其余 logger.exception 无限速)+ `main.py:19-24`(根 logger basicConfig 挂无轮转同步 FileHandler);外层 `traceback.print_exc()` 直写 stderr 还会打花 Rich TUI 全屏
- **描述**: 任一模块持续性抛异常时,主循环线程逐条格式化 traceback 并写 SD 卡,喂狗间隔随之拉长。
- **后果**: 一个次要 bug 可拖垮主循环节拍并写满存储;TUI 花屏掩盖真正信息。
- **建议**: 所有 per-tick 异常日志套 1s 限速器(复用 coordinator 的模式);stderr 输出经 ConsoleCapture 进队列。

### LOG-02 [中] 全部主日志路径无轮转,嵌入式 flash 可被写满
- **位置**: `log_util.py:12`(_LOG_FILE=logs/debug.log 追加写无上限)、`main.py:19-24`(app.log 裸 FileHandler)、LoggerFactory 的 RotatingFileHandler(log_util.py:172-177)两条主路径都没用;另 logs/performance.log 被 git 跟踪(git ls-files 实证;app.log 本身未跟踪),且 logs/app.log+performance.log 双双打进发布包(app.yaml:70-71)
- **描述**: 长时间联调日志无限增长,MaixCAM2 flash 分区写满后系统级异常。
- **后果**: 赛前联调一整天就可能触达;发布包还带着历史日志。
- **建议**: 两条主路径切 RotatingFileHandler(maxBytes+backupCount);logs/ 出包排除。

### LOG-03 [中] 过载时丢包警告永不打印,静默无限丢弃
- **位置**: `log_util.py:57-76`(警告条件含 `_LOG_QUEUE.empty()`,持续过载时队列永不清空→永不满足)
- **描述**: writer 吞吐 < 生产速率时恰是最需要诊断的时刻,却零感知地无限丢日志;console_capture 把所有子线程 print 送同一队列,视觉线程日志最先被丢。
- **后果**: 现场日志"看起来正常"实则大量缺失。
- **建议**: 丢包计数达到阈值时由 writer 线程直接写 stderr(不经队列);定期(如每 30s)汇报累计丢包数。

### LOG-04 [中] writer 对每行日志单独 open/append/close,吞吐天花板低
- **位置**: `log_util.py:34-39,63-67`;debug.log 另无轮转;磁盘写满后 open 异常被逐行吞→文件日志永久静默失效(队列继续消费,看似正常)
- **描述**: flash 上单次 open/close 毫秒级,高速率下自我制造队列满;失败后无任何降级信号。
- **建议**: 持久句柄 + 定期 flush;写失败计数并周期告警。

### LOG-05 [低] fw_log 全部 INFO 日志被根级别 WARNING 过滤
- **位置**: `framework/log.py:3-9`(fw_log 走 logger.info)+ `main.py:19`(basicConfig level=WARNING)
- **描述**: framework 层"已记录"的契约形同虚设,包括 StateMachine 回调错误(base.py:341 等)与 cpu_affinity 回显。
- **建议**: fw_log 用独立 logger 并设级别;或框架关键路径改 warning。

### LOG-06 [低] console_capture 打开的 logs/debug.log 死句柄
- **位置**: `console_capture.py:14-15`(打开后从未写入,与异步 writer 目标同路径)
- **描述**: 轻微资源浪费兼混淆(两个写入者目标相同,一个永远空)。
- **建议**: 删除该句柄。

---

## 十三、状态机引擎(注:当前仅 legacy 代码引用,属潜伏缺陷)

### SM-01 [中] RLock 可重入允许回调内嵌套 trigger,产生乱序状态通知
- **位置**: `base.py:98,169-185,239-268`(enter/exit 回调与全局回调均在锁内执行 :343-357);若 B.on_enter 中触发 B→C,B 的 on_exit 会在 on_enter 返回前被执行,嵌套返回后外层继续用陈旧局部变量执行 enter 回调与 `_notify_callbacks(old=A,new=B)`
- **后果**: 监听者收到与最终状态矛盾的转换通知。
- **建议**: 回调移出锁外执行;或 trigger 重入检测(转换进行中断言)。

### SM-02 [中] run_to_completion 三处盲区
- **位置**: `base.py:218-237`:(a) A→B→A 振荡每轮 prev≠current,静默跑满 max_steps=10 停在中间态;(b) 返回值无法区分合法 10 步链与病态截断;(c) 事件型 trigger 完全不参与级联检查
- **建议**: 振荡检测(访问集合重复即告警);截断时 log.warning;trigger 后可选 run_to_completion。

### SM-03 [低] bridge.py 先执行进入动作后执行退出动作
- **位置**: `bridge.py:28-53`(_on_change new 在前 old 在后):ALIGN_RAW→ERROR 时先跑 ERROR.activate 后跑旧 cleanup,cleanup 可能撤销错误处置;when_enter 同名状态 dict 覆盖式赋值静默替换
- **建议**: 调换顺序;重复绑定告警。

---

## 十四、生命周期与主循环

### LIFE-01 [中] display 线程永不 join,关闭后仍轮询已释放资源
- **位置**: `main.py:184-209,729`(while True daemon,句柄未存、无停止机制)、`module_manager.py:127-128`(stop_all→machine.close 之后该线程仍以 1kHz 轮询已关闭 display/vm,异常全吞)
- **描述**: 退出顺序上 display 线程存活到最后;若 close 因 UART flush 阻塞,窗口期内持续对已释放 HAL 调 show。
- **建议**: 线程保存引用 + stop event;stop_all 前 set 并 join(timeout)。

### LIFE-02 [低] MAIN_LOOP_DELAY 干完活再 sleep,无 deadline 补偿
- **位置**: `module_manager.py:15,108`
- **描述**: 实际周期 = 工作耗时 + 2ms,负载重时节拍单向下漂;DATA_STREAM 发送节奏挂在其上,协议宣称的频率无墙钟保障(PROTO-05 修复后此点更重要)。
- **建议**: next_deadline = start + period 的绝对时间调度,sleep(max(0, deadline-now))。

### LIFE-03 [低] 模块 load 失败与 start() False 返回均被忽略
- **位置**: `module_manager.py:33-53`(load 失败仅打一次 traceback,register_many 忽略返回值)+ `zw_uart_module/__init__.py:38-47`(start 返回 False 表示端口占用等,无人检查)
- **描述**: 启动带着缺失子系统静默继续;interface 对象存在但 send_raw 每帧静默失败,无聚合的启动失败信号。
- **建议**: register_many 收集失败列表,启动末尾汇总告警;start() False 视同 load 失败。

---

## 十五、UI 与调试控制台

### UI-01 [中] DebugConsole render/key 线程无视 set_global_enabled(False)
- **位置**: `debug_console.py:106-127,193-207`(标志只影响 set/log 写入端,render/key 两线程启动后永不检查)+ 装配顺序 run.py:62-64 先 start(),main() 才按配置设 False(project_config.yaml `debug_console_enabled: false` 时)
- **描述**: 生产配置下 TUI 仍以 10fps 渲染空布局耗 CPU;'q' 键仍生效;日志改裸写 stdout 会糊在 ANSI 全屏上。
- **后果**: 白白吃掉主循环 CPU 预算;误触 q 直接退出。
- **建议**: enabled=False 时不启动两个线程;或线程循环首行查标志自退出。

### UI-02 [中] 'q' 退出路径直接 os._exit(0):不关 WDT、丢日志队列、不释资源
- **位置**: `debug_console.py:201-206` vs 正确范式 main.py:141-148(退出按钮先 wdt_feed.disable() 再退)
- **描述**: TUI 退出后硬件看门狗仍在倒数,进程死后板子 ~10s 内被复位重启;log 队列最多 2048 条未落盘;uart/machine 不释放。
- **后果**: "退出后板子自己重启了"的诡异现场;调试日志尾部丢失。
- **建议**: 统一退出函数:disable wdt → flush 日志 → stop manager → exit;所有退出源(UI-02/DISP-04)汇入它。

---

## 十六、运维·安全·仓库

### OPS-01 [高] WiFi 凭据已进入公网 GitHub 仓库历史
- **位置**: remote 为 `https://github.com/ju1c3rSH/Zulu-Walker.git`;提交 3794145("feat: WiFi AP mode")在 origin/main 可达历史中,含 `project_config.yaml:69` `ap_password: "88888888"`;第二凭据 `docs/competition/streaming_plan.md:69` `"comp2026"`;`main.py:285` 还有代码级兜底默认值
- **描述**: 密码不在私有历史而在公网 main 可达历史,删文件无效,需历史重写。
- **后果**: 竞赛局域网实际风险有限,但凭据复用到其他环境即为真实暴露。
- **建议**: 若仓库将继续公开:改密码+历史重写(filter-repo)或转私有;至少确保下届不复用此凭据。

### OPS-02 [低] README 占位符与 docs 索引未抽查完毕
- **位置**: 根 `README.md` 内容为"没有";docs/README.md 索引有效性因审查范围截断未能全覆盖确认
- **建议**: 下届开工前补最小 README(是什么/怎么跑/指向 AGENTS.md);抽查 docs 索引链接。

---

## 十七、分支与工程护栏

### BRANCH-01 [中] 20+ 分支拓扑待归档
- **位置**: ElectricCompetition2026-Spacial(main 祖先,落后57)、GongChuang2026-Spacial(三分层发源地,价值已并入)、LoongXiang-Spacial(**检测器为另一套 models/detectors/debug 架构,与 main 不同源,只能当参考**)、merge-spacial-arch(落后373 一次性移植)、feat/generic-framework 等
- **建议**: 已并入内容的三支打 tag 后删除;LoongXiang 保留但改名 `archive/detector-alt-arch` 标明性质。

### BRANCH-02 [中] generic-framework 愿景(零硬依赖/Mock 默认)与 main 背离,无决策记录
- **描述**: 比赛冲刺把平台耦合加了回去,两条线(通用框架 vs H题交付)没有合流决策文档。复盘结论:目标本来就不同,**不要试图 merge**,应以 generic-framework 为骨架、按本清单 ARCH/DISP/SCHED 项把 main 功能重新装进去。
- **建议**: 把该分支的 docs/architecture/*.md 抢救到主线 docs/,然后归档分支。

### BRANCH-03 [中] 无 CI/lint/test:一切架构规则只能人肉维持
- **描述**: 本清单多项(ARCH grep 判据、DISP 类型契约、PROTO 尺寸一致性)都可自动化检查却没有载体。
- **建议**: 最小起步:pre-commit 跑两条——①`grep -rnE "import (maix|cv2|serial)" framework/ | grep -v hal/platforms/` 必须为空;②py_compile 全仓。逐步补 pytest(协议 build/parse 往返、Slot/CmdQueue 单测最划算)。

---

## 附:修复路线图(依赖排序)

```
P0 热修(与重构无关,单独出):
  PROTO-01/02 协议文档对齐 · VIS-01 陈旧帧门控 · CTRL-01 计数节拍
  main.py:453 yaml.dump 直写非原子,改 tempfile+os.replace · OPS-01 凭据处置
P1 减重:      DEAD-01..07(纯删除/清单修正,零风险)
P2 定契约:    DISP-03 FrameSink/InputSource + SCHED Slot/CmdQueue 原语 + 选型表成文
P3 收编平台:  ARCH-01..07(Touch/Wdt/SysInfo Protocol、PLATFORMS 元数据、exit_check 注入),
              fitness function 进 pre-commit(BRANCH-03)
P4 拆上帝类:  MOD-01 → 合成迁 app/display(compositor+ui_state,经 Canvas 双后端),
              sinks 多播 + 主循环 flush 泵,删除 _display_loop(依赖 P2/P3)
P5 收敛治理:  CFG-01..04 单一事实源、BRANCH-01/02 归档、LOG/CONC/VIS 中低项批量清
```

交叉依赖提示:P4 依赖 P2(Sink 协议)与 P3(平台能力收编后 vision_manager 才能瘦身);PROTO-01 必须在任何协议复用之前解决;VIS-02 需板端实验,安排在下次上电窗口。

---

## 附二:设计决策记录(2026-08-22 复盘定案)

以下为本次复盘期间拍板的架构决策,后续重构以此为准:

| # | 决策 | 理由 | 关联条目 |
|---|------|------|----------|
| D1 | 像素容器 = **平台原生类型**,不做 canonical cv2 | maix.Image 直通硬件 VO 平面/硬 JPEG 编码;cv2image 转换是全帧拷贝(实测 10-16ms 级) | DISP-01 |
| D2 | 跨平台绘图经 **Canvas 协议**(7 个冻结原语),双后端 MaixCanvas/CvCanvas | 原语集小使双后端维护成本有上界;金帧测试防漂移 | DISP-02 |
| D3 | 合成走**钩子式** `set_composer(fn)`,在视觉线程内执行,**不新增线程** | 双核 CPU 多线程=GIL 争用+上下文切换开销;框架轻量化目标 | MOD-01 |
| D4 | Sink 为**主循环 flush 泵式**,不自持线程;`_display_loop` 删除 | 稳态常驻线程 6→4(main/vision/log writer/beacon 并入 tick);imshow+waitKey 在主线程恰为 HighGUI 推荐姿势;RK3588 与 MaixCAM2 线程拓扑完全一致 | DISP-03/04/05 |
| D5 | 平台特化内容三分:能力实现→`hal/platforms/`,可复用装配→框架/模块模板,本届编排决策→`app/`(含 app/display/compositor + ui_state) | app/ 是"这一届"的代码;下届还要用的东西放 app 会重抄一遍 | DISP-02/SCHED-04/MOD-01 |
| D6 | 调度三原语:**Slot**(最新值)/ **CmdQueue**(typed 命令)/ EventBus(稀疏广播);lambda 队列废除(注:该机制在 HEAD 已随 d11207a 重构消失,见 SCHED-02 复审说明,此条固化为下届规则) | 选型规则成文化于 thread_tick_topology.md | SCHED-01/02/03 |

配套约束(写进 AGENTS.md 的候选):稳态线程预算 ≤4;fitness function `grep -rnE "import (maix|cv2|serial)" framework/ modules/` 输出必须为零(app/ 豁免)。
