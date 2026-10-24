# -*- coding: utf-8 -*-
"""显示前台（Presenter）重构方案 — v2（已按对抗性审查修订）

背景与实测依据
--------------
MaixCAM-Pro（CVI cv181x）同一块屏、同一个 ``maix.display.Display.show()``：

===========  ==============  ==============================
路径          单次 show()     帧格式
===========  ==============  ==============================
MAIN 页       2.3 ~ 6.2ms     FMT_RGB888（相机原生 CMM/VIPU）
调试页        18.0 ~ 20.5ms   FMT_BGR888（cv2image 出来的堆数组）
调试页+预览    50 ~ 53ms       同上，多一次 MaixVision 编码
===========  ==============  ==============================

调试页图**更小**（552x368 vs 640x480）却慢 3~8 倍。原因是格式白名单：
MaixCAM1 的 ``Display::show()`` 只接受 FMT_RGB888 / YVU420SP / YUV420SP /
BGRA8888 / GRAYSCALE，其它格式先在 show() 内部走一次 ``to_format(FMT_RGB888)``
（全帧分配 + 逐像素通道交换），结果落在 CPU 堆而非 CMM/VIPU，VO 再拷一次。
``maix.image.cv2image(bgr, bgr=True)`` 产出的正是 **FMT_BGR888**。

已确认（本地文档可查）
----------------------
* ``show()`` 调用即向 MaixVision 发图并自动压缩（zh/vision/display.md:92,
  zh/basic/maixvision.md:31,41）。无 per-call 关闭开关，唯一停法在客户端侧
  暂停按钮（maixvision.md:43），FAQ 给的应用层绕法是"关预览或切 png 流"
  （zh/faq.md:250-253）。
* ``Image.resize()`` / ``to_format()`` **每次返回新对象**
  （zh/vision/image_ops.md:211, 340）——所以"复用画布"不能靠 resize。

未确认假设（标注，勿当事实）
----------------------------
* 构造期 ``MAIXVISION_IMG_FMT=0`` 可关闭 show() 内的编码，以及 show() 首句是
  ``if(img_trans) img_trans->send_image(img)`` —— 来自 MaixCDK 源码阅读，
  本地无该仓库（``F:\\MaixCDK`` 不存在），**需真机验证**。
* 已排除的错误假设：早期注释称"show() 的 C++ 编码全程持 GIL"。本机实测
  ``cv2.imencode`` 期间后台纯 Python 线程推进 13.08M 次 vs 对照组 4.76M 次
  （2.75 倍），编码大部分时间**释放 GIL**。故"换后台线程减少 GIL 争用"不成立；
  真实收益只在"不阻塞显示节奏"与"帧所有权清晰"。

source 模型（按审查修订）
-------------------------
**事实**：lazy 转换是**无条件开启**的（pipeline_camera.py:55 →
camera.set_lazy_convert(True)）。所以 ``_last_frame`` 是 **LazyFrame**，
``_numpy`` 在被访问像素前为 None；而检测器读的是 ``camera.last_raw``
（maix_accel.py:58），**根本不碰 LazyFrame**。

因此在出货路径上：**不存在 image2cv 的 numpy**，只有 1 份 ``last_raw``。
（原方案"已是 2 份内存 1 次拷贝、省一次 image2cv"的说法是错的，作废。）

::

    camera 产出 raw（FMT_RGB888, CMM/VIPU, buff_num=2 双缓冲池）
      |
      +-- last_raw     : 原生帧，HUD/预览只写这里（现有路径，零新增拷贝）
      +-- LazyFrame    : 仅 cv2 回退路径按需物化（不主动 materialize）

* **不新增缓冲**：display 侧继续用 ``last_raw``。
* **真实隐患**：``last_raw`` 是双缓冲池，上一帧可能仍在 show() 内，相机就把该
  槽回收重填 → 跨帧撕裂。这是"零拷贝"的正确性代价，需在 C2 中确认
  （若确认存在，兜底是显示侧一次性拷贝或双缓冲 canvas）。
* **seq 降级为诊断计数**：不用于"过期即丢弃"（那会破坏 HUD 节流时故意发布
  旧帧的既有语义，test_competition_mode.py:136-139 已钉住）。仅用于替换
  main.py:657 的 ``frame is not _last_seen_frame`` 身份比较。
* **display_src 不能按 (hud or stream) 省略**：competition_mode 下
  ``_display_frame`` 仍必须每帧发布（vision_manager.py:357-359，
  test_competition_mode.py:83-87 / 155-164 钉住）。否则触摸退出失效、
  LCD 全黑。判据改为"只要有相机帧就发布"。

推流（JpegTap）
---------------
* **能力探测，不抛异常**：``try: from maix import http; Image.to_jpeg``，
  失败则禁用该 tap 并打一条醒目日志（与今天 streamer.py:58-80 行为一致）。
  构造期 raise 会被 ModuleManager 静默吞掉或导致 sys.exit(1)，两者都更差。
* **仅 maix（CAM1 + CAM2）**：``maix.http.JpegStreamer`` 与 ``to_jpeg()``
  是 MaixPy 通用文档（zh/video/jpeg_streaming.md:67,75,98,130-136），
  不限于 CAM1；linux/mock 用 NullTap。
* **客户端连接时才编码**：``attached()`` 为假 → service 不调 offer()。
* **按 fps 时间节流**（不是帧计数），默认 5Hz / max_width 320。
* **编码前降采样**：用原生 ``Image.resize()``（C++ 内完成）。
* **一次编码广播**不可变 JPEG；deque(maxlen=1) 丢旧。
* **MaixVision 不走 JpegTap**：与 show() 内部编码互斥。产线关闭预览。

配置（additive，不破坏 perf_log）
----------------------------------
.. code-block:: yaml

    observability:
      perf: false          # 总开关: [PERF] 行
      perf_detail: false   # 细粒度分项

    display:
      enabled: true
      interval_ms: 100

    stream:
      enabled: true
      fps: 5
      max_width: 320
      quality: 80

    maixvision:
      enabled: false       # 产线关闭；开发时开

* ``observability`` 与既有 ``perf_log:`` **并存**（别名），保留
  ``set_perf_log()/_PERF_LOG_NAMES`` 契约，否则 test_perf_log_gate.py 5 例失败
  且 DATA_STREAM/WDT/MEM 刷屏回归。
* 前缀匹配**必须带尾空格** ``"[PERF] "``：``startswith("[PERF")`` 会误吞
  ``[PERFLOG]`` 配置行（已验证）。

提交顺序（每步原子、可独立上线）
--------------------------------
C1  observability 配置化（additive）+ 修"perf 标签未进静音名单"的真实 bug
    （[PERF] page*/laser/tuner/rescue page 今天在 perf:false 下仍照打）
    + 5 个待测打点（image2cv / to_jpeg / Image.resize / HUD / 格式转换）
C2  types.py + DisplayCaps + CanvasPool；三个 tuner 改用原生画布
    （fmt=BGR888 -> RGB888）。**保留 Display.show()**，否则三个 tuner 与
    _ThrottledPageDisplay 立即断。
C3  PresenterService + LcdTap/NullTap + policy/metrics；先只接 MAIN 路径。
    单消费者线程 + deque(maxlen=1)；发布语义与 _display_frame 完全保持。
C4  JpegTap + 推流迁出视觉线程 + 修 3 个推流 bug（同一 commit）
    ：_capture_seq 自增位置（40 帧只推 1 帧）、原始 RGB 喂 JPEG write()、
    except: pass 静默。同时删除 set_capture_sink 与 _init_streamer 接线。
C5  合并两份 _maybe_show、删 _ThrottledPageDisplay、_frame_tag 的 dir() 遍历
    改为字段读取；重写 24 个 policy 测试为 policy 单元测试。

验收
----
* 调试页 ``[PERF] page show ... fmt=RGB888``（不再是 BGR888），show <= 6ms
* ``[PERF] stream ... enc=`` 有实测值，浏览器连续出图
* 视觉线程不再调用 capture_sink
* 282 个既有测试（实测 collect 数）全绿 + 新增 source/presenter/jpeg-tap 测试
* ``observability.perf: false`` 时无任何 [PERF] 行, 且 [PERFLOG] 仍可见

不做
----
* LVGL 式保留模式 + 脏区（瓶颈不在重绘）
* Android/Wayland 式多缓冲 + fence（单屏、驱动已持 VO 池）
* 四层可插拔管线（128MB + 每帧 Python 分配）
* 保留 show(np.ndarray) 快路径（静默 BGR 兜底就是该 bug 本身）
"""

__all__ = []

## TODO

- [x] **T1 (P0) 热重载 mtime 同 tick 失效** — 已修: 改内容签名 (size, sha1);实测 24% -> 0% — `gks_task_processor.py::_reload_laser_calib`
      / `_reload_rescue_params` / 阈值重载 用 `mtime == 上次` 判变化；同一文件系统
      tick 内二次 SAVE 会判"未变"→ 参数不生效。实测 300 次连续写 mtime 相同 187 次(62%)，
      紧凑循环 200 次失败 48 次(24%)。现场表现为"SAVE 了但参数没变"。
      建议：(st_mtime_ns, st_size) 双字段签名，或内容哈希兜底。同时把
      `test_laser_calib_hot_reload` / `test_hot_reload_on_mtime_change` 改成能稳定捕获该 bug。
