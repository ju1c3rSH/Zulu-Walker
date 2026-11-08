import logging
import logging.handlers
import json
import math
import os
from pathlib import Path
import queue
import sys
import threading
import time
from typing import Optional, Dict, Any
from datetime import datetime

_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "debug.log")
_LOG_QUEUE: queue.Queue = queue.Queue(maxsize=2048)
_WRITER_THREAD: Optional[threading.Thread] = None
_SENTINEL: object = object()
_DROP_COUNT: int = 0  # queue full 丢包累计计数，GIL 保护，无需额外 Lock

_CONSOLE_MODULE = "utils.debug_console"

# ---- 高频性能日志闸门（默认全开, 保持既有行为）--------------------------------
# 台架调试时 `[PERF] vision/mainloop/lcd`、`[UART TX] DATA_STREAM`、`[FPS]`、
# `[WDT] ... feed`、`[MEM]` 这些行按各自节拍反复刷, 把真正想看的信息淹掉。
# 关掉之后**只少打印, 不改变任何行为**（推流帧照发, 计数器照累加）; 唯一的
# 副作用是被吞的行不进 logs/debug.log —— 这是刻意的（debug.log 也是刷屏源）。
#
# 用模块级 name **列表而非字符串常量**: 配置里写 `[PERF] vision` 或 `PERF vision`
# 都能命中; 且用普通 `in` 判断（不是 startswith），子串误伤面可控。
#
# 默认组 = 一次性能观测 + 链路心跳/内存痕迹。**刻意不含**协议状态变化类
# （`[UART ACTION]` 门控、`[UART RX]` 收帧、`[UART TX] CMD_NACK`、`[FPS]`）——
# 那些是低频事件, 关了就看不出"订没订上/该不该 NACK"。
_DEFAULT_PERF_NAMES = (
    "[PERF] vision",        # 视觉线程每帧分项（每窗口一条）
    "[PERF] mainloop",      # 主循环分项（每窗口一条）
    "[PERF] lcd show",      # 显示 show() 开销（每窗口一条）
    "[UART TX] DATA_STREAM",  # 推流率心跳（每秒一条, 但高频）
    "[UART TX] CMD_ACK",    # Brain 重订阅洪泛时会被刷屏
    "[WDT] ",               # 主/协调/视觉三处喂狗痕迹（每 600 次一条）
    "[MEM] ",               # user/CMM 内存（每 900 次一条）
)

_PERF_LOG_NAMES: set = set()

# `observability:` 两级闸门, 只管 `[PERF] ` 行; 上面的 names 路径管非 [PERF] 高频行。
# ⚠ 前缀必须带尾空格: startswith("[PERF") 会误吞 [PERFLOG] 配置行。
_PERF_TAG = "[PERF] "
_PERF_COARSE = ("[PERF] vision", "[PERF] mainloop", "[PERF] lcd show")
_PERF_GATE = True
_PERF_DETAIL = True
_PERF_GATE_SET = False      # observability.perf 是否被显式配置过
_PERF_NAMES_MUTE_ALL = False  # 默认组被整组静音 -> 未列名的 `[PERF] ` 行也吞
_PERF_DEFAULT_SET = frozenset(_DEFAULT_PERF_NAMES)

# 汇总窗口（秒）: 常驻 `[PERF]` 行与 detail 探针的统计周期。
#   10.0 = 台架/赛场默认（1s 一条太刷屏）; 1.0 = 定位"这一秒时间花在哪"时临时改回。
# 统一从这里读, 避免"vision 10s / mainloop 1s"分叉。
DEFAULT_PERF_INTERVAL_S = 10.0
_PERF_INTERVAL_S = DEFAULT_PERF_INTERVAL_S


def _refresh_mute_all() -> None:
    """默认组被整组静音 = 关掉性能日志, 未列入名单的 `[PERF] ` 行也一起吞。

    ⚠ 判据是"超集"而非"相等": 追加 `[UART ACTION]` 等名字不能让 `[PERF] page*` 漏出来。
    """
    global _PERF_NAMES_MUTE_ALL
    _PERF_NAMES_MUTE_ALL = bool(_PERF_LOG_NAMES) and _PERF_DEFAULT_SET <= _PERF_LOG_NAMES


def set_perf_log(enabled: bool = True, names=None) -> None:
    """开关高频性能日志行。

    Args:
        enabled: False = 吞掉 names 里的所有行; True = 全放行。
        names:   要控的行标签（子串匹配）。None/空 = 只切 `_DEFAULT_PERF_NAMES`。

    调用点: `app/main.py` 从 `project_config.yaml` 的 `perf_log:` 段读取。
    默认（enabled=True）不设过滤集合, 保持"改前逐行照打"。

    ⚠ 本函数**不碰** `_PERF_GATE`/`_PERF_DETAIL`: 否则会反向放行 observability 的关闭。
    """
    global _PERF_LOG_NAMES
    if enabled:
        _PERF_LOG_NAMES = set()
    else:
        _PERF_LOG_NAMES = (set(_DEFAULT_PERF_NAMES) if not names
                           else {str(n) for n in names})
    _refresh_mute_all()


def set_perf_gate(perf=None, detail=None) -> None:
    """`observability` 闸门: perf = 所有 `[PERF] ` 行, detail = 细粒度分项。

    ⚠ None = 该键未配置, 保持现值（空字典不得把闸门重置成全开）。
    """
    global _PERF_GATE, _PERF_DETAIL, _PERF_GATE_SET
    if perf is not None:
        _PERF_GATE = bool(perf)
        _PERF_GATE_SET = True
    if detail is not None:
        _PERF_DETAIL = bool(detail)


def perf_log_enabled() -> bool:
    """性能日志当前是否全开（供测试/自检）。"""
    return not _PERF_LOG_NAMES


def set_perf_interval(seconds=None) -> None:
    """设置 `[PERF]` 汇总窗口（秒）。None/非法/非有限值 = 保持现值。

    ⚠ 必须挡掉 NaN/inf:
      · NaN 能通过 `v <= 0` 之类的检查, 而 `elapsed < nan` 恒为 False
        -> 每次 add() 都打印, 直接变成日志洪水;
      · inf 则永远等不到窗口 -> `[PERF]` 全部静默, 比不配更糟（失明且无提示）。
    ⚠ bool 是 int 子类（True == 1）: 配置写成 `perf_interval_s: true` 时
      不能把它当成"1 秒"悄悄生效。
    `<= 0` 视为写错（保持现值）; 正数下限 0.5s: 再快就回到刷屏区间, 真要逐帧看
    就该用 detail 探针而不是汇总行。
    """
    global _PERF_INTERVAL_S
    if seconds is None or isinstance(seconds, bool):
        return
    try:
        v = float(seconds)
    except (TypeError, ValueError):
        return
    if not math.isfinite(v) or v <= 0:
        return
    _PERF_INTERVAL_S = max(0.5, v)


def perf_interval_s() -> float:
    """当前 `[PERF]` 汇总窗口（秒）。各发射点统一从这里读, 避免节拍分叉。"""
    return _PERF_INTERVAL_S


def perf_log_summary() -> str:
    """`perf_log` 名单状态的三态描述（供 `[PERFLOG]` 自检行）。

    ⚠ 不能用 `perf_log_enabled()` 当文案: 它判的是"名单是否为空",
      只静音 `[WDT]` 时也返回 False, 打出来会像"全静音"。
    """
    if not _PERF_LOG_NAMES:
        return "ON"
    if _PERF_NAMES_MUTE_ALL:
        return "MUTED"
    return "custom(" + ",".join(sorted(_PERF_LOG_NAMES)) + ")"


def perf_enabled() -> bool:
    """`[PERF] ` 行总开关（observability 显式配置优先, 否则由名单推导）。"""
    if _PERF_GATE_SET:
        return _PERF_GATE
    return not _PERF_NAMES_MUTE_ALL


def perf_detail_enabled() -> bool:
    """细粒度探针开关（总开关关掉时恒 False, 热路径直接跳过）。"""
    return _PERF_DETAIL and perf_enabled()


def perf_t0() -> float:
    """探针计时起点: 关闸时返回 0.0, 调用方据此跳过计时与格式化。"""
    if _PERF_DETAIL and perf_enabled():
        return time.perf_counter()
    return 0.0


def _is_coarse_perf(msg: str) -> bool:
    for name in _PERF_COARSE:
        if msg.startswith(name):
            return True
    return False


def _is_perf_log(msg: str) -> bool:
    """消息是否命中当前被吞的性能日志标签。空集合 -> 恒 False（零开销）。"""
    # ⚠ 自检/配置行描述闸门本身, 绝不能被闸门吞掉。
    # `perf_log_summary()` 会把自定义名单原文带进 "[PERFLOG] perf_log=custom([WDT] )",
    # 而 "[WDT] " 正是被静音的子串 -> 不做这个豁免就会"自我审查", 改完配置连
    # "改成了什么"都看不到。
    if msg.startswith("[PERFLOG]"):
        return False
    if msg.startswith(_PERF_TAG):
        if not perf_enabled():
            return True
        if not _PERF_DETAIL and not _is_coarse_perf(msg):
            return True
    if not _PERF_LOG_NAMES:
        return False
    for name in _PERF_LOG_NAMES:
        if name in msg:
            return True
    return False


class PerfAgg:
    """每 `perf_interval_s()` 秒一条 `[PERF] <label>` 汇总; 关闸时零格式化。"""

    __slots__ = ("_label", "_tag", "_t0", "_n", "_ms")

    def __init__(self, label: str, tag: Optional[str] = None) -> None:
        self._label = label
        self._tag = tag
        self._t0 = 0.0
        self._n = 0
        self._ms = 0.0

    def add(self, ms: float, tag: Optional[str] = None) -> None:
        if not perf_detail_enabled():
            return
        self._n += 1
        self._ms += ms
        if tag is not None:
            self._tag = tag
        now = time.monotonic()
        if self._t0 == 0.0:
            self._t0 = now
            return
        elapsed = now - self._t0
        # 直接读模块全局: add() 在视觉帧率上调用, 不为一次函数调用付开销。
        if elapsed < _PERF_INTERVAL_S:
            return
        n = self._n
        avg = self._ms / n
        extra = " " + self._tag if self._tag else ""
        log_print(f"[PERF] {self._label} {n / elapsed:.1f}Hz "
                  f"avg={avg:.2f}ms{extra}")
        self._t0 = now
        self._n = 0
        self._ms = 0.0


_AGGS: Dict[str, PerfAgg] = {}


def perf_note(label: str, ms: float, tag: Optional[str] = None) -> None:
    """探针入口: 每 `perf_interval_s()` 秒一条 `[PERF] <label>` 汇总。闸门关闭时只有一次布尔判断。"""
    if not perf_detail_enabled():
        return
    agg = _AGGS.get(label)
    if agg is None:
        agg = _AGGS[label] = PerfAgg(label)
    agg.add(ms, tag)


def debug_console_active() -> bool:
    """调试台是否**真的**在跑（模块已导入 且 _global_enabled 为真）。

    ⚠ 存在的意义是"廉价判定, 且不把 rich 拖进来": `utils.debug_console` 在模块
    顶层 `import rich`, 而 rich 是纯 Python 大包 —— CAM1 只有 128MB 用户内存,
    为了问一句"它在不在"去 import 它是纯浪费。查 `sys.modules` 是字典查找。

    调用方约定: 返回 False 时应**跳过一切 DebugConsole 调用**
    （`dc.set()`/`dc.log()` 对一个没在跑的 TUI 毫无意义, 只是白占 GIL 时间）。
    """
    mod = sys.modules.get(_CONSOLE_MODULE)
    if mod is None:
        return False
    return bool(getattr(getattr(mod, "DebugConsole", None), "_global_enabled", False))

try:
    os.makedirs(os.path.dirname(_LOG_FILE), exist_ok=True)
except Exception:
    pass


def _writer_loop() -> None:
    global _DROP_COUNT
    while True:
        item = _LOG_QUEUE.get()
        if item is _SENTINEL:
            _LOG_QUEUE.task_done()
            break

        line, msg = item

        try:
            with open(_LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
                f.flush()
        except Exception:
            pass

        try:
            mod = sys.modules.get(_CONSOLE_MODULE)
            if mod is None:
                # 调试台从未被导入（启动器按配置禁用了它）—— 直接写 stdout。
                # ⚠ 这里**不能** `from utils.debug_console import DebugConsole`:
                # 那会把 rich 整个模块拉进内存, 只为判断"要不要写到 TUI"。
                sys.__stdout__.write(line)
                sys.__stdout__.flush()
            else:
                dc = mod.DebugConsole()
                if not mod.DebugConsole._global_enabled:
                    sys.__stdout__.write(line)
                    sys.__stdout__.flush()
                else:
                    dc.log(msg)
                    if not dc._running:
                        sys.__stdout__.write(line)
                        sys.__stdout__.flush()
        except Exception:
            pass

        # 队列排空后，若发生过丢包，由 writer 线程统一输出警告。
        # 这样 log_print 调用者完全零 I/O，对 MaixCAM2 视觉管线无任何阻塞风险。
        if _DROP_COUNT > 0 and _LOG_QUEUE.empty():
            drops = _DROP_COUNT
            _DROP_COUNT = 0
            warn_ts = datetime.now().strftime("[%H:%M:%S] ")
            warn = f"{warn_ts}[WARN] log_print queue full, dropped {drops} messages\n"
            try:
                with open(_LOG_FILE, "a", encoding="utf-8") as f:
                    f.write(warn)
                    f.flush()
            except Exception:
                pass
            sys.__stdout__.write(warn)
            sys.__stdout__.flush()
            try:
                mod = sys.modules.get(_CONSOLE_MODULE)
                if mod is not None:
                    dc = mod.DebugConsole()
                    dc.log(warn.rstrip("\n"))
                    dc.set("log_drop", str(drops))
            except Exception:
                pass

        _LOG_QUEUE.task_done()


def start_log_writer() -> None:
    global _WRITER_THREAD
    if _WRITER_THREAD is not None and _WRITER_THREAD.is_alive():
        return
    _WRITER_THREAD = threading.Thread(target=_writer_loop, daemon=True, name="log-writer")
    _WRITER_THREAD.start()


def stop_log_writer(timeout: float = 2.0) -> None:
    global _WRITER_THREAD
    if _WRITER_THREAD is None:
        return
    try:
        _LOG_QUEUE.put(_SENTINEL, timeout=1.0)
    except queue.Full:
        pass
    _WRITER_THREAD.join(timeout=timeout)
    _WRITER_THREAD = None


def log_print(msg: str = "", *args, **kwargs) -> None:
    # ⚠ 必须有: 下面 except 分支要写 _DROP_COUNT。缺 global 时 Python 把它当**局部**
    # 变量, 于是"队列满"这条本应只丢日志的路径会抛 UnboundLocalError —— 而调用点
    # 遍布 coordinator 主循环（module_manager 对 coordinator.loop() **没有** try/except,
    # 只有外层兜底), 结果是主循环整轮被拖停 1s、期间一个 DATA_STREAM 都推不出去。
    global _DROP_COUNT
    ts = datetime.now().strftime("[%H:%M:%S] ")
    if args:
        parts = [str(msg)] + [str(a) for a in args]
        msg = " ".join(parts)
    else:
        msg = str(msg)
    # 高频性能日志闸门（默认空集合=False, 零开销）。必须在拼好 msg 之后、入队之前,
    # 这样连 queue 都不碰 —— 被吞的行既不进 stdout 也不进 logs/debug.log。
    if _is_perf_log(msg):
        return
    line = f"{ts}{msg}\n"
    try:
        _LOG_QUEUE.put_nowait((line, msg))
    except queue.Full:
        # 仅计数，零 I/O，不阻塞调用者。
        # MaixCAM2 等嵌入式设备上任何 I/O 阻塞都可能影响视觉管线帧率。
        _DROP_COUNT += 1


class LoggerFactory:
    """日志工厂类，支持动态配置"""
    
    _instances: Dict[str, logging.Logger] = {}
    
    @classmethod
    def get_logger(cls, 

                    name: str,
                    level: int = logging.INFO,
                    log_file: Optional[str] = None,
                    max_bytes: int = 10 * 1024 * 1024,  # 10MB
                    backup_count: int = 5,
                    console_output: bool = True,
                    format_str: Optional[str] = None) -> logging.Logger:
        """
        获取或创建logger实例
        
        Args:
            name: logger名称
            level: 日志级别
            log_file: 日志文件路径，如果提供则保存到文件
            max_bytes: 单个日志文件最大字节数
            backup_count: 保留的备份文件数量
            console_output: 是否输出到控制台
            format_str: 自定义格式字符串
        """
        if name in cls._instances:
            return cls._instances[name]
        
        logger = logging.getLogger(name)
        logger.setLevel(level)
        

        if logger.handlers:
            return logger
        
        format_str = format_str or '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        formatter = logging.Formatter(format_str)
        
        # 控制台处理器(标准输出)
        if console_output:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setLevel(level)
            console_handler.setFormatter(formatter)
            logger.addHandler(console_handler)
        

        # 文件处理器(文件输出)
        if log_file:
            # 确保日志目录存在
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.handlers.RotatingFileHandler(
                log_file, 
                maxBytes=max_bytes, 
                backupCount=backup_count,
                encoding='utf-8'
            )
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        cls._instances[name] = logger
        return logger
    
    @classmethod
    def clear_instances(cls):
        """清空所有实例（主要用于测试）"""
        cls._instances.clear()


class ConfigurableLogger:
    """可配置的Logger封装类"""
    
    def __init__(self, config: Dict[str, Any]):
        """
        初始化可配置的Logger
        
        Args:
            config: 配置字典，示例：
                {
                    'name': 'myapp',
                    'level': 'INFO',
                    'log_file': 'logs/app.log',
                    'max_bytes': 10485760,
                    'backup_count': 5,
                    'console_output': True,
                    'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                    'extra_fields': ['user_id', 'request_id']
                }
        """
        self.config = config
        self.name = config.get('name', 'default')
        self.level = getattr(logging, config.get('level', 'INFO').upper())
        
        self.logger = LoggerFactory.get_logger(
            name=self.name,
            level=self.level,
            log_file=config.get('log_file'),
            max_bytes=config.get('max_bytes', 10 * 1024 * 1024),
            backup_count=config.get('backup_count', 5),
            console_output=config.get('console_output', True),
            format_str=config.get('format')
        )
        
        self.extra_fields = config.get('extra_fields', [])
    
    def _add_extra_fields(self, extra: Optional[Dict] = None) -> Dict:
        """添加额外字段"""
        if not extra:
            extra = {}
        for field in self.extra_fields:
            if field not in extra:
                extra[field] = getattr(self, field, None)
        return extra
    
    def info(self, msg: str, extra: Optional[Dict] = None):
        extra = self._add_extra_fields(extra)
        self.logger.info(msg, extra=extra if extra else None)
    
    def debug(self, msg: str, extra: Optional[Dict] = None):
        extra = self._add_extra_fields(extra)
        self.logger.debug(msg, extra=extra if extra else None)
    
    def warning(self, msg: str, extra: Optional[Dict] = None):
        extra = self._add_extra_fields(extra)
        self.logger.warning(msg, extra=extra if extra else None)
    
    def error(self, msg: str, exc_info: bool = True, extra: Optional[Dict] = None):
        extra = self._add_extra_fields(extra)
        self.logger.error(msg, exc_info=exc_info, extra=extra if extra else None)
    
    def set_extra_field(self, key: str, value: Any):
        """设置额外的上下文字段"""
        setattr(self, key, value)


class UARTModuleLogger(ConfigurableLogger):
    """UART模块日志类，支持模块级别的日志记录"""
    
    def __init__(self, module_name: str):
        config = {
            'name': f'module_{module_name}',
            'level': 'DEBUG',
            'log_file': f'logs/modules/{module_name}.log',
            'max_bytes': 50 * 1024 * 1024,
            'backup_count': 10,
            'console_output': True,
            'extra_fields': ['module_name']
        }
        super().__init__(config)
        self.module_name = module_name
        
    def log_send(self, data: str):
        self.info(f"发送数据: {data}")
        
    def log_receive(self, data: str):
        self.info(f"接收数据: {data}")
        
    def log_info(self, message: str):
        self.info(message)
    def log_error(self, error: Exception):
        self.error(f"模块错误: {error}")        


class DataPipelineLogger(ConfigurableLogger):
    """数据管道日志类"""
    
    def __init__(self, pipeline_name: str):
        config = {
            'name': f'pipeline_{pipeline_name}',
            'level': 'DEBUG',
            'log_file': f'logs/pipelines/{pipeline_name}.log',
            'max_bytes': 100 * 1024 * 1024,
            'backup_count': 20,
            'console_output': False,  # 只保存到文件
            'extra_fields': ['batch_id', 'step_name']
        }
        super().__init__(config)
        self.pipeline_name = pipeline_name
    
    def log_step_start(self, batch_id: str, step_name: str):
        self.set_extra_field('batch_id', batch_id)
        self.set_extra_field('step_name', step_name)
        self.info(f"开始处理步骤: {step_name}")
    
    def log_step_end(self, batch_id: str, step_name: str, records_processed: int):
        self.set_extra_field('batch_id', batch_id)
        self.set_extra_field('step_name', step_name)
        self.info(f"完成处理步骤: {step_name}, 处理记录数: {records_processed}")
    
    def log_data_quality(self, batch_id: str, issues: list):
        self.set_extra_field('batch_id', batch_id)
        self.warning(f"数据质量问题: {len(issues)}个问题 - {issues[:3]}")