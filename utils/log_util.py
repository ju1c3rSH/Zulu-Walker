import logging
import logging.handlers
import json
import os
from pathlib import Path
import queue
import sys
import threading
from typing import Optional, Dict, Any
from datetime import datetime

_LOG_FILE = os.path.join(os.path.dirname(os.path.dirname(__file__)), "logs", "debug.log")
_LOG_QUEUE: queue.Queue = queue.Queue(maxsize=2048)
_WRITER_THREAD: Optional[threading.Thread] = None
_SENTINEL: object = object()
_DROP_COUNT: int = 0  # queue full 丢包累计计数，GIL 保护，无需额外 Lock

_CONSOLE_MODULE = "utils.debug_console"

# ---- 高频性能日志闸门（默认全开, 保持既有行为）--------------------------------
# 为什么需要: 台架调试时 `[PERF] vision/mainloop/lcd`、`[UART TX] DATA_STREAM`、
# `/CMD_ACK`、`[FPS] pipeline`、`[WDT] ... feed`、`[MEM]` 这些行**每秒各刷一条**,
# 合起来把真正想看的信息淹掉。它们全是性能观测/心跳痕迹, 与协议语义无关。
#
# 关掉之后**只少打印, 不改变任何行为**:
#   · 推流帧照发（`_send` 不受影响）;  `log_print` 依旧非阻塞（队列照投, 只是 no-op）;
#   · 计数器照累加（`_result_seq` / `_tx_window_count` / `_wdt_count` …）。
# 唯一的副作用: 被吞掉的行不会进 logs/debug.log —— 这是刻意的（debug.log 也是刷屏源）。
#
# 用模块级 name **列表而非字符串常量**: 配置里写 `[PERF] vision` 或 `PERF vision`
# 都能命中; 且用普通 `in` 判断（不是 startswith），子串误伤面可控。
#
# 默认组 = 一次性能观测 + 链路心跳/内存痕迹。**刻意不含**协议状态变化类
# （`[UART ACTION]` 门控、`[UART RX]` 收帧、`[UART TX] CMD_NACK`、`[FPS]`）——
# 那些是低频事件, 关了就看不出"订没订上/该不该 NACK"。
_DEFAULT_PERF_NAMES = (
    "[PERF] vision",        # 视觉线程每帧分项（每秒一条）
    "[PERF] mainloop",      # 主循环分项（每秒一条）
    "[PERF] lcd show",      # 显示 show() 开销（每秒一条）
    "[UART TX] DATA_STREAM",  # 推流率心跳（每秒一条, 但高频）
    "[UART TX] CMD_ACK",    # Brain 重订阅洪泛时会被刷屏
    "[WDT] ",               # 主/协调/视觉三处喂狗痕迹（每 600 次一条）
    "[MEM] ",               # user/CMM 内存（每 900 次一条）
)

_PERF_LOG_NAMES: set = set()


def set_perf_log(enabled: bool = True, names=None) -> None:
    """开关高频性能日志行。

    Args:
        enabled: False = 吞掉 names 里的所有行; True = 全放行。
        names:   要控的行标签（子串匹配）。None/空 = 只切 `_DEFAULT_PERF_NAMES`。

    调用点: `app/main.py` 从 `project_config.yaml` 的 `perf_log:` 段读取。
    默认（enabled=True）不设过滤集合, 保持"改前逐行照打"。
    """
    global _PERF_LOG_NAMES
    if enabled:
        _PERF_LOG_NAMES = set()
        return
    selected = names if names else _DEFAULT_PERF_NAMES
    _PERF_LOG_NAMES = {str(n) for n in selected}


def perf_log_enabled() -> bool:
    """性能日志当前是否全开（供测试/自检）。"""
    return not _PERF_LOG_NAMES


def _is_perf_log(msg: str) -> bool:
    """消息是否命中当前被吞的性能日志标签。空集合 -> 恒 False（零开销）。"""
    if not _PERF_LOG_NAMES:
        return False
    for name in _PERF_LOG_NAMES:
        if name in msg:
            return True
    return False


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
    # 只在 queue 满的瞬间才触发, 所以能潜伏很久（2026-10-09 台架实测）。
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