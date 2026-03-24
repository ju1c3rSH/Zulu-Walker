import logging
import logging.handlers
import json
import os
from pathlib import Path
import sys
from typing import Optional, Dict, Any
from datetime import datetime


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