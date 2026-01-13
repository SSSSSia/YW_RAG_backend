"""
日志配置模块
"""
import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(log_level: str = "INFO", log_file: Optional[str] = None):
    """配置日志"""
    # 创建日志格式
    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 配置根日志记录器
    logger = logging.getLogger()
    logger.setLevel(getattr(logging, log_level.upper()))

    # 清除现有处理器
    logger.handlers.clear()

    # 控制台处理器
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    # 文件处理器（如果指定）
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        logger.add_handler(file_handler)

    return logger

def log_step(step_num: int, total_steps: int, description: str, grag_id: str = None):
    """记录步骤信息"""
    prefix = f"[{grag_id}]" if grag_id else ""
    logger.info(f"{prefix} 📍 步骤 {step_num}/{total_steps}: {description}")

# 默认日志配置
logger = setup_logging()