"""
通用工具函数
"""
import re
import subprocess
from typing import Tuple, Optional
from utils.logger import logger


def clean_graphrag_output(raw_text: str) -> str:
    """清理GraphRAG的原始输出"""
    ansi_escape = re.compile(r'(\x1B\[[0-9;]*m|\[[0-9;]*m)')
    text = ansi_escape.sub('', raw_text)
    text = re.sub(r'\[Data: [^\]]+\]', '', text)
    text = text.strip()
    text = re.sub(r'\n\s*\n', '\n\n', text)
    return text


def run_command_with_progress(
        command: str,
        description: str,
        grag_id: str = None
) -> Tuple[bool, str, str]:
    """执行命令并显示简洁的进度信息"""
    prefix = f"[{grag_id}]" if grag_id else ""
    logger.info(f"{prefix} 🚀 开始: {description}")
    logger.info(f"{prefix} 💻 命令: {command}")

    try:
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            encoding='utf-8',
            errors='replace',
            bufsize=1,
            universal_newlines=True
        )

        stdout_lines = []
        stderr_lines = []

        important_patterns = [
            r'Loading', r'Processing', r'Creating', r'Building', r'Indexing',
            r'Complete', r'Success', r'Error', r'Warning', r'progress',
            r'⠋|⠙|⠹|⠸|⠼|⠴|⠦|⠧|⠇|⠏', r'\d+%', r'Extracting',
            r'Embedding', r'Graph',
        ]

        pattern = re.compile('|'.join(important_patterns), re.IGNORECASE)

        while True:
            stdout_line = process.stdout.readline()
            if stdout_line:
                stdout_lines.append(stdout_line)
                clean_line = stdout_line.strip()
                if clean_line and pattern.search(clean_line):
                    clean_line = re.sub(r'\x1B\[[0-9;]*m', '', clean_line)
                    logger.info(f"{prefix} 📝 {clean_line}")

            stderr_line = process.stderr.readline()
            if stderr_line:
                stderr_lines.append(stderr_line)
                clean_line = stderr_line.strip()
                if clean_line and pattern.search(clean_line):
                    clean_line = re.sub(r'\x1B\[[0-9;]*m', '', clean_line)
                    logger.warning(f"{prefix} ⚠️ {clean_line}")

            if stdout_line == '' and stderr_line == '' and process.poll() is not None:
                break

        returncode = process.wait()
        stdout = ''.join(stdout_lines)
        stderr = ''.join(stderr_lines)

        if returncode == 0:
            logger.info(f"{prefix} ✅ 完成: {description}")
            return True, stdout, stderr
        else:
            logger.error(f"{prefix} ❌ 失败: {description} (返回码: {returncode})")
            return False, stdout, stderr

    except Exception as e:
        logger.error(f"{prefix} ❌ 异常: {description} - {str(e)}")
        return False, "", str(e)
