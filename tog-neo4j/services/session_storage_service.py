"""
会话存储服务 - 用于保存图片和操作记录
"""
import os
import json
import base64
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path
from utils.logger import logger
from core.config import settings


class SessionStorageService:
    """会话存储服务"""

    def __init__(self, base_dir: str = None):
        """
        初始化会话存储服务

        Args:
            base_dir: 基础存储目录，默认为 ./sessions
        """
        self.base_dir = Path(base_dir or "./sessions")
        self.base_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"✅ 文件存储服务初始化完成，存储目录: {self.base_dir}")

    def _get_session_dir(self, session_id: str) -> Path:
        """获取会话目录路径"""
        session_dir = self.base_dir / session_id
        session_dir.mkdir(parents=True, exist_ok=True)
        return session_dir

    def _get_data_file(self, session_id: str) -> Path:
        """获取会话数据文件路径"""
        return self._get_session_dir(session_id) / "session_data.json"

    def _get_images_dir(self, session_id: str) -> Path:
        """获取会话图片目录路径"""
        images_dir = self._get_session_dir(session_id) / "images"
        images_dir.mkdir(parents=True, exist_ok=True)
        return images_dir

    def save_image(self, session_id: str, filename: str, image_data: bytes) -> str:
        """
        保存图片到会话目录

        Args:
            session_id: 会话ID
            filename: 图片文件名
            image_data: 图片二进制数据

        Returns:
            保存后的图片完整路径
        """
        try:
            images_dir = self._get_images_dir(session_id)
            image_path = images_dir / filename

            with open(image_path, "wb") as f:
                f.write(image_data)

            logger.info(f"图片保存成功: {image_path}")
            return str(image_path)
        except Exception as e:
            logger.error(f"保存图片失败: {e}")
            raise

    def save_operation_record(self, session_id: str, filename: str, operation: str, image_path: str) -> None:
        """
        保存操作记录到会话数据文件

        Args:
            session_id: 会话ID
            filename: 图片文件名
            operation: 操作描述
            image_path: 图片路径
        """
        try:
            data_file = self._get_data_file(session_id)

            # 读取现有数据
            records = []
            if data_file.exists():
                with open(data_file, "r", encoding="utf-8") as f:
                    records = json.load(f)

            # 添加新记录
            record = {
                "filename": filename,
                "operation": operation,
                "image_path": image_path,
                "timestamp": datetime.now().isoformat()
            }
            records.append(record)

            # 保存到文件
            with open(data_file, "w", encoding="utf-8") as f:
                json.dump(records, f, ensure_ascii=False, indent=2)

            logger.info(f"操作记录保存成功: {session_id} - {operation}")
        except Exception as e:
            logger.error(f"保存操作记录失败: {e}")
            raise

    def get_session_records(self, session_id: str) -> List[Dict[str, Any]]:
        """
        获取会话的所有操作记录

        Args:
            session_id: 会话ID

        Returns:
            操作记录列表
        """
        try:
            data_file = self._get_data_file(session_id)

            if not data_file.exists():
                logger.warning(f"会话数据文件不存在: {session_id}")
                return []

            with open(data_file, "r", encoding="utf-8") as f:
                records = json.load(f)

            logger.info(f"获取会话记录成功: {session_id}, 记录数: {len(records)}")
            return records
        except Exception as e:
            logger.error(f"获取会话记录失败: {e}")
            return []

    def get_image_base64(self, image_path: str) -> Optional[str]:
        """
        读取图片并转换为base64编码

        Args:
            image_path: 图片路径

        Returns:
            base64编码的图片数据
        """
        try:
            if not os.path.exists(image_path):
                logger.error(f"图片文件不存在: {image_path}")
                return None

            with open(image_path, "rb") as f:
                image_data = f.read()

            # 判断图片类型
            ext = os.path.splitext(image_path)[1].lower()
            mime_type = {
                ".jpg": "image/jpeg",
                ".jpeg": "image/jpeg",
                ".png": "image/png",
                ".gif": "image/gif",
                ".bmp": "image/bmp"
            }.get(ext, "image/jpeg")

            base64_str = base64.b64encode(image_data).decode("utf-8")
            return f"data:{mime_type};base64,{base64_str}"
        except Exception as e:
            logger.error(f"读取图片失败: {e}")
            return None


# 全局会话存储服务实例（延迟初始化）
_session_storage_service_instance: Optional[SessionStorageService] = None


def get_session_storage_service() -> SessionStorageService:
    """获取会话存储服务单例"""
    global _session_storage_service_instance
    if _session_storage_service_instance is None:
        _session_storage_service_instance = SessionStorageService()
    return _session_storage_service_instance
