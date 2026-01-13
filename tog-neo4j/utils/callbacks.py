"""
Java后端回调通知模块
"""
import httpx
from datetime import datetime
from typing import Optional
from utils.logger import logger
from core.config import settings


async def notify_java_backend(
        grag_id: str,
        success: bool,
        message: str,
        file_saved: Optional[str] = None,
        error: Optional[str] = None,
        output_path: Optional[str] = None,
        json_extracted: Optional[str] = None
):
    """通知Java后端图谱创建结果"""
    callback_url = f"{settings.java_backend_url}{settings.java_callback_path}"

    payload = {
        "grag_id": grag_id,
        "success": success,
        "message": message,
        "timestamp": datetime.now().isoformat(),
        "file_saved": file_saved,
        "error": error,
        "output_path": output_path,
        "json_extracted": json_extracted,
        "database_imported": success
    }

    try:
        logger.info(f"[{grag_id}] 📤 发送结果通知到Java后端: {callback_url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload)

            if response.status_code == 200:
                logger.info(f"[{grag_id}] ✅ 成功通知Java后端")
            else:
                logger.warning(f"[{grag_id}] ⚠️ Java后端返回非200状态码: {response.status_code}")

    except httpx.TimeoutException:
        logger.error(f"[{grag_id}] ❌ 通知Java后端超时")
    except Exception as e:
        logger.error(f"[{grag_id}] ❌ 通知Java后端失败: {e}", exc_info=True)