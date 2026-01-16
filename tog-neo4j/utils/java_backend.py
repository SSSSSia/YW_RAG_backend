
"""
Java后端回调通知模块
"""
import httpx
from datetime import datetime
from typing import Optional
from utils.logger import logger
from core.config import settings


async def notify_java_backend(
        graph_key: str,
        code:int,
        build_message: str,
):
    """通知Java后端图谱创建结果"""
    callback_url = f"{settings.java_backend_url}{settings.java_callback_path}"

    payload = {
        "graph_key": graph_key,
        "code": code,
        "build_message": build_message,
    }

    try:
        logger.info(f"[{graph_key}] 📤 发送结果通知到Java后端: {callback_url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(callback_url, json=payload)

            if response.status_code == 200:
                logger.info(f"[{graph_key}] ✅ 成功通知Java后端")
            else:
                logger.warning(f"[{graph_key}] ⚠️ Java后端返回非200状态码: {response.status_code}")

    except httpx.TimeoutException:
        logger.error(f"[{graph_key}] ❌ 通知Java后端超时")
    except Exception as e:
        logger.error(f"[{graph_key}] ❌ 通知Java后端失败: {e}", exc_info=True)


async def get_knowledge_bases(page: int = 1, page_size: int = 10000):
    """获取知识库列表"""
    url = f"{settings.java_backend_url}/graphs/list"

    payload = {
        "page": page,
        "page_size": page_size
    }

    try:
        logger.info(f"📤 获取知识库列表: {url}")

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(url, json=payload)

            if response.status_code == 200:
                result = response.json()
                if result.get("code") == 200:
                    records = result.get("data", {}).get("records", [])
                    # 只提取graph_key、name、intro字段，并过滤掉name为"kk"的知识库以及status不为0的知识库
                    knowledge_bases = [
                        {
                            "graph_key": record.get("graph_key"),
                            "name": record.get("name"),
                            "intro": record.get("intro")
                        }
                        for record in records
                        if record.get("name") != "kk" and (record.get("status") == 2 or record.get("status") == "2")  # 过滤掉name为"kk"的知识库且只保留status为0的知识库（考虑字符串和数字两种情况）
                    ]
                    logger.info(f"✅ 成功获取知识库列表，共{len(knowledge_bases)}条记录")
                    return knowledge_bases
                else:
                    logger.warning(f"⚠️ Java后端返回错误: {result.get('message')}")
                    return []
            else:
                logger.warning(f"⚠️ Java后端返回非200状态码: {response.status_code}")
                return []

    except httpx.TimeoutException:
        logger.error(f"❌ 获取知识库列表超时")
        return []
    except Exception as e:
        logger.error(f"❌ 获取知识库列表失败: {e}", exc_info=True)
        return []