"""
LLM客户端封装
"""
import ollama
import httpx
from typing import Dict, Any
from utils.logger import logger
from core.config import settings


class LLMClient:
    """LLM客户端"""

    def __init__(self, model: str = None, api_url: str = None):
        self.model = model or settings.llm_model
        self.api_url = api_url or settings.llm_api_url

    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 3000) -> str:
        """调用Ollama聊天接口"""
        try:
            response = ollama.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                options={
                    "temperature": temperature,
                    "num_predict": max_tokens
                }
            )
            return response['message']['content'].strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return ""

    async def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """调用LLM生成接口"""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    "top_p": 0.9
                }
            }

            async with httpx.AsyncClient(timeout=120.0) as client:
                response = await client.post(self.api_url, json=payload)
                response.raise_for_status()
                result = response.json()

                if isinstance(result, dict):
                    return result.get("response", "")
                return str(result)

        except httpx.TimeoutException:
            logger.error("LLM调用超时")
            raise Exception("大模型调用超时")
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise Exception(f"大模型调用失败: {str(e)}")


# 全局LLM客户端实例
llm_client = LLMClient()