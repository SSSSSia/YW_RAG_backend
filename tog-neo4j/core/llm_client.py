"""
LLM客户端封装
"""
from typing import Dict, Any
from utils.logger import logger
from core.config import settings
from langchain_ollama import ChatOllama
from langchain_core.messages import HumanMessage


class LLMClient:
    """LLM客户端"""

    def __init__(self, model: str = None, api_url: str = None):
        self.model = model or settings.llm_model
        self.api_url = api_url or settings.llm_api_url
        
        # 创建 ChatOllama 实例
        self.client = ChatOllama(
            model=self.model,
            temperature=0.0,
            base_url=self.api_url,
            timeout=120
        )

    def chat(self, prompt: str, temperature: float = 0.0, max_tokens: int = 3000) -> str:
        """调用Ollama聊天接口"""
        try:
            # 临时创建具有指定温度和max_tokens的客户端
            temp_client = ChatOllama(
                model=self.model,
                temperature=temperature,
                base_url=self.api_url,
                timeout=120,
                num_predict=max_tokens
            )
            
            response = temp_client.invoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            return ""

    async def generate(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2000) -> str:
        """调用LLM生成接口"""
        try:
            # 临时创建具有指定温度和max_tokens的客户端
            temp_client = ChatOllama(
                model=self.model,
                temperature=temperature,
                base_url=self.api_url,
                timeout=120,
                num_predict=max_tokens
            )
            
            response = await temp_client.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()

        except Exception as e:
            logger.error(f"LLM调用失败: {e}")
            raise Exception(f"大模型调用失败: {str(e)}")


# 全局LLM客户端实例
llm_client = LLMClient()