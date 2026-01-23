"""
LLM客户端封装
"""
from typing import Dict, Any, Optional
from utils.logger import logger
from core.config import settings
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage


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

    # ==================== 硅基流动API方法 ====================

    def chat_with_siliconflow(self, prompt: str, temperature: float = 0.0, max_tokens: int = 3000,
                              api_key: str = None, model: str = None, system_prompt: str = None) -> str:
        """
        使用硅基流动API调用聊天接口

        Args:
            prompt: 用户提示词
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成token数
            api_key: API密钥，如果不提供则使用配置中的默认值
            model: 模型名称，如果不提供则使用配置中的默认值
            system_prompt: 系统提示词，如果不提供则使用默认值

        Returns:
            模型响应文本
        """
        try:
            # 使用传入的参数或配置中的默认值
            sf_api_key = api_key or settings.siliconflow_api_key
            sf_model = model or settings.siliconflow_model

            # 默认系统提示词
            default_system_prompt = "你是一个智能助手，请根据用户的问题提供准确、有用的回答。"

            if not sf_api_key:
                logger.error("硅基流动API密钥未设置，请在.env文件中设置SILICONFLOW_API_KEY")
                return ""

            # 创建硅基流动客户端
            sf_client = ChatOpenAI(
                model=sf_model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=sf_api_key,
                openai_api_base=settings.siliconflow_api_url,
                timeout=settings.siliconflow_timeout,
                max_retries=settings.siliconflow_max_retries
            )

            # 构建消息列表
            messages = []
            if system_prompt:
                messages.append(SystemMessage(content=system_prompt))
            else:
                messages.append(SystemMessage(content=default_system_prompt))
            messages.append(HumanMessage(content=prompt))

            response = sf_client.invoke(messages)
            return response.content.strip()
        except Exception as e:
            logger.error(f"硅基流动API调用失败: {e}")
            return ""

    async def generate_with_siliconflow(self, prompt: str, temperature: float = 0.3, max_tokens: int = 2000,
                                        api_key: str = None, model: str = None) -> str:
        """
        使用硅基流动API异步生成内容

        Args:
            prompt: 用户提示词
            temperature: 温度参数，控制随机性
            max_tokens: 最大生成token数
            api_key: API密钥，如果不提供则使用配置中的默认值
            model: 模型名称，如果不提供则使用配置中的默认值

        Returns:
            模型生成的文本
        """
        try:
            # 使用传入的参数或配置中的默认值
            sf_api_key = api_key or settings.siliconflow_api_key
            sf_model = model or settings.siliconflow_model

            if not sf_api_key:
                logger.error("硅基流动API密钥未设置，请在.env文件中设置SILICONFLOW_API_KEY")
                raise Exception("API密钥未设置")

            # 创建硅基流动客户端
            sf_client = ChatOpenAI(
                model=sf_model,
                temperature=temperature,
                max_tokens=max_tokens,
                openai_api_key=sf_api_key,
                openai_api_base=settings.siliconflow_api_url,
                timeout=settings.siliconflow_timeout,
                max_retries=settings.siliconflow_max_retries
            )

            response = await sf_client.ainvoke([HumanMessage(content=prompt)])
            return response.content.strip()
        except Exception as e:
            logger.error(f"硅基流动API调用失败: {e}")
            raise Exception(f"硅基流动大模型调用失败: {str(e)}")


# 全局LLM客户端实例
llm_client = LLMClient()