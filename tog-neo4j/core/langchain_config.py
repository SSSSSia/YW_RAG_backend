"""
LangChain 配置和初始化
"""
from langchain_ollama import ChatOllama
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_classic.memory import ConversationBufferMemory
from langchain_core.callbacks import StdOutCallbackHandler
from langchain_core.globals import set_llm_cache
from langchain_core.caches import InMemoryCache
from core.config import settings
from utils.logger import logger


class LangChainConfig:
    """LangChain 全局配置"""

    def __init__(self):
        # 启用缓存
        set_llm_cache(InMemoryCache())

        # 初始化 LLM (使用 Ollama)
        self.llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=0.7,
            base_url=settings.llm_api_url,
            timeout=60
        )

        # 用于规划的 LLM (低温度)
        self.planning_llm = ChatOllama(
            model=settings.LLM_MODEL,
            temperature=0.1,
            base_url=settings.llm_api_url,
        )

        logger.info("✅ LangChain 配置初始化完成")

    def create_memory(self, session_id: str = None) -> ConversationBufferMemory:
        """创建对话内存"""
        return ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="input",
            output_key="output"
        )

    def get_callbacks(self):
        """获取回调处理器"""
        return [StdOutCallbackHandler()]


# 全局配置实例
langchain_config = LangChainConfig()