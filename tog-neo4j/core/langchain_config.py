"""
LangChain 配置和初始化
"""
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
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
            model=settings.llm_model,
            temperature=0.7,
            base_url=settings.llm_api_url,
            timeout=settings.llm_timeout,
            num_retries=settings.llm_max_retries
        )

        # 用于规划的 LLM (低温度)
        self.planning_llm = ChatOllama(
            model=settings.llm_model,
            temperature=0.1,
            base_url=settings.llm_api_url,
            timeout=settings.llm_timeout,
            num_retries=settings.llm_max_retries
        )

        logger.info("✅ LangChain 配置初始化完成")

    # ==================== 硅基流动API方法 ====================

    def get_siliconflow_llm(self, temperature: float = 0.7) -> ChatOpenAI:
        """
        获取硅基流动LLM实例

        Args:
            temperature: 温度参数

        Returns:
            ChatOpenAI实例
        """
        if not settings.siliconflow_api_key:
            logger.warning("硅基流动API密钥未设置，请在.env文件中设置SILICONFLOW_API_KEY")

        return ChatOpenAI(
            model=settings.siliconflow_model,
            temperature=temperature,
            openai_api_key=settings.siliconflow_api_key,
            openai_api_base=settings.siliconflow_api_url,
            timeout=settings.siliconflow_timeout,
            max_retries=settings.siliconflow_max_retries
        )

    def get_siliconflow_planning_llm(self, temperature: float = 0.1) -> ChatOpenAI:
        """
        获取硅基流动规划LLM实例（低温度）

        Args:
            temperature: 温度参数

        Returns:
            ChatOpenAI实例
        """
        if not settings.siliconflow_api_key:
            logger.warning("硅基流动API密钥未设置，请在.env文件中设置SILICONFLOW_API_KEY")

        return ChatOpenAI(
            model=settings.siliconflow_model,
            temperature=temperature,
            openai_api_key=settings.siliconflow_api_key,
            openai_api_base=settings.siliconflow_api_url,
            timeout=settings.siliconflow_timeout,
            max_retries=settings.siliconflow_max_retries
        )

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