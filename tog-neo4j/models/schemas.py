"""
所有Pydantic模型定义
"""
from typing import Any, Optional, List
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum


class ResponseCode(str, Enum):
    """响应状态码枚举"""
    SUCCESS = "200"
    FAIL = "500"
    ERROR = "500"
    NOT_FOUND = "404"
    UNAUTHORIZED = "401"
    FORBIDDEN = "403"
    BAD_REQUEST = "400"


class R(BaseModel):
    """统一API响应格式"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    code: str = ResponseCode.SUCCESS.value
    message: str = "操作成功"
    data: Optional[Any] = None

    @staticmethod
    def ok(message: str = "操作成功", data: Any = None, code: str = None) -> "R":
        if code is None:
            code = ResponseCode.SUCCESS.value
        return R(code=code, message=message, data=data)

    @staticmethod
    def fail(message: str = "操作失败", data: Any = None, code: str = None) -> "R":
        if code is None:
            code = ResponseCode.FAIL.value
        return R(code=code, message=message, data=data)

    @staticmethod
    def error(message: str, error_detail: str = None, code: str = None) -> "R":
        if code is None:
            code = ResponseCode.ERROR.value
        data = {"error_detail": error_detail} if error_detail else None
        return R(code=code, message=message, data=data)

    @staticmethod
    def not_found(message: str = "资源不存在", data: Any = None) -> "R":
        return R(code=ResponseCode.NOT_FOUND.value, message=message, data=data)

    @staticmethod
    def unauthorized(message: str = "未授权", data: Any = None) -> "R":
        return R(code=ResponseCode.UNAUTHORIZED.value, message=message, data=data)

    def to_dict(self, **kwargs) -> dict:
        return self.model_dump(**kwargs)


class MessageItem(BaseModel):
    """消息项"""
    role: str
    content: str


class ToGQueryRequest(BaseModel):
    """ToG查询请求"""
    grag_id: str
    max_depth: Optional[int] = 10
    max_width: Optional[int] = 3
    message_items: Optional[List[MessageItem]] = None


class GraphRAGQueryRequest(BaseModel):
    """GraphRAG查询请求"""
    grag_id: str
    message_items: Optional[List[MessageItem]] = None
    method: Optional[str] = "local"


class ToGGraphRAGQueryRequest(BaseModel):
    """ToG+GraphRAG混合查询请求"""
    grag_id: str
    max_depth: Optional[int] = 10
    max_width: Optional[int] = 3
    method: Optional[str] = "local"
    message_items: Optional[List[MessageItem]] = None


class CallbackPayload(BaseModel):
    """Java回调数据模型"""
    grag_id: str
    success: bool
    message: str
    timestamp: str
    file_saved: Optional[str] = None
    error: Optional[str] = None
    output_path: Optional[str] = None
    json_extracted: Optional[str] = None
    database_imported: bool

class AgentChatRequest(BaseModel):
    """Agent 统一对话请求模型"""
    user_id: str
    company_id: Optional[str]=None
    grag_id: Optional[str] = None
    message_items: Optional[List[MessageItem]] = None
