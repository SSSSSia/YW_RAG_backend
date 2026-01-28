"""
所有Pydantic模型定义
"""
from typing import Any, Optional, List
from pydantic import BaseModel, ConfigDict, Field
from enum import Enum
from datetime import datetime


class ResponseCode(str, Enum):
    """响应状态码枚举"""
    SUCCESS = "200"
    FAIL = "500"
    ERROR = "500"
    NOT_FOUND = "404"
    UNAUTHORIZED = "401"
    FORBIDDEN = "403"
    BAD_REQUEST = "400"
    ALARM = "30001"  # AI审计告警


class R(BaseModel):
    """统一API响应格式"""

    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        use_enum_values=True,
    )

    code: str = Field(default=ResponseCode.SUCCESS.value, description="响应状态码")
    message: str = Field(default="操作成功", description="响应消息")
    data: Optional[Any] = Field(default=None, description="响应数据")

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
    def error(message: str, error_detail: str = None, code: str = None, data: Any = None) -> "R":
        if code is None:
            code = ResponseCode.ERROR.value
        if data is not None:
            # 如果提供了data参数，直接使用
            return R(code=code, message=message, data=data)
        else:
            # 否则使用旧的逻辑（包含error_detail）
            final_data = {"error_detail": error_detail} if error_detail else None
            return R(code=code, message=message, data=final_data)

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
    role: str = Field(..., description="消息角色（system/user/assistant）")
    content: str = Field(..., description="消息内容")


class ToGQueryRequest(BaseModel):
    """ToG查询请求"""
    grag_id: str = Field(..., description="知识图谱ID")
    max_depth: Optional[int] = Field(default=3, description="最大搜索深度")
    max_width: Optional[int] = Field(default=3, description="最大搜索宽度")
    message_items: Optional[List[MessageItem]] = Field(default=None, description="消息历史记录")


class GraphRAGQueryRequest(BaseModel):
    """GraphRAG查询请求"""
    grag_id: str = Field(..., description="知识图谱ID")
    message_items: Optional[List[MessageItem]] = Field(default=None, description="消息历史记录")
    method: Optional[str] = Field(default="local", description="查询方法（local/global）")


class ToGGraphRAGQueryRequest(BaseModel):
    """ToG+GraphRAG混合查询请求"""
    grag_id: str = Field(..., description="知识图谱ID")
    max_depth: Optional[int] = Field(default=5, description="最大搜索深度")
    max_width: Optional[int] = Field(default=5, description="最大搜索宽度")
    method: Optional[str] = Field(default="local", description="查询方法（local/global）")
    message_items: Optional[List[MessageItem]] = Field(default=None, description="消息历史记录")


class CallbackPayload(BaseModel):
    """Java回调数据模型"""
    grag_id: str = Field(..., description="知识图谱ID")
    success: bool = Field(..., description="是否成功")
    message: str = Field(..., description="回调消息")
    timestamp: str = Field(..., description="时间戳")
    file_saved: Optional[str] = Field(default=None, description="保存的文件路径")
    error: Optional[str] = Field(default=None, description="错误信息")
    output_path: Optional[str] = Field(default=None, description="输出路径")
    json_extracted: Optional[str] = Field(default=None, description="提取的JSON数据")
    database_imported: bool = Field(..., description="是否已导入数据库")

class AgentChatRequest(BaseModel):
    """Agent 统一对话请求模型"""
    user_id: str = Field(..., description="用户ID")
    company_id: Optional[str] = Field(default=None, description="公司ID")
    grag_id: Optional[str] = Field(default=None, description="知识图谱ID")
    message_items: Optional[List[MessageItem]] = Field(default=None, description="消息历史记录")


class SiliconFlowQueryRequest(BaseModel):
    """硅基流动API直接查询请求"""
    question: str = Field(..., description="用户问题")


# ==================== AI审计和总结接口相关模型 ====================

class CheckRequest(BaseModel):
    """AI审计请求"""
    sessionID: str = Field(..., description="会话ID（设备ID）")
    operation: str = Field(..., description="图片对应的操作")


class AuditOpt(BaseModel):
    """审计操作数据模型"""
    event_time: str = Field(..., description="事件时间", alias="eventTime")
    event_type: str = Field(..., description="事件类型", alias="eventType")
    event_content: str = Field(..., description="事件内容", alias="eventContent")
    event_status: str = Field(..., description="事件状态", alias="eventStatus")
    device_id: str = Field(..., description="设备ID", alias="deviceId")
    device_ip: str = Field(..., description="设备IP", alias="deviceIp")
    user: str = Field(..., description="用户")

    class Config:
        populate_by_name = True  # 允许使用别名


class AlarmData(BaseModel):
    """告警信息数据（内部使用，包含完整字段）"""
    equipment_asset: str = Field(..., description="设备编号（会话ID）")
    alarm: str = Field(..., description="告警信息")
    work_content: str = Field(..., description="本次工作内容")
    alarm_time: datetime = Field(..., description="告警时间")
    risk_level: str = Field(default="medium", description="风险等级（high/medium/low/none）")

    def to_api_response(self) -> dict:
        """转换为API响应格式（只包含Java后端需要的3个字段）"""
        return {
            "equipment_asset": self.equipment_asset,
            "alarm": self.alarm,
            "alarm_time": self.alarm_time
        }


class SummaryRequest(BaseModel):
    """AI总结请求"""
    sessionID: str = Field(..., description="会话ID（设备ID）")
    delete: bool = Field(False, description="是否在总结后删除该会话的所有记录")


class WorkOrderData(BaseModel):
    """工单数据"""
    ds_id: int = Field(..., description="设备ID（从sessionID转换而来）")
    work_class: int = Field(..., description="工单分类：1=软件，2=硬件")
    work_notice: str = Field(..., description="工作内容总结")
