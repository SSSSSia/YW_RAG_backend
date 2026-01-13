"""
统一响应格式类
参考Java的R类实现
支持 Pydantic v2
"""
from typing import Any, Optional
from pydantic import BaseModel, ConfigDict
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

    # Pydantic v2 配置方式
    model_config = ConfigDict(
        # 允许任意类型的data字段
        arbitrary_types_allowed=True,
        # 使用枚举值而非对象
        use_enum_values=True,
        # JSON 序列化配置
        json_encoders={
            # 可以在这里添加特殊类型的序列化器
        }
    )

    code: str = ResponseCode.SUCCESS.value  # 响应状态码：字符串类型，便于前后端对接
    message: str = "操作成功"  # 响应描述信息
    data: Optional[Any] = None  # 响应数据

    @staticmethod
    def ok(message: str = "操作成功", data: Any = None, code: str = None) -> "R":
        """
        成功响应

        Args:
            message: 成功描述信息
            data: 响应数据
            code: 状态码，默认200

        Returns:
            R实例
        """
        if code is None:
            code = ResponseCode.SUCCESS.value
        return R(code=code, message=message, data=data)

    @staticmethod
    def fail(message: str = "操作失败", data: Any = None, code: str = None) -> "R":
        """
        失败响应

        Args:
            message: 失败描述信息
            data: 错误详情数据（可选）
            code: 错误状态码，默认500

        Returns:
            R实例
        """
        if code is None:
            code = ResponseCode.FAIL.value
        return R(code=code, message=message, data=data)

    @staticmethod
    def error(message: str, error_detail: str = None, code: str = None) -> "R":
        """
        错误响应（包含错误详情）

        Args:
            message: 错误描述信息
            error_detail: 详细错误信息
            code: 错误状态码

        Returns:
            R实例
        """
        if code is None:
            code = ResponseCode.ERROR.value
        data = {"error_detail": error_detail} if error_detail else None
        return R(code=code, message=message, data=data)

    @staticmethod
    def not_found(message: str = "资源不存在", data: Any = None) -> "R":
        """
        404 响应

        Args:
            message: 描述信息
            data: 额外数据

        Returns:
            R实例
        """
        return R(code=ResponseCode.NOT_FOUND.value, message=message, data=data)

    @staticmethod
    def unauthorized(message: str = "未授权", data: Any = None) -> "R":
        """
        401 响应

        Args:
            message: 描述信息
            data: 额外数据

        Returns:
            R实例
        """
        return R(code=ResponseCode.UNAUTHORIZED.value, message=message, data=data)

    def set_message(self, message: str) -> "R":
        """链式设置message"""
        self.message = message
        return self

    def set_data(self, data: Any) -> "R":
        """链式设置data"""
        self.data = data
        return self

    def set_code(self, code: str) -> "R":
        """链式设置code"""
        self.code = code
        return self

    # ===== 以下是兼容性方法 =====

    def dict(self, **kwargs) -> dict:
        """
        兼容 Pydantic v1 的 dict() 方法
        已弃用，建议使用 model_dump() 或直接返回模型实例
        """
        import warnings
        warnings.warn(
            "The `dict` method is deprecated; use `model_dump` instead or return model directly.",
            DeprecationWarning,
            stacklevel=2
        )
        return self.model_dump(**kwargs)

    def to_dict(self, **kwargs) -> dict:
        """
        转换为字典（推荐方法名）

        Args:
            **kwargs: model_dump 的参数
                - exclude_none: 排除 None 值
                - by_alias: 使用字段别名
                - mode: 序列化模式 ('python' 或 'json')

        Returns:
            字典格式的响应数据
        """
        return self.model_dump(**kwargs)

    def to_json_str(self, **kwargs) -> str:
        """
        转换为 JSON 字符串

        Args:
            **kwargs: model_dump_json 的参数

        Returns:
            JSON 字符串
        """
        return self.model_dump_json(**kwargs)

    def to_response(self) -> dict:
        """
        转换为标准字典格式（排除 None 值）

        Returns:
            不包含 None 值的字典
        """
        return self.model_dump(exclude_none=True)
