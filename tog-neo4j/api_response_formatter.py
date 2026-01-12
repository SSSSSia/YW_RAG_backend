"""
API 响应格式化器
为前端提供清晰的步骤化输出
"""
from typing import Dict, Any, List


class ResponseFormatter:
    """响应格式化器"""

    @staticmethod
    def format_operation_response(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化操作流程响应

        Args:
            result: ToG 推理引擎返回的结果

        Returns:
            格式化后的响应，适合前端展示
        """
        if not result.get("success"):
            # ✅ 修复：确保包含所有必需字段
            return {
                "success": False,
                "question": result.get("question", ""),
                "answer": result.get("answer", result.get("message", "查询失败")),
                "message": result.get("answer", "查询失败"),
                "error": result.get("error_message"),
                "execution_time": result.get("execution_time", 0)
            }

        # 获取步骤信息
        steps = result.get("structured_steps", [])

        # 如果没有结构化步骤，从 steps 列表生成
        if not steps and result.get("steps"):
            steps = [
                {
                    "step_number": i,
                    "step_name": step,
                    "description": step
                }
                for i, step in enumerate(result["steps"], 1)
            ]

        # 生成简洁的步骤文本（用于前端直接显示）
        steps_text = "\n".join([
            f"步骤{step['step_number']}: {step['step_name']}"
            for step in steps
        ])

        return {
            "success": True,
            "question": result.get("question", ""),
            "operation": result.get("operation", ""),
            "answer": result.get("answer", steps_text),
            "execution_time": result.get("execution_time", 0),
        }

    @staticmethod
    def format_for_frontend_display(result: Dict[str, Any]) -> str:
        """
        生成适合前端直接显示的 Markdown 格式文本

        Returns:
            Markdown 格式的步骤文本
        """
        if not result.get("success"):
            return f"**查询失败**: {result.get('message', '未知错误')}"

        operation = result.get("operation", "操作")
        steps = result.get("steps", {}).get("items", [])

        if not steps:
            return "未找到相关步骤信息。"

        # 生成 Markdown 格式
        markdown = f"## {operation}\n\n"
        markdown += "### 操作步骤\n\n"

        for step in steps:
            markdown += f"**步骤 {step['step_number']}**: {step['step_name']}\n\n"

        return markdown

    @staticmethod
    def format_for_json_api(result: Dict[str, Any]) -> Dict[str, Any]:
        """
        格式化为标准 JSON API 响应
        适合 RESTful API 返回
        """
        formatted = ResponseFormatter.format_operation_response(result)

        return {
            "status": "success" if formatted["success"] else "error",
            "data": {
                "question": formatted.get("question"),
                "operation": formatted.get("operation"),
                "answer": formatted.get("answer"),
            } if formatted["success"] else None,
            "error": formatted.get("error") if not formatted["success"] else None,
            "execution_time": formatted.get("execution_time"),
        }
