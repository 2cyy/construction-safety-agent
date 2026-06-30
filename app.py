"""
广州腾讯总部大楼 - 施工安全管理AI智能体 后端服务
基于 Qwen-VL 视觉大模型实现真实工地图片智能分析
"""
import os
import json
import base64
import re
from datetime import datetime
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import dashscope
from dashscope import MultiModalConversation

app = FastAPI(title="施工安全管理AI智能体后端", version="1.0.0")

# CORS 允许前端跨域访问
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== 配置 ====================
# 从环境变量读取 API Key（更安全），也可以直接写在这里
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-your-api-key-here")
dashscope.api_key = DASHSCOPE_API_KEY

# 默认模型
MODEL_NAME = os.getenv("QWEN_MODEL", "qwen-vl-max")


# ==================== 请求/响应模型 ====================
class ImageAnalysisRequest(BaseModel):
    image_base64: str  # base64 编码的图片
    model: str = "qwen-vl-max"  # 可选：qwen-vl-max / qwen-vl-plus


class DetectionResult(BaseModel):
    success: bool
    timestamp: str
    total_people: int = 0
    no_helmet: int = 0
    smoking: int = 0
    channel_blocked: int = 0
    material_issue: int = 0
    details: str = ""
    raw_response: str = ""
    model_used: str = ""


# ==================== 核心：千问视觉分析 ====================
def analyze_image(image_base64: str, model: str = "qwen-vl-max") -> dict:
    """
    调用千问 Qwen-VL 对工地图进行安全检测
    """
    system_prompt = """你是一个严格的工地安全检查AI检测员，正在监控广州腾讯总部大楼的硬装修工程。
你的任务是用计算机视觉能力仔细检查上传的图片，识别以下5类安全问题：
1. 未佩戴安全帽 (no_helmet) — 工人在施工区域但头部没有安全帽
2. 吸烟 (smoking) — 任何人在工地范围内吸烟或手持点燃的香烟
3. 安全通道堵塞 (channel_blocked) — 安全通道/消防通道被材料、工具或杂物阻塞
4. 材料摆放违规 (material_issue) — 建筑材料未按规定区域码放整齐，散落在通道上
5. 不区分安全帽颜色，只要戴了就算合规

重要规则：
- 只返回JSON格式，不要有任何解释文字
- 数值必须是整数
- 仔细数清楚图片中的人数
- 如果没有发现违规，对应字段填0
- detail字段用中文一句话描述最重要的安全风险，没有风险就说"现场安全状况良好"
"""

    user_prompt = """请检测这张工地图片，严格按照以下JSON格式返回结果：

{
    "total_people": 图片中的总人数(整数),
    "no_helmet": 未佩戴安全帽的人数(整数),
    "smoking": 吸烟的人数(整数),
    "channel_blocked": 安全通道是否被堵塞(0或1),
    "material_issue": 材料摆放是否有违规(0或1),
    "details": "一句话描述主要安全风险"
}

示例输出：
{"total_people": 8, "no_helmet": 2, "smoking": 1, "channel_blocked": 0, "material_issue": 0, "details": "A-3F区域2人未佩戴安全帽，1人在材料区附近吸烟，存在火灾隐患"}
"""

    messages = [
        {"role": "system", "content": [{"text": system_prompt}]},
        {
            "role": "user",
            "content": [
                {"image": f"data:image/jpeg;base64,{image_base64}"},
                {"text": user_prompt}
            ]
        }
    ]

    try:
        response = MultiModalConversation.call(
            model=model,
            messages=messages
        )

        # 提取模型返回的文本
        result_text = ""
        if response.output and response.output.choices:
            for choice in response.output.choices:
                if choice.message and choice.message.content:
                    for item in choice.message.content:
                        if isinstance(item, dict) and "text" in item:
                            result_text += item["text"]
                        elif isinstance(item, str):
                            result_text += item

        if not result_text:
            return {
                "success": False,
                "error": "模型未返回有效文本",
                "raw_response": str(response)
            }

        # 尝试从返回文本中提取JSON
        parsed = extract_json(result_text)

        if parsed:
            return {
                "success": True,
                **parsed,
                "raw_response": result_text,
                "model_used": model
            }
        else:
            return {
                "success": False,
                "error": "无法解析模型返回的JSON",
                "raw_response": result_text
            }

    except Exception as e:
        return {
            "success": False,
            "error": f"千问API调用失败: {str(e)}",
            "raw_response": ""
        }


def extract_json(text: str) -> dict:
    """
    从模型返回的文本中提取JSON对象
    千问有时会在JSON外面包裹 markdown 代码块或额外文字
    """
    # 策略1：尝试匹配 ```json ... ``` 代码块
    code_block_match = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if code_block_match:
        try:
            return json.loads(code_block_match.group(1))
        except json.JSONDecodeError:
            pass

    # 策略2：尝试匹配 { ... } 对象
    brace_match = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}', text, re.DOTALL)
    if brace_match:
        try:
            return json.loads(brace_match.group(0))
        except json.JSONDecodeError:
            pass

    # 策略3：尝试直接解析整个文本
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    return None


# ==================== API 端点 ====================
@app.get("/")
def root():
    return {
        "service": "施工安全管理AI智能体后端",
        "version": "1.0.0",
        "model": MODEL_NAME,
        "endpoints": {
            "POST /analyze": "上传base64图片进行安全分析",
            "GET /health": "健康检查",
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "model": MODEL_NAME,
        "api_configured": DASHSCOPE_API_KEY != "sk-your-api-key-here",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze", response_model=DetectionResult)
def analyze_image_endpoint(request: ImageAnalysisRequest):
    """
    接收 base64 编码的图片，调用千问 Qwen-VL 进行工地安全分析
    """
    if DASHSCOPE_API_KEY == "sk-your-api-key-here":
        raise HTTPException(
            status_code=500,
            detail="请在 app.py 中设置 DASHSCOPE_API_KEY 或通过环境变量传入"
        )

    # 调用千问分析
    result = analyze_image(request.image_base64, request.model)

    if not result.get("success"):
        raise HTTPException(status_code=500, detail=result.get("error", "分析失败"))

    return DetectionResult(
        success=True,
        timestamp=datetime.now().isoformat(),
        total_people=result.get("total_people", 0),
        no_helmet=result.get("no_helmet", 0),
        smoking=result.get("smoking", 0),
        channel_blocked=result.get("channel_blocked", 0),
        material_issue=result.get("material_issue", 0),
        details=result.get("details", ""),
        raw_response=result.get("raw_response", ""),
        model_used=result.get("model_used", MODEL_NAME),
    )


# ==================== 启动说明 ====================
if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════╗
    ║  广州腾讯总部大楼 - 施工安全管理AI智能体 后端服务    ║
    ║                                                      ║
    ║  启动命令: uvicorn app:app --reload --port 8000       ║
    ║  API文档:  http://localhost:8000/docs                 ║
    ║  健康检查: http://localhost:8000/health               ║
    ║                                                      ║
    ║  使用前请设置 DASHSCOPE_API_KEY 环境变量:            ║
    ║  set DASHSCOPE_API_KEY=sk-your-api-key-here           ║
    ╚══════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)
