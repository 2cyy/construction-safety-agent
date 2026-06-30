"""
广州腾讯总部大楼 - 施工安全管理AI智能体 后端服务
双引擎架构：YOLOv8 (精确检测) + Qwen-VL (语义理解)
"""
import os
import json
import base64
import re
import io
from datetime import datetime
from typing import List, Optional
from PIL import Image
import numpy as np

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import dashscope
from dashscope import MultiModalConversation

# ==================== 禁用代理 ====================
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

app = FastAPI(title="施工安全管理AI智能体后端 - 双引擎", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# ==================== 配置 ====================
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-0cc5e281843e4e9dbc0c9fdff2ef8d55")
dashscope.api_key = DASHSCOPE_API_KEY
MODEL_NAME = os.getenv("QWEN_MODEL", "qwen-vl-max")

# ==================== YOLO 双模型引擎 ====================
person_model = None      # 通用 YOLOv8n: 检测所有人
hardhat_model = None     # 专用安全帽模型: 分类 Hardhat / NO-Hardhat
models_ready = False

def init_models():
    """加载双模型：通用行人检测 + 专用安全帽分类"""
    global person_model, hardhat_model, models_ready
    try:
        from ultralytics import YOLO
        base_dir = os.path.dirname(os.path.abspath(__file__))

        # 模型1: 通用行人检测
        print("[YOLO] 加载通用行人检测模型 YOLOv8n...")
        person_model = YOLO("yolov8n.pt")

        # 模型2: 专用安全帽分类
        hardhat_path = os.path.join(base_dir, "hardhat_model.pt")
        print(f"[YOLO] 加载专用安全帽检测模型...")
        hardhat_model = YOLO(hardhat_path)

        models_ready = True
        print("[YOLO] 双模型就绪: 通用检测 + 专用安全帽分类")
    except ImportError:
        print("[YOLO] ultralytics 未安装")
    except FileNotFoundError as e:
        print(f"[YOLO] 模型文件未找到: {e}")
    except Exception as e:
        print(f"[YOLO] 模型加载失败: {e}")

def detect_persons(image) -> dict:
    """
    YOLOv8n 负责精确人员定位和计数
    安全帽/吸烟等安全判断由 Qwen-VL 负责
    """
    if not models_ready:
        init_models()
    if not models_ready or person_model is None:
        return {"engine":"error","total_persons":0,"no_helmet_detected":0,"persons":[],"image_width":image.width,"image_height":image.height,"error":"模型未加载"}

    img_array = np.array(image)
    persons = []

    try:
        person_results = person_model(img_array, classes=[0], conf=0.3, verbose=False)
        if person_results and len(person_results) > 0 and person_results[0].boxes is not None:
            for i, box in enumerate(person_results[0].boxes):
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()[:4].astype(int)
                conf = float(box.conf[0]) if hasattr(box.conf, '__len__') else float(box.conf)
                persons.append({
                    "id": i + 1,
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "confidence": round(conf, 2),
                    "has_helmet": None,  # 由Qwen-VL判断
                    "detector": "YOLOv8n"
                })

        return {
            "engine": "YOLOv8n",
            "total_persons": len(persons),
            "no_helmet_detected": 0,  # YOLO只数人，安全帽由千问判断
            "persons": persons,
            "image_width": image.width,
            "image_height": image.height
        }

    except Exception as e:
        return {
            "engine": "error",
            "total_persons": 0,
            "no_helmet_detected": 0,
            "persons": [],
            "image_width": image.width,
            "image_height": image.height,
            "error": f"检测引擎异常: {str(e)}"
        }


# ==================== Qwen-VL 语义引擎 ====================
QWEN_SYSTEM_PROMPT = """你是一个极度严格的工地安全检查AI，正在监控广州腾讯总部大楼的硬装修工程。

你的检测标准：安全第一，零容忍！哪怕只有一点点疑似违规，也要标记出来。

你需要同时检查以下所有维度：
1. 安全帽 — 逐个人看头部！没戴安全帽的、安全帽拿在手里的、戴鸭舌帽的，都算违规
2. 吸烟 — 是否有香烟、烟头、烟雾，手在嘴边的吸烟姿态
3. 安全通道堵塞 — 通道/走廊/楼梯口是否被材料、工具阻挡
4. 材料摆放违规 — 材料是否散落在非指定区域、是否阻挡通行
5. 整体安全评估

重要：只返回JSON，不要有任何解释文字。"""

QWEN_USER_PROMPT = """请极其仔细地分析这张工地图片。按照以下步骤逐一检查（在心里完成即可，只输出最终JSON）：

第1步：先数清楚图中一共有多少个人（包括远处和部分可见的人）
第2步：逐个人检查头部——这个人头上有没有戴安全帽？
  - 安全帽 = 硬质的、有明显帽檐的黄色/白色/红色/蓝色头盔
  - 鸭舌帽、布帽、头巾、光头、头发可见 ≠ 安全帽
  - 看不清头部、头部被遮挡 → 严格原则：标记为疑似未戴安全帽
第3步：检查有没有人在吸烟
第4步：检查安全通道有没有被堵
第5步：检查材料有没有乱放

返回JSON格式（数值必须是整数）：
{"total_people": 图中总人数, "no_helmet": 未戴/疑似未戴安全帽人数, "smoking": 0或1, "channel_blocked": 0或1, "material_issue": 0或1, "details": "具体描述所有违规位置和类型"}"""


def qwen_scene_analysis(image_base64: str) -> dict:
    """千问引擎：场景级语义分析"""
    messages = [
        {"role": "system", "content": [{"text": QWEN_SYSTEM_PROMPT}]},
        {"role": "user", "content": [
            {"image": f"data:image/jpeg;base64,{image_base64}"},
            {"text": QWEN_USER_PROMPT}
        ]}
    ]

    try:
        response = MultiModalConversation.call(model=MODEL_NAME, messages=messages)
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
            return {"error": "千问未返回有效文本"}

        parsed = extract_json(result_text)
        if parsed:
            return {"success": True, "engine": f"Qwen-VL ({MODEL_NAME})", **parsed, "raw_response": result_text}
        else:
            return {"error": "JSON解析失败", "raw": result_text[:300]}

    except Exception as e:
        return {"error": f"千问API调用失败: {str(e)}"}


def extract_json(text: str) -> Optional[dict]:
    """从模型返回文本中提取JSON"""
    # 策略1: markdown代码块
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    # 策略2: 匹配花括号
    m = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    # 策略3: 直接解析
    try: return json.loads(text.strip())
    except: pass
    return None


# ==================== 请求/响应模型 ====================
class ImageAnalysisRequest(BaseModel):
    image_base64: str
    model: str = "qwen-vl-max"


class DetectionResult(BaseModel):
    success: bool
    timestamp: str
    # YOLO引擎结果
    yolo_total_persons: int = 0
    yolo_no_helmet: int = 0
    persons: List[dict] = []
    # 千问引擎结果
    qwen_total_people: int = 0
    qwen_no_helmet: int = 0
    qwen_smoking: int = 0
    qwen_channel_blocked: int = 0
    qwen_material_issue: int = 0
    qwen_details: str = ""
    # 综合（双引擎融合，取更严格的值）
    total_people: int = 0
    no_helmet: int = 0
    overall_score: int = 0
    engines_used: List[str] = []
    # 调试
    qwen_raw: str = ""
    error: str = ""


# ==================== 核心API ====================
@app.get("/")
def root():
    return {
        "service": "施工安全管理AI智能体后端",
        "version": "2.0.0",
        "architecture": "双引擎：YOLOv8n (精确检测) + Qwen-VL (语义理解)",
        "endpoints": {
            "POST /analyze": "上传base64图片进行双引擎安全分析",
            "GET /health": "健康检查"
        }
    }


@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "qwen_model": MODEL_NAME,
        "yolo_ready": models_ready,
        "api_configured": DASHSCOPE_API_KEY != "sk-your-api-key-here",
        "timestamp": datetime.now().isoformat()
    }


@app.post("/analyze", response_model=DetectionResult)
def analyze_image_endpoint(request: ImageAnalysisRequest):
    """
    双引擎分析：
    1. YOLOv8n — 精确检测人员位置 + 安全帽颜色判断
    2. Qwen-VL — 场景级语义理解（吸烟/通道/材料）
    """
    if DASHSCOPE_API_KEY == "sk-your-api-key-here":
        raise HTTPException(status_code=500, detail="请配置 DASHSCOPE_API_KEY")

    errors = []
    yolo_result = {}
    qwen_result = {}
    engines_used = []

    # 解码图片
    try:
        img_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解码失败: {str(e)}")

    # ===== 引擎1: YOLOv8 精确检测 =====
    try:
        yolo_result = detect_persons(image)
        engines_used.append("YOLOv8n")
    except Exception as e:
        errors.append(f"YOLO引擎: {str(e)}")

    # ===== 引擎2: Qwen-VL 语义分析 =====
    try:
        qwen_result = qwen_scene_analysis(request.image_base64)
        if qwen_result.get("success"):
            engines_used.append(f"Qwen-VL ({MODEL_NAME})")
        else:
            errors.append(f"千问引擎: {qwen_result.get('error', '未知错误')}")
    except Exception as e:
        errors.append(f"千问引擎: {str(e)}")

    # ===== 双引擎数据融合 =====
    yolo_persons = yolo_result.get("total_persons", 0)

    qwen_people = qwen_result.get("total_people", 0)
    qwen_no_helmet = qwen_result.get("no_helmet", 0)
    qwen_smoking = qwen_result.get("smoking", 0)
    qwen_blocked = qwen_result.get("channel_blocked", 0)
    qwen_material = qwen_result.get("material_issue", 0)

    total_people = max(yolo_persons, qwen_people)
    no_helmet = qwen_no_helmet

    score = max(0, 100 - no_helmet*10 - qwen_smoking*25 - qwen_blocked*15 - qwen_material*10)

    return DetectionResult(
        success=len(engines_used) > 0,
        timestamp=datetime.now().isoformat(),
        yolo_total_persons=yolo_persons,
        yolo_no_helmet=0,
        persons=yolo_result.get("persons", []),
        qwen_total_people=qwen_people,
        qwen_no_helmet=qwen_no_helmet,
        qwen_smoking=qwen_smoking,
        qwen_channel_blocked=qwen_blocked,
        qwen_material_issue=qwen_material,
        qwen_details=qwen_result.get("details", ""),
        total_people=total_people,
        no_helmet=no_helmet,
        overall_score=score,
        engines_used=engines_used,
        qwen_raw=qwen_result.get("raw_response", ""),
        error="; ".join(errors) if errors else ""
    )


# ==================== 启动 ====================
if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  广州腾讯总部大楼 - 施工安全管理AI智能体 后端 v2.0     ║
    ║  双引擎: YOLOv8n (精确检测) + Qwen-VL (语义理解)       ║
    ║                                                        ║
    ║  uvicorn app:app --reload --port 8000                  ║
    ║  API文档: http://localhost:8000/docs                   ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8000)
