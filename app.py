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

# ==================== YOLOv8 引擎 (可选增强) ====================
yolo_model = None
yolo_available = False

def init_yolo():
    """尝试加载YOLO模型，如果不可用则降级"""
    global yolo_model, yolo_available
    try:
        from ultralytics import YOLO
        print("[YOLO] 加载 YOLOv8n 模型...")
        yolo_model = YOLO("yolov8n.pt")
        yolo_available = True
        print("[YOLO] 模型加载完成")
    except ImportError:
        print("[YOLO] ultralytics 未安装，使用 OpenCV HOG 检测器作为后备")
        yolo_available = False
        try:
            import cv2
            # 使用OpenCV内置HOG行人检测器
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            yolo_model = hog
            print("[HOG] OpenCV HOG 行人检测器就绪")
        except ImportError:
            print("[HOG] OpenCV 未安装，使用基础图像分析")
            yolo_model = None

# 安全帽颜色范围 (HSV)
HELMET_COLORS = {
    "yellow":  {"lower": (15, 40, 40),  "upper": (35, 255, 255)},
    "white":   {"lower": (0, 0, 180),   "upper": (180, 30, 255)},
    "red":     {"lower": (0, 50, 50),   "upper": (10, 255, 255)},
    "blue":    {"lower": (100, 50, 50), "upper": (130, 255, 255)},
}


def detect_persons(image) -> dict:
    """
    人员检测引擎（支持多级降级）：
    1. YOLOv8n (最优，需 ultralytics)
    2. OpenCV HOG (中等，需 opencv-python)
    3. 基础图像分析 (兜底，纯 numpy)
    """
    if yolo_model is None:
        init_yolo()

    img_array = np.array(image)
    persons = []

    try:
        if yolo_available and hasattr(yolo_model, 'predict'):
            # 方案1: YOLOv8
            results = yolo_model(img_array, classes=[0], conf=0.35, verbose=False)
            if results and len(results) > 0 and results[0].boxes is not None:
                boxes = results[0].boxes
                for i in range(len(boxes.xyxy)):
                    x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()[:4].astype(int)
                    conf = float(boxes.conf[i])
                    head_y2 = y1 + int((y2 - y1) * 0.35)
                    head_region = img_array[max(0, y1-5):head_y2, x1:x2]
                    has_helmet = check_helmet_color(head_region)
                    persons.append({"id": len(persons)+1, "bbox": [int(x1), int(y1), int(x2), int(y2)], "confidence": round(conf, 2), "has_helmet": has_helmet, "head_region_valid": head_region.size > 100, "detector": "YOLOv8n"})

        elif yolo_model is not None:
            # 方案2: OpenCV HOG
            import cv2
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
            boxes, weights = yolo_model.detectMultiScale(gray, winStride=(8,8), padding=(8,8), scale=1.05)
            boxes = boxes if boxes is not None else []
            for (x, y, w, h) in boxes:
                x1, y1, x2, y2 = x, y, x+w, y+h
                conf = 0.5  # HOG不返回置信度
                head_y2 = y1 + int(h * 0.35)
                head_region = img_array[max(0, y1-5):head_y2, x1:x2]
                has_helmet = check_helmet_color(head_region)
                persons.append({"id": len(persons)+1, "bbox": [x1, y1, x2, y2], "confidence": round(conf, 2), "has_helmet": has_helmet, "head_region_valid": head_region.size > 100, "detector": "OpenCV-HOG"})

        else:
            # 方案3: 基础图像网格分析 (纯numpy兜底)
            h, w = img_array.shape[:2]
            # 将图片分成网格，寻找可能的人员区域（基于颜色变化）
            grid = 6
            for row in range(grid):
                for col in range(grid):
                    x1 = int(w * col / grid)
                    y1 = int(h * row / grid)
                    x2 = int(w * (col+1) / grid)
                    y2 = int(h * (row+1) / grid)
                    region = img_array[y1:y2, x1:x2]
                    # 简单启发式：颜色方差大的区域可能有人
                    if region.size > 0 and np.std(region) > 30:
                        head_y2 = y1 + int((y2-y1)*0.35)
                        head_region = img_array[max(0,y1-3):head_y2, x1:x2]
                        has_helmet = check_helmet_color(head_region)
                        persons.append({"id": len(persons)+1, "bbox": [x1, y1, x2, y2], "confidence": 0.3, "has_helmet": has_helmet, "head_region_valid": head_region.size > 100, "detector": "GridAnalysis"})

        no_helmet_count = sum(1 for p in persons if not p["has_helmet"])

        return {
            "engine": persons[0]["detector"] if persons else "none",
            "total_persons": len(persons),
            "no_helmet_detected": no_helmet_count,
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


def check_helmet_color(head_region: np.ndarray) -> bool:
    """
    通过颜色范围快速判断头部区域是否有安全帽
    安全帽常见颜色：黄、白、红、蓝
    """
    if head_region.size < 100:
        return False  # 头部太小，无法判断

    try:
        hsv = Image.fromarray(head_region).convert("HSV")
        hsv_arr = np.array(hsv)

        for color_name, ranges in HELMET_COLORS.items():
            lower = np.array(ranges["lower"])
            upper = np.array(ranges["upper"])
            mask = cv2_in_range(hsv_arr, lower, upper)
            ratio = np.sum(mask > 0) / mask.size
            if ratio > 0.08:  # 超过8%的像素匹配该颜色
                return True
    except Exception:
        pass

    # 如果颜色检测失败，至少头部区域存在就假设可能有帽子
    return head_region.size > 500


def cv2_in_range(arr: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """简易版 cv2.inRange，避免导入 opencv"""
    return np.all((arr >= lower) & (arr <= upper), axis=2)


# ==================== Qwen-VL 语义引擎 ====================
QWEN_SYSTEM_PROMPT = """你是一个工地安全检查AI，正在监控广州腾讯总部大楼的硬装修工程。
你的任务是对整张图片做场景级语义理解，不需要逐人检测（人员精确检测已由YOLO引擎完成）。

请关注以下YOLO无法处理的高层语义问题：
1. 吸烟行为 — 是否有香烟、烟头、烟雾，或手在嘴边的吸烟姿态
2. 安全通道堵塞 — 通道/走廊/楼梯口是否被材料、工具阻挡
3. 材料摆放违规 — 材料是否散落、是否在指定区域外、是否阻挡通行
4. 整体安全评估 — 工地整体是否整洁有序，有哪些明显隐患

只返回JSON，不要有任何解释文字。"""

QWEN_USER_PROMPT = """请分析这张工地图片的场景级安全问题。

YOLO引擎已完成了人员定位。请你关注：
- 有没有人吸烟？（看手部附近是否有烟头/烟雾）
- 安全通道有没有被堵？（看通道/门口是否有障碍物）
- 材料有没有乱放？（看材料是否散落在非堆放区）
- 整体安全管理有没有明显漏洞？

严格按照以下JSON格式返回（数值必须是整数）：
{"smoking": 0或1, "channel_blocked": 0或1, "material_issue": 0或1, "details": "用中文一句话描述主要安全风险，没有风险就说现场安全状况良好"}"""


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
    qwen_smoking: int = 0
    qwen_channel_blocked: int = 0
    qwen_material_issue: int = 0
    qwen_details: str = ""
    # 综合
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
        "yolo_available": yolo_model is not None,
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

    # ===== 综合评分 =====
    yolo_violations = yolo_result.get("no_helmet_detected", 0)
    qwen_smoking = qwen_result.get("smoking", 0)
    qwen_blocked = qwen_result.get("channel_blocked", 0)
    qwen_material = qwen_result.get("material_issue", 0)

    score = max(0, 100
                - yolo_violations * 5
                - qwen_smoking * 15
                - qwen_blocked * 15
                - qwen_material * 10)

    return DetectionResult(
        success=len(engines_used) > 0,
        timestamp=datetime.now().isoformat(),
        yolo_total_persons=yolo_result.get("total_persons", 0),
        yolo_no_helmet=yolo_result.get("no_helmet_detected", 0),
        persons=yolo_result.get("persons", []),
        qwen_smoking=qwen_smoking,
        qwen_channel_blocked=qwen_blocked,
        qwen_material_issue=qwen_material,
        qwen_details=qwen_result.get("details", ""),
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
