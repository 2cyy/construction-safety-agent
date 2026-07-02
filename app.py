"""
================================================================================
广州腾讯总部大楼 · 施工安全管理AI智能体 V2.0
Event-Driven Industrial Architecture

感知层(YOLO) → 事件层(Event Generator) → 决策层(Rule Engine + Risk Scorer) → 解释层(Qwen-VL)
================================================================================
"""
import os, json, base64, re, io, uuid, time
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Tuple, Literal
from dataclasses import dataclass, field
from enum import Enum

import numpy as np
from PIL import Image
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import dashscope
from dashscope import MultiModalConversation

# ============================================================================
# SECTION 1: Configuration
# ============================================================================
os.environ["NO_PROXY"] = "*"
os.environ["no_proxy"] = "*"

DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-ws-H.RXEDMHH.JKCS.MEQCIBoSGeoet3bROI3O0m4I-IF9InfLMXO_yKEvD2WRKfbxAiAuUs9Lx9a-OYMkSkSE4v8j2UDK81oSFGHo63qERibhqA")
dashscope.api_key = DASHSCOPE_API_KEY
MODEL_NAME = os.getenv("QWEN_MODEL", "qwen-vl-max")

app = FastAPI(title="Construction Safety Agent V2 - Event-Driven Architecture", version="2.0.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])


# ============================================================================
# SECTION 2: Data Structures (Event Schema, Memory, Rules, Risk)
# ============================================================================

class AlertGrade(str, Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

class SafetyEvent(BaseModel):
    """标准化安全事件 — 所有检测输出统一为此格式"""
    event_id: str
    event_type: Literal["person_detected", "no_helmet", "smoking", "channel_blocked", "material_misplaced"]
    person_id: Optional[int] = None
    confidence: float = 0.0
    zone: Optional[str] = None
    bbox: Optional[List[int]] = None          # [x1, y1, x2, y2]
    timestamp: str = ""
    severity: AlertGrade = AlertGrade.INFO
    source: str = "yolo"                      # "yolo" | "hardhat_model" | "qwen-vl" | "rule_engine"
    metadata: dict = {}

@dataclass
class PersonTrack:
    """人员追踪记录"""
    track_id: int
    last_bbox: List[int]
    last_seen: float
    frames_seen: int = 1
    helmet_status: Optional[bool] = None      # True=wearing, False=not, None=unknown
    event_ids: List[str] = field(default_factory=list)

@dataclass
class SafetyRule:
    """安全规则定义"""
    rule_id: str
    name: str
    description: str
    event_type: str                            # 匹配的事件类型
    severity_on_violation: AlertGrade
    zone_weights: Dict[str, float] = field(default_factory=dict)
    cooldown_seconds: float = 5.0
    penalty: int = 10                          # 评分扣分值

class RiskScore(BaseModel):
    """风险评分报告"""
    overall: int = 100
    dimensions: Dict[str, int] = field(default_factory=lambda: {"helmet": 40, "smoking": 30, "passage": 20, "material": 10})
    alert_grade: AlertGrade = AlertGrade.INFO
    triggered_rules: List[str] = field(default_factory=list)
    zone_breakdown: Dict[str, int] = field(default_factory=dict)
    recommendations: List[str] = field(default_factory=list)
    event_count: int = 0
    timestamp: str = ""

class MemoryBuffer:
    """记忆缓冲：人员追踪 + 事件历史 + 去重"""
    def __init__(self, max_frames: int = 30):
        self.max_frames = max_frames
        self.tracks: Dict[int, PersonTrack] = {}
        self.event_history: List[SafetyEvent] = []
        self.frame_buffer: List[dict] = []
        self._next_track_id = 1
        self._last_event_times: Dict[str, float] = {}  # dedup key → timestamp

    def match_or_create_track(self, bbox: List[int], iou_threshold: float = 0.3) -> int:
        """IoU匹配现有追踪或创建新ID"""
        best_id, best_iou = None, 0.0
        for tid, track in self.tracks.items():
            iou = self._compute_iou(bbox, track.last_bbox)
            if iou > best_iou:
                best_iou, best_id = iou, tid
        if best_id is not None and best_iou > iou_threshold:
            return best_id
        tid = self._next_track_id
        self._next_track_id += 1
        return tid

    def update_track(self, track_id: int, bbox: List[int], helmet_status: Optional[bool] = None):
        """更新追踪记录"""
        now = time.time()
        if track_id in self.tracks:
            t = self.tracks[track_id]
            t.last_bbox = bbox
            t.last_seen = now
            t.frames_seen += 1
            if helmet_status is not None:
                t.helmet_status = helmet_status
        else:
            self.tracks[track_id] = PersonTrack(track_id=track_id, last_bbox=bbox, last_seen=now, helmet_status=helmet_status)

    def add_event(self, event: SafetyEvent, cooldown: float = 5.0) -> bool:
        """添加事件，冷却期内重复事件被过滤。返回True=已添加"""
        dedup_key = f"{event.event_type}_{event.person_id}_{event.zone}"
        now = time.time()
        if dedup_key in self._last_event_times:
            if now - self._last_event_times[dedup_key] < cooldown:
                return False
        self._last_event_times[dedup_key] = now
        self.event_history.append(event)
        if len(self.event_history) > 500:
            self.event_history = self.event_history[-300:]
        return True

    def get_recent_events(self, event_type: Optional[str] = None, zone: Optional[str] = None, limit: int = 50, since_seconds: float = 300) -> List[SafetyEvent]:
        """查询最近事件（默认5分钟内）"""
        cutoff = time.time() - since_seconds
        result = []
        for e in reversed(self.event_history):
            ts = datetime.fromisoformat(e.timestamp).timestamp() if e.timestamp else 0
            if ts < cutoff:
                continue
            if event_type and e.event_type != event_type:
                continue
            if zone and e.zone != zone:
                continue
            result.append(e)
            if len(result) >= limit:
                break
        return result

    def prune_old_tracks(self, max_age_seconds: float = 60):
        """清理过期追踪"""
        now = time.time()
        self.tracks = {tid: t for tid, t in self.tracks.items() if now - t.last_seen < max_age_seconds}

    @staticmethod
    def _compute_iou(a: List[int], b: List[int]) -> float:
        """计算两个bbox的IoU"""
        x1 = max(a[0], b[0]); y1 = max(a[1], b[1])
        x2 = min(a[2], b[2]); y2 = min(a[3], b[3])
        inter = max(0, x2-x1) * max(0, y2-y1)
        area_a = (a[2]-a[0]) * (a[3]-a[1])
        area_b = (b[2]-b[0]) * (b[3]-b[1])
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0

# 全局记忆缓冲（模块级单例）
_memory_buffer: Optional[MemoryBuffer] = None

def get_memory_buffer() -> MemoryBuffer:
    global _memory_buffer
    if _memory_buffer is None:
        _memory_buffer = MemoryBuffer()
    return _memory_buffer

# ============================================================================
# SECTION 3: YOLO Dual-Model Engine (Perception Layer)
# ============================================================================
person_model = None
hardhat_model = None
models_ready = False

def init_models():
    global person_model, hardhat_model, models_ready
    try:
        from ultralytics import YOLO
        base_dir = os.path.dirname(os.path.abspath(__file__))
        print("[PERCEPTION] Loading YOLOv8n (person detection)...")
        person_model = YOLO("yolov8n.pt")
        hardhat_path = os.path.join(base_dir, "hardhat_model.pt")
        if os.path.exists(hardhat_path):
            print("[PERCEPTION] Loading hardhat_model.pt (helmet classification)...")
            hardhat_model = YOLO(hardhat_path)
        else:
            print("[PERCEPTION] hardhat_model.pt not found, helmet classification delegated to Qwen-VL only")
        models_ready = True
        print("[PERCEPTION] Perception layer ready")
    except ImportError:
        print("[PERCEPTION] ultralytics not installed")
    except Exception as e:
        print(f"[PERCEPTION] Model load failed: {e}")

def detect_persons(image) -> dict:
    """感知层：YOLOv8n人员检测 + hardhat_model安全帽分类"""
    if not models_ready:
        init_models()
    if not models_ready:
        return {"engine": "error", "total_persons": 0, "no_helmet_detected": 0, "persons": [], "image_width": image.width, "image_height": image.height}

    img_array = np.array(image)
    persons = []
    helmet_ok = 0
    no_helmet = 0

    try:
        # Step 1: 通用YOLO检测所有人
        results = person_model(img_array, classes=[0], conf=0.3, verbose=False)
        if not results or len(results) == 0 or results[0].boxes is None:
            return {"engine": "YOLOv8n", "total_persons": 0, "no_helmet_detected": 0, "persons": [], "image_width": image.width, "image_height": image.height}

        for i, box in enumerate(results[0].boxes):
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()[:4].astype(int)
            conf = float(box.conf[0]) if hasattr(box.conf, '__iter__') else float(box.conf)

            # Step 2: 裁剪头部区域 → 专用硬帽模型分类
            has_helmet = None  # None = unknown
            head_y2 = y1 + int((y2 - y1) * 0.4)
            head_crop = img_array[max(0, y1-10):head_y2, x1:x2]

            if hardhat_model is not None and head_crop.size > 200:
                try:
                    hresults = hardhat_model(head_crop, conf=0.2, verbose=False)
                    if hresults and len(hresults) > 0 and hresults[0].boxes is not None:
                        best_conf, best_cls = 0, -1
                        for j in range(len(hresults[0].boxes)):
                            c = float(hresults[0].boxes.conf[j])
                            if c > best_conf:
                                best_conf, best_cls = c, int(hresults[0].boxes.cls[j])
                        if best_cls >= 0:
                            has_helmet = (best_cls == 0)  # class 0 = Hardhat
                except Exception:
                    pass

            if has_helmet is True:
                helmet_ok += 1
            elif has_helmet is False:
                no_helmet += 1
            # None → 不计数（unknown）

            persons.append({
                "id": i + 1,
                "bbox": [int(x1), int(y1), int(x2), int(y2)],
                "confidence": round(conf, 2),
                "has_helmet": has_helmet,
                "detector": "YOLOv8n+HardhatModel" if hardhat_model else "YOLOv8n"
            })

    except Exception as e:
        return {"engine": "error", "total_persons": 0, "no_helmet_detected": 0, "persons": [], "image_width": image.width, "image_height": image.height, "error": str(e)}

    return {
        "engine": "YOLOv8n" + ("+HardhatModel" if hardhat_model else ""),
        "total_persons": len(persons),
        "helmet_ok": helmet_ok,
        "no_helmet_detected": no_helmet,
        "persons": persons,
        "image_width": image.width,
        "image_height": image.height
    }


# ============================================================================
# SECTION 3B: Smoking Behavior Detection (Cigarette + Hand-Mouth + Temporal)
# ============================================================================
# 三级抽烟检测管道: ① cigarette专用模型 ② hand-mouth距离 ③ 5帧时序确认
_mp_pose = None
_mp_hands = None
_mp_drawing = None
_smoking_frame_buffer: List[dict] = []  # 5帧滑动窗口
SMOKING_FRAME_WINDOW = 5
SMOKING_CONFIRM_THRESHOLD = 3  # 3/5帧阳性 = 确认

def get_mediapipe():
    """懒加载 MediaPipe"""
    global _mp_pose, _mp_hands, _mp_drawing
    if _mp_pose is None:
        import mediapipe as mp
        _mp_pose = mp.solutions.pose
        _mp_hands = mp.solutions.hands
        _mp_drawing = mp.solutions.drawing_utils
    return _mp_pose, _mp_hands, _mp_drawing


def detect_smoking_behavior(image, persons: List[dict]) -> dict:
    """
    三级抽烟检测管道:
    ① cigarette专用模型检测烟头 (hardhat_model也可检测小目标)
    ② MediaPipe hand-mouth距离特征
    ③ 5帧时序窗口 (3/5阳性 = 确认吸烟)
    """
    img_array = np.array(image)
    h, w = img_array.shape[:2]
    results = {"smoking_detected": 0, "suspicious_persons": [], "hand_mouth_suspicious": [], "confidence": 0.0}

    # ===== ① Cigarette检测：用手部区域放大 + hardhat_model 检测小目标 =====
    for p in persons:
        x1, y1, x2, y2 = p.get("bbox", [0, 0, 0, 0])
        # 手部区域 (人体中部偏上，约40%-65%高度)
        hand_y1 = y1 + int((y2 - y1) * 0.40)
        hand_y2 = y1 + int((y2 - y1) * 0.65)
        hand_region = img_array[hand_y1:hand_y2, x1:x2]

        cigarette_found = False
        if hand_region.size > 300 and hardhat_model is not None:
            try:
                # 放大手部 2×
                from PIL import Image as PILImage
                hand_pil = PILImage.fromarray(hand_region).resize(
                    ((x2-x1)*2, (hand_y2-hand_y1)*2), PILImage.BILINEAR)
                hand_arr = np.array(hand_pil)
                hresults = hardhat_model(hand_arr, conf=0.15, verbose=False)
                if hresults and len(hresults) > 0 and hresults[0].boxes is not None:
                    # 检测到任何小目标在手部区域 → 标记可疑
                    for j in range(len(hresults[0].boxes)):
                        cls_id = int(hresults[0].boxes.cls[j])
                        conf = float(hresults[0].boxes.conf[j])
                        if conf > 0.2:  # 低阈值，捕捉小目标
                            cigarette_found = True
                            results["confidence"] = max(results["confidence"], conf)
                            break
            except Exception:
                pass

        if cigarette_found:
            results["suspicious_persons"].append(p.get("id", -1))

    # ===== ② Hand-Mouth 距离特征 (MediaPipe Pose) =====
    try:
        pose_mp, hands_mp, _ = get_mediapipe()
        with pose_mp.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            pose_results = pose.process(img_array)
            if pose_results.pose_landmarks:
                landmarks = pose_results.pose_landmarks.landmark
                # 嘴部: landmarks 9,10 (上下唇)
                mouth_x = (landmarks[9].x + landmarks[10].x) / 2 * w
                mouth_y = (landmarks[9].y + landmarks[10].y) / 2 * h
                # 手腕: landmarks 15,16 (左右手腕)
                for wrist_idx in [15, 16]:
                    wrist_x = landmarks[wrist_idx].x * w
                    wrist_y = landmarks[wrist_idx].y * h
                    distance = np.sqrt((wrist_x - mouth_x)**2 + (wrist_y - mouth_y)**2)
                    # 阈值: 手到嘴距离 < 头部尺寸的0.6倍
                    head_size = abs(landmarks[7].x - landmarks[8].x) * w * 0.8  # 耳朵距离估算头宽
                    threshold = head_size * 0.6
                    if distance < max(threshold, 30):  # 至少30px
                        results["hand_mouth_suspicious"].append({
                            "wrist": "left" if wrist_idx == 15 else "right",
                            "distance_px": round(distance, 1),
                            "threshold_px": round(threshold, 1)
                        })
    except Exception:
        pass  # MediaPipe 可能不可用

    # ===== ③ 5帧时序确认 =====
    frame_result = {
        "cigarette": len(results["suspicious_persons"]) > 0,
        "hand_mouth": len(results["hand_mouth_suspicious"]) > 0,
        "timestamp": time.time()
    }
    _smoking_frame_buffer.append(frame_result)
    if len(_smoking_frame_buffer) > SMOKING_FRAME_WINDOW:
        _smoking_frame_buffer.pop(0)

    # 3/5帧阳性 = 确认
    if len(_smoking_frame_buffer) >= SMOKING_CONFIRM_THRESHOLD:
        cigarette_pos = sum(1 for f in _smoking_frame_buffer if f["cigarette"])
        hand_mouth_pos = sum(1 for f in _smoking_frame_buffer if f["hand_mouth"])
        both_pos = sum(1 for f in _smoking_frame_buffer if f["cigarette"] and f["hand_mouth"])

        if both_pos >= SMOKING_CONFIRM_THRESHOLD:
            results["smoking_detected"] = 1  # 确认吸烟
            results["confidence"] = both_pos / SMOKING_FRAME_WINDOW
        elif cigarette_pos >= SMOKING_CONFIRM_THRESHOLD:
            results["smoking_detected"] = -1  # 疑似 (只有烟头，无手口动作)
            results["confidence"] = cigarette_pos / SMOKING_FRAME_WINDOW
        elif hand_mouth_pos >= SMOKING_CONFIRM_THRESHOLD:
            results["smoking_detected"] = -1  # 疑似 (只有动作，未检测到烟头)
            results["confidence"] = hand_mouth_pos / SMOKING_FRAME_WINDOW

    return results


# ============================================================================
# SECTION 4: Event Generator (Perception → Structured Events)
# ============================================================================

# 安全规则集
SAFETY_RULES: List[SafetyRule] = [
    SafetyRule(rule_id="R01", name="Helmet Mandatory", description="所有进入施工区人员必须佩戴安全帽",
               event_type="no_helmet", severity_on_violation=AlertGrade.CRITICAL,
               zone_weights={"B-1F": 1.5, "A-3F": 1.2}, cooldown_seconds=5.0, penalty=8),
    SafetyRule(rule_id="R02", name="Smoking Forbidden", description="施工区域严禁吸烟，材料堆放区吸烟直接触发紧急告警",
               event_type="smoking", severity_on_violation=AlertGrade.EMERGENCY,
               zone_weights={"B-1F": 2.0}, cooldown_seconds=10.0, penalty=25),
    SafetyRule(rule_id="R03", name="Passage Clear", description="安全通道必须保持畅通，堵塞超过5分钟升级为严重告警",
               event_type="channel_blocked", severity_on_violation=AlertGrade.WARNING,
               zone_weights={}, cooldown_seconds=60.0, penalty=15),
    SafetyRule(rule_id="R04", name="Material Orderly", description="材料必须按指定区域码放整齐，下班前必须检查",
               event_type="material_misplaced", severity_on_violation=AlertGrade.WARNING,
               zone_weights={}, cooldown_seconds=120.0, penalty=10),
]

def generate_events_from_yolo(persons: List[dict], zone: str = "unknown") -> List[SafetyEvent]:
    """YOLO检测结果 → 标准化SafetyEvent列表"""
    events = []
    ts = datetime.now().isoformat()
    buffer = get_memory_buffer()

    for p in persons:
        bbox = p.get("bbox", [0, 0, 0, 0])
        track_id = buffer.match_or_create_track(bbox)
        buffer.update_track(track_id, bbox, p.get("has_helmet"))

        # 事件1: 人员检测
        evt_person = SafetyEvent(
            event_id=str(uuid.uuid4())[:8],
            event_type="person_detected",
            person_id=track_id,
            confidence=p.get("confidence", 0),
            zone=zone,
            bbox=bbox,
            timestamp=ts,
            severity=AlertGrade.INFO,
            source="yolo"
        )
        buffer.add_event(evt_person, cooldown=1.0)
        events.append(evt_person)

        # 事件2: 未戴安全帽（仅当硬帽模型判定为False时）
        if p.get("has_helmet") is False:
            evt_helmet = SafetyEvent(
                event_id=str(uuid.uuid4())[:8],
                event_type="no_helmet",
                person_id=track_id,
                confidence=p.get("confidence", 0),
                zone=zone,
                bbox=bbox,
                timestamp=ts,
                severity=AlertGrade.WARNING,
                source="hardhat_model"
            )
            if buffer.add_event(evt_helmet, cooldown=5.0):
                events.append(evt_helmet)

    buffer.prune_old_tracks()
    return events


# ============================================================================
# SECTION 5: Rule Engine (Decision Layer)
# ============================================================================

class RuleEngine:
    """规则引擎：匹配事件 → 触发违规告警"""

    @staticmethod
    def evaluate(events: List[SafetyEvent], buffer: MemoryBuffer) -> Tuple[List[SafetyEvent], List[str]]:
        triggered_events = []
        triggered_rules = []

        for rule in SAFETY_RULES:
            matching = [e for e in events if e.event_type == rule.event_type]
            if not matching:
                continue

            triggered_rules.append(rule.rule_id)
            for evt in matching:
                # 应用规则中的严重等级和区域权重
                severity = rule.severity_on_violation
                # 高权重区域升级告警等级
                zone_w = rule.zone_weights.get(evt.zone or "", 1.0)
                if zone_w >= 2.0 and severity == AlertGrade.CRITICAL:
                    severity = AlertGrade.EMERGENCY
                elif zone_w >= 1.5 and severity == AlertGrade.WARNING:
                    severity = AlertGrade.CRITICAL

                rule_event = SafetyEvent(
                    event_id=str(uuid.uuid4())[:8],
                    event_type=evt.event_type,
                    person_id=evt.person_id,
                    confidence=evt.confidence,
                    zone=evt.zone,
                    bbox=evt.bbox,
                    timestamp=evt.timestamp,
                    severity=severity,
                    source="rule_engine",
                    metadata={"rule_id": rule.rule_id, "rule_name": rule.name, "zone_weight": zone_w}
                )
                if buffer.add_event(rule_event, cooldown=rule.cooldown_seconds):
                    triggered_events.append(rule_event)

        return triggered_events, triggered_rules


# ============================================================================
# SECTION 6: Risk Scorer
# ============================================================================

class RiskScorer:
    """风险评分引擎：zone加权评分 + 告警分级 + 建议生成"""

    @staticmethod
    def calculate(events: List[SafetyEvent], triggered_rules: List[str],
                  yolo_helmet_ok: int = 0, yolo_no_helmet: int = 0,
                  qwen_smoking: int = 0, qwen_blocked: int = 0, qwen_material: int = 0) -> RiskScore:

        # 维度扣分
        helmet_penalty = yolo_no_helmet * 8
        smoking_penalty = qwen_smoking * 25
        passage_penalty = qwen_blocked * 15
        material_penalty = qwen_material * 10

        dims = {
            "helmet": max(0, 40 - helmet_penalty),
            "smoking": max(0, 30 - smoking_penalty),
            "passage": max(0, 20 - passage_penalty),
            "material": max(0, 10 - material_penalty),
        }
        overall = max(0, sum(dims.values()))

        # 告警等级
        if overall >= 80:
            grade = AlertGrade.INFO
        elif overall >= 55:
            grade = AlertGrade.WARNING
        elif overall >= 25:
            grade = AlertGrade.CRITICAL
        else:
            grade = AlertGrade.EMERGENCY

        # 建议生成
        recs = []
        if dims["helmet"] < 30:
            recs.append("安全帽佩戴率偏低，建议加强A-3F楼层巡查")
        if dims["smoking"] < 25:
            recs.append("检测到吸烟违规，B-1F材料堆放区为重点禁烟区域")
        if dims["passage"] < 15:
            recs.append("安全通道堵塞，请立即清理并设置警示标识")
        if dims["material"] < 8:
            recs.append("材料摆放不达标，下班前安排人员重新规整")
        if not recs:
            recs.append("现场安全状况良好，继续保持")

        return RiskScore(
            overall=overall,
            dimensions=dims,
            alert_grade=grade,
            triggered_rules=triggered_rules,
            recommendations=recs,
            event_count=len(events),
            timestamp=datetime.now().isoformat()
        )


# ============================================================================
# SECTION 7: Qwen-VL Semantic Engine (Explanation Layer)
# ============================================================================

QWEN_SYSTEM_PROMPT = """你是一个极度严格的工地安全检查AI。安全第一，零容忍。

你需要将事件层检测到的违规转化为自然语言解释。关注以下维度：
1. 安全帽 — 未佩戴安全帽的人员位置和数量
2. 吸烟 — 吸烟行为的具体描述和火灾风险
3. 安全通道 — 堵塞程度和位置
4. 材料摆放 — 不合规的具体表现

只返回JSON，不要任何解释文字。"""

QWEN_USER_PROMPT = """分析这张工地图片。已经通过感知层检测到一些初步结果。
请极其仔细地逐人检查。

【吸烟检测 — 最高优先级！极易漏检！】
- 香烟目标极小（可能只有几个像素），但往往在工人手指间
- 逐人放大看手部：有没有细长白色/浅色物体夹在指间？
- 手靠近嘴部 + 指间有疑似烟头 → smoking=1
- 手部持物靠近口部但看不清是否为烟 → smoking=0 但 details 中说明"疑似"
- 不要因为目标小就说"未发现"！即使只有疑似也要在 details 中记录
- 关键特征：白色细长条、手指夹持姿势、手靠近嘴
- 不要求必须有烟雾——很多情况下只有烟头没有可见烟雾

【安全帽检测】
- 逐人看头部：有没有黄色/白色/红色/蓝色硬质安全帽？
- 鸭舌帽、布帽、头巾 ≠ 安全帽
- 安全帽拿在手里/挂腰间 = 未佩戴

【通道和材料】
- 通道/走廊是否有障碍物？
- 材料是否散落在非指定区域？

返回JSON（数值必须是整数，details必须用中文具体描述）：
{"total_people": 总人数, "no_helmet": 未戴安全帽人数, "smoking": 0或1, "channel_blocked": 0或1, "material_issue": 0或1, "details": "具体描述每个违规的位置和类型，未发现的维度请说明原因"}"""


def qwen_scene_analysis(image_base64: str) -> dict:
    """Qwen-VL 场景级语义分析（DashScope原生SDK）"""
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
        return {"error": "JSON解析失败", "raw": result_text[:300]}
    except Exception as e:
        return {"error": f"千问API调用失败: {str(e)}"}


def extract_json(text: str) -> Optional[dict]:
    m = re.search(r'```(?:json)?\s*\n?(.*?)\n?```', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(1))
        except: pass
    m = re.search(r'\{[^{}]*\{[^{}]*\}[^{}]*\}|\{[^{}]*\}', text, re.DOTALL)
    if m:
        try: return json.loads(m.group(0))
        except: pass
    try: return json.loads(text.strip())
    except: pass
    return None


# ============================================================================
# SECTION 8: API Endpoints
# ============================================================================

class ImageAnalysisRequest(BaseModel):
    image_base64: str
    model: str = "qwen-vl-max"
    zone: str = "unknown"

class DetectionResult(BaseModel):
    # V1 backward-compatible fields
    success: bool
    timestamp: str
    yolo_total_persons: int = 0
    yolo_no_helmet: int = 0
    persons: List[dict] = []
    qwen_total_people: int = 0
    qwen_no_helmet: int = 0
    qwen_smoking: int = 0
    qwen_channel_blocked: int = 0
    qwen_material_issue: int = 0
    qwen_details: str = ""
    total_people: int = 0
    no_helmet: int = 0
    overall_score: int = 0
    engines_used: List[str] = []
    qwen_raw: str = ""
    error: str = ""
    # V2 new fields
    events: List[dict] = []
    risk_score: Optional[dict] = None
    alert_grade: str = "info"
    rule_triggers: List[str] = []
    tracked_persons: int = 0
    helmet_ok: int = 0
    smoking_detected: int = 0
    smoking_confidence: float = 0.0
    smoking_method: str = ""
    cigarette_suspicious: int = 0
    hand_mouth_suspicious: int = 0
    recommendations: List[str] = []


@app.get("/")
def root():
    return {
        "service": "Construction Safety Agent V2",
        "architecture": "Event-Driven: Perception(YOLO) → Events → Rules → Risk → Explanation(Qwen-VL)",
        "version": "2.0.0",
        "endpoints": ["POST /analyze", "GET /events", "GET /risk", "GET /health", "GET /rules"]
    }

@app.get("/health")
def health_check():
    buffer = get_memory_buffer()
    return {
        "status": "ok",
        "qwen_model": MODEL_NAME,
        "perception_ready": models_ready,
        "hardhat_model_loaded": hardhat_model is not None,
        "api_configured": DASHSCOPE_API_KEY != "sk-your-api-key-here",
        "memory_buffer": {"active_tracks": len(buffer.tracks), "event_history": len(buffer.event_history)},
        "timestamp": datetime.now().isoformat()
    }

@app.get("/events")
def get_events(
    event_type: Optional[str] = Query(None),
    zone: Optional[str] = Query(None),
    limit: int = Query(50, ge=1, le=200),
    minutes: int = Query(5, ge=1, le=60)
):
    """查询最近的安全事件"""
    buffer = get_memory_buffer()
    events = buffer.get_recent_events(event_type=event_type, zone=zone, limit=limit, since_seconds=minutes*60)
    return {"count": len(events), "events": [e.model_dump() for e in events]}

@app.get("/risk")
def get_risk():
    """获取当前风险评分"""
    buffer = get_memory_buffer()
    recent = buffer.get_recent_events(since_seconds=600)
    triggered = list(set(e.metadata.get("rule_id", "") for e in recent if e.source == "rule_engine" and e.metadata.get("rule_id")))
    triggered = [r for r in triggered if r]  # filter empty strings
    no_helmet_count = sum(1 for e in recent if e.event_type == "no_helmet")
    score = RiskScorer.calculate(recent, triggered, yolo_no_helmet=no_helmet_count)
    return score.model_dump()

@app.get("/rules")
def get_rules():
    """获取当前安全规则集"""
    return {
        "rules": [
            {"rule_id": r.rule_id, "name": r.name, "description": r.description,
             "event_type": r.event_type, "severity": r.severity_on_violation.value,
             "zone_weights": r.zone_weights, "cooldown_seconds": r.cooldown_seconds}
            for r in SAFETY_RULES
        ]
    }

@app.post("/analyze", response_model=DetectionResult)
def analyze_image_endpoint(request: ImageAnalysisRequest):
    """核心分析端点：完整V2事件驱动流水线"""
    if DASHSCOPE_API_KEY == "sk-your-api-key-here":
        raise HTTPException(status_code=500, detail="请配置 DASHSCOPE_API_KEY")

    errors = []
    engines_used = []

    # Decode image
    try:
        img_bytes = base64.b64decode(request.image_base64)
        image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"图片解码失败: {str(e)}")

    # ===== Layer 1: Perception (YOLO) =====
    yolo_result = {}
    try:
        yolo_result = detect_persons(image)
        engines_used.append(yolo_result.get("engine", "YOLOv8n"))
    except Exception as e:
        errors.append(f"Perception layer: {str(e)}")

    # ===== Layer 1B: Smoking Behavior Detection =====
    smoking_result = {}
    try:
        smoking_result = detect_smoking_behavior(image, persons)
        if smoking_result.get("smoking_detected", 0) == 1:
            engines_used.append("SmokingDetector(3-stage)")
        elif smoking_result.get("smoking_detected", 0) == -1:
            engines_used.append("SmokingDetector(suspicious)")
    except Exception as e:
        errors.append(f"SmokingDetector: {str(e)}")

    # ===== Layer 2: Event Generation =====
    zone = request.zone or "unknown"
    persons = yolo_result.get("persons", [])
    yolo_events = generate_events_from_yolo(persons, zone)
    # Inject smoking events
    if smoking_result.get("smoking_detected", 0) != 0:
        smoking_event = SafetyEvent(
            event_id=str(uuid.uuid4())[:8],
            event_type="smoking",
            confidence=smoking_result.get("confidence", 0.5),
            zone=zone,
            timestamp=datetime.now().isoformat(),
            severity=AlertGrade.EMERGENCY if smoking_result.get("smoking_detected")==1 else AlertGrade.WARNING,
            source="smoking_detector",
            metadata={
                "hand_mouth_suspicious": len(smoking_result.get("hand_mouth_suspicious", [])),
                "cigarette_suspicious": len(smoking_result.get("suspicious_persons", [])),
                "method": "cigarette_model+hand_mouth+temporal_5frame"
            }
        )
        yolo_events.append(smoking_event)
    buffer = get_memory_buffer()

    # ===== Layer 3: Rule Engine =====
    rule_events, rule_triggers = RuleEngine.evaluate(yolo_events, buffer)

    # ===== Layer 4: Qwen-VL (conditional — only for complex scene analysis) =====
    qwen_result = {}
    yolo_has_violation = yolo_result.get("no_helmet_detected", 0) > 0
    try:
        qwen_result = qwen_scene_analysis(request.image_base64)
        if qwen_result.get("success"):
            engines_used.append(f"Qwen-VL ({MODEL_NAME})")
        else:
            errors.append(f"Qwen-VL: {qwen_result.get('error', 'unknown')}")
    except Exception as e:
        errors.append(f"Qwen-VL: {str(e)}")

    # Generate Qwen events if violations found
    if qwen_result.get("smoking", 0) > 0:
        evt = SafetyEvent(event_id=str(uuid.uuid4())[:8], event_type="smoking", confidence=0.8, zone=zone,
                          timestamp=datetime.now().isoformat(), severity=AlertGrade.EMERGENCY, source="qwen-vl")
        buffer.add_event(evt, cooldown=10.0)
    if qwen_result.get("channel_blocked", 0) > 0:
        evt = SafetyEvent(event_id=str(uuid.uuid4())[:8], event_type="channel_blocked", confidence=0.8, zone=zone,
                          timestamp=datetime.now().isoformat(), severity=AlertGrade.WARNING, source="qwen-vl")
        buffer.add_event(evt, cooldown=60.0)
    if qwen_result.get("material_issue", 0) > 0:
        evt = SafetyEvent(event_id=str(uuid.uuid4())[:8], event_type="material_misplaced", confidence=0.8, zone=zone,
                          timestamp=datetime.now().isoformat(), severity=AlertGrade.WARNING, source="qwen-vl")
        buffer.add_event(evt, cooldown=120.0)

    # ===== Layer 5: Risk Scoring =====
    # 融合 Qwen + Smoking Detector 结果
    smoke_detector_result = 1 if smoking_result.get("smoking_detected", 0) == 1 else 0
    qwen_smoking = max(qwen_result.get("smoking", 0), smoke_detector_result)
    qwen_blocked = qwen_result.get("channel_blocked", 0)
    qwen_material = qwen_result.get("material_issue", 0)
    qwen_people = qwen_result.get("total_people", 0)
    qwen_no_helmet = qwen_result.get("no_helmet", 0)

    all_events = yolo_events + rule_events
    risk = RiskScorer.calculate(all_events, rule_triggers,
                                yolo_helmet_ok=yolo_result.get("helmet_ok", 0),
                                yolo_no_helmet=yolo_result.get("no_helmet_detected", 0),
                                qwen_smoking=qwen_smoking,
                                qwen_blocked=qwen_blocked,
                                qwen_material=qwen_material)

    # ===== Data Fusion =====
    yolo_persons = yolo_result.get("total_persons", 0)
    total_people = max(yolo_persons, qwen_people)
    no_helmet = max(yolo_result.get("no_helmet_detected", 0), qwen_no_helmet)

    return DetectionResult(
        success=len(engines_used) > 0,
        timestamp=datetime.now().isoformat(),
        yolo_total_persons=yolo_persons,
        yolo_no_helmet=yolo_result.get("no_helmet_detected", 0),
        persons=persons,
        qwen_total_people=qwen_people,
        qwen_no_helmet=qwen_no_helmet,
        qwen_smoking=qwen_smoking,
        qwen_channel_blocked=qwen_blocked,
        qwen_material_issue=qwen_material,
        qwen_details=qwen_result.get("details", ""),
        total_people=total_people,
        no_helmet=no_helmet,
        overall_score=risk.overall,
        engines_used=engines_used,
        qwen_raw=qwen_result.get("raw_response", ""),
        error="; ".join(errors) if errors else "",
        # V2 fields
        events=[e.model_dump() for e in all_events[-20:]],
        risk_score=risk.model_dump(),
        alert_grade=risk.alert_grade.value,
        rule_triggers=rule_triggers,
        tracked_persons=len(buffer.tracks),
        helmet_ok=yolo_result.get("helmet_ok", 0),
        smoking_detected=smoking_result.get("smoking_detected", 0),
        smoking_confidence=smoking_result.get("confidence", 0.0),
        smoking_method="cigarette_model+hand_mouth+temporal_5frame",
        cigarette_suspicious=len(smoking_result.get("suspicious_persons", [])),
        hand_mouth_suspicious=len(smoking_result.get("hand_mouth_suspicious", [])),
        recommendations=risk.recommendations,
    )


# ============================================================================
# SECTION 9: Startup
# ============================================================================
if __name__ == "__main__":
    import uvicorn
    print("""
    ╔══════════════════════════════════════════════════════════╗
    ║  Construction Safety Agent V2 — Event-Driven Industrial  ║
    ║  Perception → Events → Rules → Risk → Explanation       ║
    ║                                                          ║
    ║  uvicorn app:app --port 9000                            ║
    ║  API Docs: http://localhost:9000/docs                    ║
    ╚══════════════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=9000)
