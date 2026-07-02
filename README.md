# 🏗️ 广州腾讯总部大楼 — 施工安全管理 AI 智能体

> 一个事件驱动的施工安全 AI Agent，从感知到决策的完整闭环。

[![GitHub Pages](https://img.shields.io/badge/在线体验-GitHub_Pages-blue)](https://2cyy.github.io/construction-safety-agent/)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-teal)](https://fastapi.tiangolo.com/)

---

## 为什么做这个项目？

广州腾讯总部大楼整栋硬装修工程，32 路摄像头覆盖 12 个施工楼层。安全总监不可能 24 小时盯着监控屏幕。这个 Agent 的目标是替代人力完成 4 项核心职责：

| 职责 | 为什么需要 AI | 为什么不是简单监控 |
|------|-------------|-----------------|
| 识别未戴安全帽 | 人工盯 32 路画面无法实时覆盖 | YOLO 逐帧检测 + 硬帽模型分类，40ms/帧 |
| 识别工地吸烟 | 香烟仅 10-20 像素，人眼也容易漏 | 多阶段行为识别（非单帧目标检测） |
| 监测安全通道堵塞 | 堵塞判断需要量化标准 | SAM 前景分割 + 占用率阈值 + 持续时间 |
| 每日材料摆放检查 | "合规"标准因材料类型而异 | 每区独立模板 + SSIM 结构相似度量化 |

---

## 架构设计理念

### 不是 "YOLO + LLM 拼接" ，而是事件驱动 Agent

**为什么不用简单拼接？** 因为安全决策需要可追溯和一致性。YOLO 结果和 LLM 结果冲突时，谁说了算？规则需要可审计。

**设计选择：五层事件驱动架构**

```
📸 感知层 (Perception)     YOLOv8n + hardhat_model.pt
        ↓                   精确人员定位，不负责安全判断
📋 事件层 (Event)           SafetyEvent — 标准化 JSON 事件
        ↓                   统一 Schema，可追溯，可审计
⚡ 决策层 (Decision)        Rule Engine — 4 条可配置安全规则
        ↓                   规则独立于模型，可热更新
📊 评估层 (Assessment)      Risk Scorer — 4 维度 + 4 级告警
        ↓                   zone 加权评分，info/warning/critical/emergency
🧠 解释层 (Explanation)     Qwen-VL — 自然语言报告生成
        ↓                   仅在复杂场景调用，降成本
📱 交互层 (Interaction)     Web Dashboard + Chat
```

### 为什么每层独立？

- **可替换**：换 YOLO 模型不影响下游，换 LLM 不影响规则
- **可审计**：每个事件记录 source 字段（yolo/hardhat_model/qwen-vl/rule_engine）
- **可扩展**：新增检测类型只需加 SafetyEvent 枚举 + SafetyRule，不改架构

### 为什么用 Qwen-VL 而不是本地模型？

高频任务（人员检测、安全帽分类）用本地 YOLO，40ms/帧。低频任务（场景描述、报告生成）调千问 API。成本约 0.02 元/次，不需要 GPU 服务器。

### 为什么告警分 4 级而不是 2 级？

- **info**：正常检测记录，不推送
- **warning**：需关注，推送到现场安全员 APP
- **critical**：需立即处理，推送 + 声光报警
- **emergency**：火灾/坍塌级别，推送 + 报警 + 通知项目经理

分级避免"狼来了"效应——所有告警都标红，就没有告警了。

---

## 快速开始

### 前提

- Python 3.9+
- [阿里云 DashScope API Key](https://dashscope.console.aliyun.com/)

### 安装与启动

```bash
git clone git@github.com:2cyy/construction-safety-agent.git
cd construction-safety-agent
pip install -r requirements.txt
# 编辑 app.py 第 31 行，填入 API Key
uvicorn app:app --port 9000
```

浏览器打开 `index.html` 或 `https://2cyy.github.io/construction-safety-agent/`

---

## 项目结构

```
├── app.py                    # 事件驱动 Agent 后端 (V2)
│   ├── Section 1: Configuration
│   ├── Section 2: Data Structures (SafetyEvent, MemoryBuffer, SafetyRule, RiskScore)
│   ├── Section 3: Perception Layer (YOLOv8n + hardhat_model.pt)
│   ├── Section 4: Event Generator
│   ├── Section 5: Rule Engine (R01-R04)
│   ├── Section 6: Risk Scorer (4 dimensions + 4 alert grades)
│   ├── Section 7: Qwen-VL Explanation Layer
│   └── Section 8-9: API Endpoints + Startup
├── index.html                # Web 交互界面 (Dashboard + Chat + Upload)
├── hardhat_model.pt          # 专用安全帽分类模型 (6MB)
├── images/                   # 23 张真实工地图片
├── requirements.txt
└── 理论分析_施工安全管理AI智能体.md  # 25 个场景的商业→技术→方案分析
```

---

## 部署说明

### 本地开发
```bash
# 1. 安装依赖
pip install -r requirements.txt

# 2. 配置 API Key（编辑 app.py 第 32 行）
DASHSCOPE_API_KEY = "sk-your-key"

# 3. 下载模型（首次运行自动下载，或手动）
# YOLOv8n 自动下载；hardhat_model.pt 需手动下载：
curl -L -o hardhat_model.pt "https://hf-mirror.com/keremberke/yolov8n-hard-hat-detection/resolve/main/best.pt"

# 4. 启动后端
uvicorn app:app --reload --port 9101

# 5. 打开前端
# 直接双击 index.html，或访问 https://2cyy.github.io/construction-safety-agent/
```

### 环境变量
| 变量 | 默认值 | 说明 |
|------|--------|------|
| `DASHSCOPE_API_KEY` | (app.py 内置) | 阿里云百炼 API Key |
| `QWEN_MODEL` | `qwen-vl-max` | 视觉模型名称 |

### 数据库
- **当前**: SQLite (`safety_events.db`)，自动创建，适合 Demo
- **生产建议**: PostgreSQL/MySQL + 对象存储（MinIO/OSS）保存证据图

### 生产部署架构
```
摄像头 (RTSP/ONVIF/GB28181)
  → 边缘盒子 (Jetson Orin NX)
  → AI 推理服务 (YOLO + Qwen-VL)
  → 事件数据库 (PostgreSQL)
  → 对象存储 (证据图)
  → 后端 API (FastAPI)
  → 前端大屏 (Nginx)
  → 工单系统 + 权限账号 + 日志监控
```

---

## API

| 端点 | 说明 |
|------|------|
| `POST /analyze` | 上传 base64 图片，返回完整五层分析结果 |
| `GET /events?type=&zone=&limit=` | 查询事件历史 |
| `GET /risk` | 当前风险评分 + zone breakdown |
| `GET /rules` | 查看安全规则集 |
| `GET /health` | 健康检查(后端/DB/模型/API) |
| `GET /tickets?status=` | 工单查询 |
| `POST /tickets/{id}/resolve` | 标记工单已处理 |
| `POST /tickets/{id}/close` | 复检关闭工单 |
| `GET /events?type=&zone=&limit=` | 事件查询(SQLite持久化) |

启动后访问 `http://localhost:9101/docs` 查看交互式文档。

---

## 数据集

| 数据集 | 图像数 | 用途 |
|--------|--------|------|
| GDUT-HWD | 3,174 | 安全帽 5 分类训练 |
| Construction-PPE | 1,416 | 11 类 PPE 检测 |
| SH17 | 8,099 | 工业安全 17 类 |

---

## 许可证

MIT
