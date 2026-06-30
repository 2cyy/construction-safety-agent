# 🏗️ 广州腾讯总部大楼 — 施工安全管理 AI 智能体

> 基于 **YOLOv8n + Qwen-VL 双引擎架构**，实现整栋楼硬装修工程的 7×24 智能安全监控。

[![GitHub Pages](https://img.shields.io/badge/在线体验-GitHub_Pages-blue)](https://2cyy.github.io/construction-safety-agent/)
[![Python](https://img.shields.io/badge/Python-3.9+-green)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.128-teal)](https://fastapi.tiangolo.com/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8n-ultralytics-orange)](https://docs.ultralytics.com/)
[![Qwen-VL](https://img.shields.io/badge/Qwen--VL-max-通义千问-purple)](https://tongyi.aliyun.com/)

---

## 📖 目录

- [项目概述](#项目概述)
- [系统架构](#系统架构)
- [功能清单](#功能清单)
- [快速开始](#快速开始)
- [项目结构](#项目结构)
- [数据集与模型](#数据集与模型)
- [API 文档](#api-文档)
- [技术细节](#技术细节)
- [常见问题](#常见问题)

---

## 项目概述

### 场景

广州腾讯总部大楼整栋硬装修工程，需要 AI 智能体替代安全总监履行以下职责：

| 序号 | 业务需求 | AI 实现 |
|------|----------|---------|
| 1 | 实时识别工人未佩戴安全帽 | YOLOv8n 精确人员定位 + Qwen-VL 逐人头部检查 |
| 2 | 实时识别工地吸烟行为 | Qwen-VL 场景级吸烟姿态+烟头识别 |
| 3 | 实时识别安全通道堵塞 | Qwen-VL 通道障碍物检测 + 三级预警 |
| 4 | 每日材料摆放合规检查 | Qwen-VL 材料区域合规分析 + 评分 |
| 5 | 自然语言安全查询与日报 | LLM 意图解析 + 结构化数据查询 |

### 亮点

- 🎯 **双引擎架构**：YOLO 精确数人 + 千问 安全分类，各司其职
- 📸 **真实图片分析**：上传任意工地图，千问自动识别违规
- 🌐 **永久在线**：前端部署在 GitHub Pages，后端本地运行
- 📊 **真实数据驱动**：基于 GDUT-HWD (3,174张) + Construction-PPE (1,416张) + SH17 (8,099张) 学术数据集
- 🔌 **API Key 安全**：密钥仅在后端，前端静态页面不含任何敏感信息

---

## 系统架构

```
┌──────────────────────────────────────────────────────────────────┐
│                       前端 (GitHub Pages)                         │
│  https://2cyy.github.io/construction-safety-agent/               │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌────────────────┐  │
│  │ 📸 上传   │  │ 💬 对话   │  │ 📊 面板   │  │ 🚨 告警推送    │  │
│  └────┬─────┘  └──────────┘  └──────────┘  └────────────────┘  │
│       │ base64 图片                                              │
├───────┼──────────────────────────────────────────────────────────┤
│       ▼                                                          │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │              后端 (FastAPI + Uvicorn)                       │  │
│  │              http://localhost:9001                          │  │
│  │                                                             │  │
│  │  ┌─────────────────┐    ┌──────────────────────────────┐   │  │
│  │  │ 🎯 YOLOv8n      │    │ 🧠 Qwen-VL-max               │   │  │
│  │  │ 人员精确定位     │    │ 安全帽逐人检查               │   │  │
│  │  │ 逐个bbox坐标     │    │ 吸烟/通道/材料语义分析       │   │  │
│  │  │ 人数精确计数     │    │ 自然语言安全描述             │   │  │
│  │  └────────┬────────┘    └──────────┬───────────────────┘   │  │
│  │           │                        │                        │  │
│  │           └────────┬───────────────┘                        │  │
│  │                    ▼                                         │  │
│  │           📊 双引擎融合                                       │  │
│  │           综合评分 + 违规列表 + bbox坐标                      │  │
│  └────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────┘
```

---

## 功能清单

### 前端功能（`index.html`）

| 功能 | 说明 |
|------|------|
| 📸 上传图片分析 | 选择工地照片 → base64 编码 → 发送后端 → 展示双引擎结果 |
| 💬 智能对话 | 自然语言查询安全帽/吸烟/通道/材料状况 |
| 📊 实时面板 | 今日违规统计、楼层选择、摄像头状态 |
| 🚨 告警推送 | 实时违规告警（模拟 + AI 分析结果） |
| 📋 每日报告 | 综合安全评分 + 各维度详情 + 材料区合规表 |
| 📈 趋势对比 | 今日 vs 昨日数据对比 |
| 🔌 后端监测 | 每30秒检查后端连接状态和 API Key 配置 |

### 后端功能（`app.py`）

| 端点 | 方法 | 说明 |
|------|------|------|
| `/` | GET | 服务信息和版本 |
| `/health` | GET | 健康检查（模型状态、API Key 状态） |
| `/analyze` | POST | 双引擎安全分析（base64图片 → 结构化结果） |
| `/docs` | GET | 自动生成的 Swagger API 文档 |

---

## 快速开始

### 前提条件

- Python 3.9+
- 阿里云 DashScope API Key（[免费申请](https://dashscope.console.aliyun.com/)）
- Git（可选，用于克隆项目）

### 1. 克隆项目

```bash
git clone git@github.com:2cyy/construction-safety-agent.git
cd construction-safety-agent
```

### 2. 安装依赖

```bash
pip install -r requirements.txt
```

如果公司网络受限，使用镜像：
```bash
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
```

### 3. 配置 API Key

编辑 `app.py` 第 32 行，将 `sk-your-api-key-here` 替换为你的真实 Key：

```python
DASHSCOPE_API_KEY = os.getenv("DASHSCOPE_API_KEY", "sk-你的真实Key")
```

或通过环境变量传入：
```bash
# Windows
set DASHSCOPE_API_KEY=sk-你的真实Key

# Mac/Linux
export DASHSCOPE_API_KEY=sk-你的真实Key
```

### 4. 下载安全帽检测模型（可选）

```bash
# 从 HuggingFace 镜像下载专用安全帽检测模型
curl -L -o hardhat_model.pt "https://hf-mirror.com/keremberke/yolov8n-hard-hat-detection/resolve/main/best.pt"
```

> 该模型为可选增强。不下载也不影响核心功能，YOLOv8n 会自动下载用于人员定位。

### 5. 启动后端

```bash
# 如果公司有代理，先绕过
set NO_PROXY=*
uvicorn app:app --reload --port 9001
```

看到以下输出表示启动成功：
```
[YOLO] 双模型就绪: 通用检测 + 专用安全帽分类
Uvicorn running on http://127.0.0.1:9001
```

### 6. 打开前端

浏览器打开：
```
https://2cyy.github.io/construction-safety-agent/
```

或本地打开：
```
E:\construction_safety_agent\index.html
```

### 7. 开始使用

1. 页面底部查看后端连接状态（显示"已连接 ✓"即正常）
2. 点击紫色 **📸 上传真实工地图AI分析** 按钮
3. 选择一张工地照片
4. 等待 5-15 秒，查看双引擎分析结果

---

## 项目结构

```
construction-safety-agent/
│
├── app.py                          # FastAPI 后端（双引擎核心）
├── index.html                      # 前端界面（单文件 SPA）
├── hardhat_model.pt                # 专用安全帽检测模型（6MB，可选）
├── requirements.txt                # Python 依赖清单
├── README.md                       # 本文件
├── .gitignore                      # Git 忽略规则
├── .nojekyll                       # GitHub Pages 跳过 Jekyll 处理
│
├── images/                         # 真实工地图片（23张）
│   ├── construction_*.jpg          # Pexels/Unsplash 免费商用摄影
│   ├── gdut_*.jpg                  # GDUT-HWD 学术数据集样本
│   ├── material_storage.jpg        # 材料堆放场景
│   ├── safety_passage.jpg          # 安全通道场景
│   └── worker_smoking.jpg          # 工人场景
│
└── 理论分析_施工安全管理AI智能体.md   # 完整理论分析文档
```

---

## 数据集与模型

### 训练数据参考

| 数据集 | 图像数 | 类别 | 来源 | 年份 |
|--------|--------|------|------|------|
| **GDUT-HWD** | 3,174 | 5类（蓝/白/黄/红/无安全帽） | 广东工业大学 | 2023 |
| **Construction-PPE** | 1,416 | 11类（安全帽/手套/背心等） | Ultralytics | 2025 |
| **SH17** | 8,099 | 17类（PPE+安全检测） | arXiv 公开 | 2024 |
| **PPE-Mendeley** | 3,212 | 4类（安全帽/背心） | 台湾工地实拍 | 2025 |
| **总计** | **15,901** | — | — | — |

### 模型架构

| 模型 | 用途 | 来源 |
|------|------|------|
| YOLOv8n | 通用人员检测（COCO预训练） | Ultralytics |
| keremberke/yolov8n-hard-hat-detection | 专用安全帽分类（可选增强） | HuggingFace (mAP 83.6%) |
| Qwen-VL-max | 场景语义分析 | 阿里云 DashScope |

---

## API 文档

启动后端后，访问 `http://localhost:9001/docs` 查看交互式 Swagger 文档。

### POST /analyze

**请求：**
```json
{
  "image_base64": "base64编码的图片字符串",
  "model": "qwen-vl-max"
}
```

**响应：**
```json
{
  "success": true,
  "timestamp": "2026-06-30T22:08:45",
  "yolo_total_persons": 7,
  "yolo_no_helmet": 0,
  "persons": [
    {"id": 1, "bbox": [301, 36, 338, 137], "confidence": 0.85}
  ],
  "qwen_total_people": 8,
  "qwen_no_helmet": 1,
  "qwen_smoking": 0,
  "qwen_channel_blocked": 1,
  "qwen_material_issue": 1,
  "qwen_details": "第8个人未戴安全帽，仅戴鸭舌帽；通道被杂物堵塞",
  "total_people": 8,
  "no_helmet": 1,
  "overall_score": 65,
  "engines_used": ["YOLOv8n", "Qwen-VL (qwen-vl-max)"]
}
```

---

## 技术细节

### 端口说明

| 端口 | 服务 | 说明 |
|------|------|------|
| 9001 | FastAPI 后端 | 双引擎分析服务 |
| 443 | GitHub Pages | 前端静态页面 |

### 代理问题

如果公司网络有代理（常见于企业环境），启动时需要绕过：

```bash
# Windows
set NO_PROXY=*
uvicorn app:app --port 9001

# 或在 app.py 中已默认设置
os.environ["NO_PROXY"] = "*"
```

### 双引擎融合策略

```
人数 = max(YOLO检测人数, 千问检测人数)     # 取最大值，避免漏检
未戴安全帽 = 千问检测的未戴安全帽人数        # 千问负责安全判断
评分 = 100 - 未戴帽×10 - 吸烟×25 - 通道×15 - 材料×10
```

### 前端兼容性

- Chrome / Edge / Firefox 最新版
- 支持移动端响应式布局
- 需要后端在 `localhost:9001` 运行才能使用 AI 分析功能

---

## 常见问题

**Q: 为什么上传图片后显示"无法连接后端"？**
A: 确认后端已启动：`uvicorn app:app --port 9001`，且端口未被占用。

**Q: 为什么分析结果不准确？**
A: Qwen-VL 是通用视觉模型，非专用安全检测器。对于关键场景，建议使用专用 YOLO 模型 + 大量标注数据微调。本项目中的 YOLOv8n 仅用于人员定位，安全分类由千问完成。

**Q: API Key 安全吗？**
A: API Key 仅存储在本地 `app.py` 中，前端静态页面不含任何敏感信息。前端通过 `localhost` 调用后端，Key 不会暴露到公网。

**Q: 可以部署到服务器吗？**
A: 可以。将后端部署到云服务器，修改前端 `BACKEND_URL` 为服务器地址即可。建议增加 HTTPS 和 API 鉴权。

**Q: 千问调用一次多少钱？**
A: Qwen-VL-max 约 0.02 元/张图。Qwen-VL-plus 更便宜，将 `app.py` 中 `MODEL_NAME` 改为 `qwen-vl-plus` 即可。

---

## 许可证

MIT License

---

## 作者

安全智能体项目组 · 广州腾讯总部大楼硬装修工程

*最后更新: 2026-06-30*
