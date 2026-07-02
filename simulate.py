"""
施工安全Agent — 仿真测试环境 (Simulation Environment)
生成模拟数据用于验证系统端到端流程: 感知→事件→风险→工单→复核→日报
所有数据均标注为模拟数据，不代表真实现场结果。
"""
import sqlite3, random, json, uuid, os
from datetime import datetime, timedelta

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "safety_events.db")

# 配置
NUM_EVENTS = 5000
NUM_TICKETS = 200
NUM_CAMERAS = 32
ZONES = ["A-1F","A-2F","A-3F","A-4F","A-5F","A-6F","B-1F","B-2F","B-3F"]
EVENT_TYPES = ["person_detected", "no_helmet", "smoking", "channel_blocked", "material_misplaced"]
WORKERS = [f"W-{i:04d}" for i in range(1, 51)]
SEVERITIES = ["info", "warning", "critical", "emergency"]
SOURCES = ["yolo", "hardhat_model", "qwen-vl", "rule_engine", "smoking_detector"]

def init_db():
    conn = sqlite3.connect(DB_PATH)
    conn.execute("""CREATE TABLE IF NOT EXISTS events (
        id INTEGER PRIMARY KEY AUTOINCREMENT, event_id TEXT UNIQUE, event_type TEXT,
        person_id INTEGER, confidence REAL, zone TEXT, bbox TEXT, timestamp TEXT,
        severity TEXT, source TEXT, metadata TEXT, image_base64 TEXT)""")
    conn.execute("""CREATE TABLE IF NOT EXISTS tickets (
        id INTEGER PRIMARY KEY AUTOINCREMENT, ticket_no TEXT UNIQUE, event_type TEXT,
        zone TEXT, description TEXT, assignee TEXT, deadline TEXT, status TEXT DEFAULT 'open',
        created_at TEXT, resolved_at TEXT, verified_at TEXT, verify_image TEXT)""")
    conn.commit()
    return conn

def generate_events(conn, count):
    """生成模拟违规事件"""
    print(f"[模拟] 生成 {count} 条事件...")
    base_time = datetime(2026, 7, 3, 6, 0, 0)
    inserted = 0

    for i in range(count):
        t = base_time + timedelta(seconds=random.randint(0, 43200))  # 6:00-18:00
        event_type = random.choices(EVENT_TYPES, weights=[40, 30, 8, 12, 10])[0]
        zone = random.choice(ZONES)
        severity = "warning"
        if event_type == "smoking": severity = "emergency"
        elif event_type == "no_helmet": severity = "warning"
        elif event_type == "channel_blocked": severity = random.choice(["warning","critical"])
        confidence = round(random.uniform(0.55, 0.98), 2)

        conn.execute("""INSERT OR IGNORE INTO events (event_id, event_type, person_id, confidence,
            zone, bbox, timestamp, severity, source, metadata)
            VALUES (?,?,?,?,?,?,?,?,?,?)""", (
            f"sim_{uuid.uuid4().hex[:8]}", event_type, random.randint(1,50), confidence,
            zone, json.dumps([random.randint(100,800) for _ in range(4)]),
            t.isoformat(), severity, random.choice(SOURCES),
            json.dumps({"simulated": True, "note": "模拟数据，用于流程验证"})
        ))
        inserted += 1

    conn.commit()
    print(f"  [OK] 已插入 {inserted} 条")

def generate_tickets(conn, count):
    """生成模拟工单，含完整生命周期"""
    print(f"[模拟] 生成 {count} 个工单（含状态流转）...")
    base_date = datetime(2026, 7, 3)
    inserted = 0

    for i in range(count):
        ticket_no = f"TK-{base_date.strftime('%Y%m%d')}-{i+1:04d}"
        event_type = random.choice(["no_helmet", "smoking", "channel_blocked", "material_misplaced"])
        zone = random.choice(ZONES)
        created = (base_date + timedelta(seconds=random.randint(0, 43200))).isoformat()

        # 工单状态: 40% closed, 30% resolved, 20% open, 10% in_progress
        status = random.choices(["open","in_progress","resolved","closed"], weights=[20,10,30,40])[0]
        resolved_at = None
        closed_at = None
        assignees = ["安全员-张工", "安全主管-李工", "现场监理-王工", "项目经理-赵工"]

        if status in ("resolved", "closed"):
            resolved_at = (datetime.fromisoformat(created) + timedelta(hours=random.randint(1,8))).isoformat()
        if status == "closed":
            closed_at = (datetime.fromisoformat(resolved_at) + timedelta(hours=random.randint(1,4))).isoformat()

        conn.execute("""INSERT OR IGNORE INTO tickets (ticket_no, event_type, zone, description,
            assignee, deadline, status, created_at, resolved_at, verified_at)
            VALUES (?,?,?,?,?,?,?,?,?,?)""", (
            ticket_no, event_type, zone, f"[模拟] {event_type} 违规检测",
            random.choice(assignees), (base_date + timedelta(days=1)).strftime("%Y-%m-%d 17:00"),
            status, created, resolved_at, closed_at
        ))
        inserted += 1

    conn.commit()
    print(f"  [OK] 已插入 {inserted} 个")
    # 统计
    stats = conn.execute("SELECT status, count(*) FROM tickets GROUP BY status").fetchall()
    for s in stats:
        print(f"    {s[0]}: {s[1]}")

def generate_camera_sim(conn):
    """生成摄像头模拟状态"""
    print(f"[模拟] 生成 {NUM_CAMERAS} 路摄像头状态...")
    cameras = []
    for i in range(1, NUM_CAMERAS+1):
        floor = random.choice(ZONES)
        status = random.choices(["online","online","online","offline","lagging"], weights=[80,80,80,5,5])[0]
        cameras.append({
            "id": f"CAM_{floor}_{i:02d}",
            "floor": floor,
            "status": status,
            "fps": random.randint(20,30) if status == "online" else 0,
            "resolution": "1080P",
            "protocol": random.choice(["RTSP","ONVIF","GB28181"]),
            "last_frame": datetime.now().isoformat()
        })
    online = sum(1 for c in cameras if c["status"] == "online")
    print(f"  [OK] 在线: {online}/{NUM_CAMERAS}")
    return cameras

def print_stats(conn):
    """打印统计信息"""
    print(f"\n{'='*50}")
    print(f"  仿真测试环境统计")
    print(f"{'='*50}")

    events = conn.execute("SELECT count(*) FROM events").fetchone()[0]
    print(f"  事件总数: {events}")

    for et in EVENT_TYPES:
        cnt = conn.execute("SELECT count(*) FROM events WHERE event_type=?", (et,)).fetchone()[0]
        print(f"    {et}: {cnt}")

    tickets = conn.execute("SELECT count(*) FROM tickets").fetchone()[0]
    print(f"\n  工单总数: {tickets}")
    for row in conn.execute("SELECT status, count(*) FROM tickets GROUP BY status"):
        print(f"    {row[0]}: {row[1]}")

    print(f"\n  所有数据均标注为模拟，用于验证系统端到端流程。")
    print(f"  感知→事件→风险→工单→复核→日报")

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════════╗
║  施工安全Agent — 仿真测试环境 v1.0                   ║
║  模拟数据用于验证系统流程，不代表真实现场结果         ║
╚══════════════════════════════════════════════════════╝
""")
    conn = init_db()
    conn.execute("DELETE FROM events WHERE metadata LIKE '%simulated%'")
    conn.execute("DELETE FROM tickets WHERE description LIKE '%[模拟]%'")
    conn.commit()

    generate_events(conn, NUM_EVENTS)
    generate_tickets(conn, NUM_TICKETS)
    cameras = generate_camera_sim(conn)
    conn.close()
    print_stats(sqlite3.connect(DB_PATH))
    print("\n  数据库: safety_events.db")
    print("  启动后端: uvicorn app:app --port 9101")
    print("  然后测试: curl http://localhost:9101/tickets")
    print("  或: curl http://localhost:9101/events?limit=20")
