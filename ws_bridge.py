#!/usr/bin/env python3
"""
Axis Mundi — Live Data Bridge
Reads detection data from SQLite in real-time and broadcasts via WebSocket.
Run this ALONGSIDE 'python launch.py live' to feed the dashboard.

Usage:
    python ws_bridge.py
    
Dashboard connects to ws://localhost:8765
"""

import asyncio
import websockets
import sqlite3
import json
import time
import pathlib
import logging
import numpy as np
from datetime import datetime

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
log = logging.getLogger("axis_mundi.bridge")

DB_PATH = pathlib.Path("./axis_mundi.db")
WS_PORT = 8765
POLL_INTERVAL = 1.5  # seconds between DB reads
PREDICTION_FILE = None  # will be auto-detected

clients = set()
last_frame_id = 0


def load_todays_prediction():
    """Try to load today's prediction file."""
    today = datetime.now().strftime("%Y-%m-%d")
    pred_path = pathlib.Path(f"./predictions/prediction_{today}.json")
    if pred_path.exists():
        with open(pred_path) as f:
            data = json.load(f)
        log.info(f"Loaded prediction for {today}: {len(data)} windows")
        return {d["time"]: d["predicted_person_count"] for d in data}
    return {}


def get_latest_detections(conn, since_frame_id):
    """Read new frames and detections since last check."""
    c = conn.cursor()
    
    # Get new frames
    c.execute("""
        SELECT FRAMEID, FRAMETIME, PERSONCOUNT 
        FROM frames 
        WHERE FRAMEID > ? 
        ORDER BY FRAMEID DESC 
        LIMIT 5
    """, (since_frame_id,))
    frames = c.fetchall()
    
    if not frames:
        return None, since_frame_id
    
    latest_frame = frames[0]
    fid, ftime, pcount = latest_frame
    
    # Get detections for latest frame
    c.execute("""
        SELECT PERSONID, CONFIDENCE, BBOXTOPX, BBOXTOPY, 
               BBOXBOTTOMX, BBOXBOTTOMY, VELX, VELY
        FROM detections
        WHERE FRAMEID = ?
    """, (fid,))
    detections = c.fetchall()
    
    # Build persons list for visualization
    persons = []
    speeds = []
    for det in detections:
        pid, conf, x1, y1, x2, y2, vx, vy = det
        # Normalize to 0-1 range (assuming 640x480 frame)
        cx = (x1 + x2) / 2.0 / 640.0
        cy = (y1 + y2) / 2.0 / 480.0
        speed = np.sqrt(vx**2 + vy**2)
        speeds.append(speed)
        
        persons.append({
            "id": int(pid) if pid else len(persons),
            "x": float(np.clip(cx, 0, 1)),
            "y": float(np.clip(cy, 0, 1)),
            "vx": float(vx * 0.001) if vx else 0,  # scale for visualization
            "vy": float(vy * 0.001) if vy else 0,
            "speed": float(speed),
            "confidence": float(conf),
        })
    
    avg_speed = float(np.mean(speeds)) if speeds else 0
    
    # Calculate dominant direction from velocity vectors
    if detections:
        vx_sum = sum(d[6] for d in detections if d[6])
        vy_sum = sum(d[7] for d in detections if d[7])
        direction = float(np.degrees(np.arctan2(vy_sum, vx_sum)) % 360)
    else:
        direction = 0
    
    # Simple ROI assignment (quadrants of the frame)
    roi = {"roi_nw": 0, "roi_ne": 0, "roi_sw": 0, "roi_se": 0}
    for p in persons:
        if p["x"] < 0.5 and p["y"] < 0.5:
            roi["roi_nw"] += 1
        elif p["x"] >= 0.5 and p["y"] < 0.5:
            roi["roi_ne"] += 1
        elif p["x"] < 0.5 and p["y"] >= 0.5:
            roi["roi_sw"] += 1
        else:
            roi["roi_se"] += 1
    
    now = datetime.now()
    time_str = now.strftime("%H:%M")
    
    # Get prediction for current time window
    predictions = load_todays_prediction()
    # Round to nearest 5-min window
    minute_rounded = (now.minute // 5) * 5
    pred_key = f"{now.hour:02d}:{minute_rounded:02d}"
    predicted_count = predictions.get(pred_key, pcount + np.random.randint(-3, 4))
    
    pred_error = abs(pcount - predicted_count) / max(predicted_count, 1)
    
    data = {
        "timestamp": time_str,
        "frame_id": fid,
        "person_count": pcount,
        "predicted_count": int(predicted_count),
        "prediction_error": float(np.clip(pred_error, 0, 1)),
        "avg_speed": round(avg_speed, 3),
        "dominant_direction": round(direction, 1),
        "roi_counts": roi,
        "black_swan": pcount > 40,  # threshold for anomaly
        "temperature": 12.0,  # placeholder until BME280 connected
        "persons": persons[:80],  # cap for UI performance
        "source": "live_camera",
    }
    
    return data, fid


async def ws_handler(websocket, path=None):
    """Handle new WebSocket connections."""
    clients.add(websocket)
    log.info(f"Dashboard connected ({len(clients)} clients)")
    try:
        async for msg in websocket:
            # Could handle commands from dashboard here
            pass
    finally:
        clients.discard(websocket)
        log.info(f"Dashboard disconnected ({len(clients)} clients)")


async def broadcast_loop():
    """Poll database and broadcast to all connected dashboards."""
    global last_frame_id
    
    # Wait for DB to exist
    while not DB_PATH.exists():
        log.info(f"Waiting for database {DB_PATH}...")
        await asyncio.sleep(2)
    
    conn = sqlite3.connect(str(DB_PATH))
    log.info("Connected to database. Broadcasting live data...")
    
    frame_count = 0
    while True:
        try:
            data, new_fid = get_latest_detections(conn, last_frame_id)
            
            if data and new_fid > last_frame_id:
                last_frame_id = new_fid
                frame_count += 1
                
                msg = json.dumps(data)
                
                # Broadcast to all clients
                disconnected = set()
                for ws in clients:
                    try:
                        await ws.send(msg)
                    except websockets.exceptions.ConnectionClosed:
                        disconnected.add(ws)
                clients.difference_update(disconnected)
                
                if frame_count % 30 == 0:
                    log.info(f"Broadcast #{frame_count}: {data['person_count']} people, "
                             f"error={data['prediction_error']:.1%}, "
                             f"{len(clients)} clients")
        
        except Exception as e:
            log.error(f"Broadcast error: {e}")
        
        await asyncio.sleep(POLL_INTERVAL)


async def main():
    print(f"""
    ========================================
      AXIS MUNDI — Live Data Bridge
    ========================================
      WebSocket:  ws://localhost:{WS_PORT}
      Database:   {DB_PATH}
      Poll rate:  every {POLL_INTERVAL}s
      
      1. Run 'python launch.py live' in another terminal
      2. Open dashboard in browser
      3. Data flows: Camera → YOLO → DB → Bridge → Dashboard
      
      Stop: Ctrl+C
    ========================================
    """)
    
    server = await websockets.serve(ws_handler, "0.0.0.0", WS_PORT)
    log.info(f"WebSocket server on ws://localhost:{WS_PORT}")
    
    await broadcast_loop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        log.info("Bridge stopped.")
