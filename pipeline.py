"""
Axis Mundi — Data Pipeline
Handles: weather DB storage, nightly aggregation, model training, prediction loading.

Usage:
  python pipeline.py aggregate     # Run nightly aggregation for today
  python pipeline.py train         # Train prediction model from history
  python pipeline.py predict       # Generate tomorrow's predictions
  python pipeline.py status        # Show pipeline status
  
Normally called by the scheduler in the main script at 02:00 each night.
Can also be run manually.
"""

import json
import time
import pathlib
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from collections import defaultdict

try:
    from config import *
except ImportError:
    DB_PATH = "./axis_mundi.db"
    HISTORY_DIR = "./history"
    PREDICTIONS_DIR = "./predictions"
    MODELS_DIR = "./models"
    WEATHER_LAT = 50.0614
    WEATHER_LON = 19.9372
    WINDOW_MINUTES = 10

try:
    import requests
    REQ_OK = True
except ImportError:
    REQ_OK = False

# ═══════════════════════════════════════════════════════════════
# DATABASE SETUP
# ═══════════════════════════════════════════════════════════════

def init_db(db_path=DB_PATH):
    """Create all tables if they don't exist."""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    c.execute('''CREATE TABLE IF NOT EXISTS frames (
        FRAMEID INTEGER PRIMARY KEY,
        PATH TEXT,
        LOCATION TEXT,
        FRAMETIME REAL,
        PERSONCOUNT INTEGER,
        VEHICLECOUNT INTEGER DEFAULT 0
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS detections (
        FRAMEID INTEGER,
        PERSONID INTEGER,
        CLASSID INTEGER DEFAULT 0,
        CONFIDENCE REAL,
        BBOXTOPX REAL, BBOXTOPY REAL,
        BBOXBOTTOMX REAL, BBOXBOTTOMY REAL,
        VELX REAL, VELY REAL,
        PRIMARY KEY (FRAMEID, PERSONID)
    )''')
    
    c.execute('''CREATE TABLE IF NOT EXISTS weather (
        TIMESTAMP INTEGER PRIMARY KEY,
        TEMPERATURE REAL,
        HUMIDITY REAL,
        RAIN REAL,
        CLOUD_COVER INTEGER,
        WIND_SPEED REAL,
        WEATHER_CODE INTEGER
    )''')
    
    # Add VEHICLECOUNT column if missing (migration)
    try:
        c.execute("ALTER TABLE frames ADD COLUMN VEHICLECOUNT INTEGER DEFAULT 0")
    except:
        pass
    # Add CLASSID column if missing
    try:
        c.execute("ALTER TABLE detections ADD COLUMN CLASSID INTEGER DEFAULT 0")
    except:
        pass
    
    conn.commit()
    return conn


def store_weather(db_path, temp, humidity, rain, cloud, wind, code):
    """Store current weather reading. Thread-safe — opens own connection."""
    ts = int(time.time())
    try:
        conn = sqlite3.connect(str(db_path))
        conn.execute(
            "INSERT OR REPLACE INTO weather VALUES (?,?,?,?,?,?,?)",
            (ts, temp, humidity, rain, cloud, wind, code)
        )
        conn.commit()
        conn.close()
    except Exception as e:
        print(f"[DB] Weather store error: {e}")


# ═══════════════════════════════════════════════════════════════
# AGGREGATION — Turn raw frames into daily summary
# ═══════════════════════════════════════════════════════════════

def aggregate_day(date_str=None, db_path=DB_PATH):
    """
    Aggregate one day's data into a history JSON file.
    Groups by 10-minute windows, joins weather.
    
    Args:
        date_str: "2026-02-27" or None for yesterday
    """
    if date_str is None:
        yesterday = datetime.now() - timedelta(days=1)
        date_str = yesterday.strftime("%Y-%m-%d")
    
    print(f"\n[AGGREGATE] Processing {date_str}...")
    
    # Parse date range
    day_start = datetime.strptime(date_str, "%Y-%m-%d")
    day_end = day_start + timedelta(days=1)
    ts_start = day_start.timestamp()
    ts_end = day_end.timestamp()
    
    conn = sqlite3.connect(db_path)
    
    # Get all frames for this day
    frames = conn.execute(
        "SELECT FRAMEID, FRAMETIME, PERSONCOUNT, VEHICLECOUNT FROM frames "
        "WHERE FRAMETIME >= ? AND FRAMETIME < ? ORDER BY FRAMETIME",
        (ts_start, ts_end)
    ).fetchall()
    
    if not frames:
        print(f"  No data for {date_str}")
        conn.close()
        return None
    
    print(f"  {len(frames)} frames found")
    
    # Get detections with velocities
    frame_ids = [f[0] for f in frames]
    detections = {}
    # Batch query in chunks
    for i in range(0, len(frame_ids), 500):
        chunk = frame_ids[i:i+500]
        placeholders = ",".join("?" * len(chunk))
        rows = conn.execute(
            f"SELECT FRAMEID, PERSONID, CLASSID, CONFIDENCE, VELX, VELY "
            f"FROM detections WHERE FRAMEID IN ({placeholders})",
            chunk
        ).fetchall()
        for row in rows:
            fid = row[0]
            if fid not in detections:
                detections[fid] = []
            detections[fid].append(row)
    
    # Get weather for this day
    weather_rows = conn.execute(
        "SELECT TIMESTAMP, TEMPERATURE, HUMIDITY, RAIN, CLOUD_COVER, WIND_SPEED, WEATHER_CODE "
        "FROM weather WHERE TIMESTAMP >= ? AND TIMESTAMP < ?",
        (ts_start, ts_end)
    ).fetchall()
    conn.close()
    
    # Index weather by hour
    hourly_weather = {}
    for ts, temp, hum, rain, cloud, wind, code in weather_rows:
        hour = datetime.fromtimestamp(ts).hour
        hourly_weather[hour] = {
            "temperature": temp, "humidity": hum, "rain": rain,
            "cloud_cover": cloud, "wind_speed": wind, "weather_code": code
        }
    
    # Aggregate into windows
    window_sec = WINDOW_MINUTES * 60
    windows = defaultdict(lambda: {
        "person_counts": [], "vehicle_counts": [],
        "speeds": [], "directions": [], "confidences": []
    })
    
    for fid, ftime, pcount, vcount in frames:
        # Window key: minutes since midnight
        dt = datetime.fromtimestamp(ftime)
        window_key = (dt.hour * 60 + dt.minute) // WINDOW_MINUTES * WINDOW_MINUTES
        w = windows[window_key]
        w["person_counts"].append(pcount or 0)
        w["vehicle_counts"].append(vcount or 0)
        
        if fid in detections:
            for _, pid, cls, conf, vx, vy in detections[fid]:
                if vx is not None and vy is not None:
                    speed = (vx**2 + vy**2)**0.5
                    w["speeds"].append(speed)
                    if speed > 0.1:
                        import math
                        w["directions"].append(math.degrees(math.atan2(vy, vx)) % 360)
                w["confidences"].append(conf or 0)
    
    # Build output
    result = {
        "date": date_str,
        "day_of_week": day_start.strftime("%A"),
        "is_weekend": day_start.weekday() >= 5,
        "total_frames": len(frames),
        "windows": []
    }
    
    for wkey in sorted(windows.keys()):
        w = windows[wkey]
        hour = wkey // 60
        minute = wkey % 60
        
        pc = w["person_counts"]
        vc = w["vehicle_counts"]
        sp = w["speeds"]
        dr = w["directions"]
        
        window_data = {
            "time": f"{hour:02d}:{minute:02d}",
            "hour": hour,
            "minute": minute,
            "person_count_avg": float(np.mean(pc)) if pc else 0,
            "person_count_max": max(pc) if pc else 0,
            "person_count_min": min(pc) if pc else 0,
            "person_count_std": float(np.std(pc)) if len(pc) > 1 else 0,
            "vehicle_count_avg": float(np.mean(vc)) if vc else 0,
            "speed_avg": float(np.mean(sp)) if sp else 0,
            "speed_std": float(np.std(sp)) if len(sp) > 1 else 0,
            "direction_avg": float(np.mean(dr)) % 360 if dr else 0,
            "direction_std": float(np.std(dr)) if len(dr) > 1 else 0,
            "n_frames": len(pc),
        }
        
        # Add weather for this hour
        if hour in hourly_weather:
            window_data["weather"] = hourly_weather[hour]
        
        result["windows"].append(window_data)
    
    # Save
    out_dir = pathlib.Path(HISTORY_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"day_{date_str}.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2)
    
    print(f"  Saved: {out_path}")
    print(f"  {len(result['windows'])} windows, "
          f"avg {np.mean([w['person_count_avg'] for w in result['windows']]):.1f} people")
    
    return result


# ═══════════════════════════════════════════════════════════════
# TRAINING — Build prediction model from history
# ═══════════════════════════════════════════════════════════════

def train_model(history_dir=HISTORY_DIR, models_dir=MODELS_DIR):
    """
    Train a prediction model from accumulated daily history files.
    Needs at least 2 days of data.
    
    Features: hour, minute, weekday, is_weekend, temp, rain, cloud, wind,
              prev_window_count, rolling_avg_count
    Target: person_count_avg
    """
    import pickle
    
    hist_path = pathlib.Path(history_dir)
    files = sorted(hist_path.glob("day_*.json"))
    
    if len(files) < 2:
        print(f"[TRAIN] Need at least 2 days of history, have {len(files)}. Skipping.")
        return None
    
    print(f"\n[TRAIN] Loading {len(files)} days of history...")
    
    X, y = [], []
    
    for fpath in files:
        with open(fpath) as f:
            day = json.load(f)
        
        weekday = datetime.strptime(day["date"], "%Y-%m-%d").weekday()
        is_weekend = 1 if day.get("is_weekend") else 0
        
        prev_count = 0
        rolling = []
        
        for w in day["windows"]:
            wx = w.get("weather", {})
            features = [
                w["hour"],
                w["minute"],
                weekday,
                is_weekend,
                wx.get("temperature", 10),
                wx.get("rain", 0),
                wx.get("cloud_cover", 50),
                wx.get("wind_speed", 5),
                prev_count,
                np.mean(rolling[-6:]) if rolling else 0,  # last hour rolling avg
                w["speed_avg"],
                w["vehicle_count_avg"],
            ]
            X.append(features)
            y.append(w["person_count_avg"])
            
            prev_count = w["person_count_avg"]
            rolling.append(w["person_count_avg"])
    
    X = np.array(X)
    y = np.array(y)
    
    print(f"  {len(X)} training samples")
    
    # Try gradient boosting first, fall back to ridge regression
    try:
        from sklearn.ensemble import GradientBoostingRegressor
        from sklearn.model_selection import cross_val_score
        
        model = GradientBoostingRegressor(
            n_estimators=100, max_depth=4, learning_rate=0.1,
            min_samples_leaf=5, random_state=42
        )
        
        # Cross-validate
        if len(X) > 20:
            scores = cross_val_score(model, X, y, cv=min(5, len(files)), scoring='r2')
            print(f"  Cross-val R²: {scores.mean():.3f} ± {scores.std():.3f}")
        
        model.fit(X, y)
        
        # Feature importance
        feat_names = ["hour", "minute", "weekday", "is_weekend", "temp", "rain",
                      "cloud", "wind", "prev_count", "rolling_avg", "speed", "vehicles"]
        importances = sorted(zip(feat_names, model.feature_importances_), key=lambda x: -x[1])
        print("  Feature importance:")
        for name, imp in importances[:6]:
            print(f"    {name}: {imp:.3f}")
        
    except ImportError:
        print("  sklearn not available, using simple linear model")
        # Fallback: numpy least squares
        X_bias = np.column_stack([X, np.ones(len(X))])
        coeffs, _, _, _ = np.linalg.lstsq(X_bias, y, rcond=None)
        model = {"type": "linear", "coeffs": coeffs.tolist()}
    
    # Save model
    model_dir = pathlib.Path(models_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    model_path = model_dir / "predictor.pkl"
    
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    
    # Save metadata
    meta = {
        "trained_at": datetime.now().isoformat(),
        "n_days": len(files),
        "n_samples": len(X),
        "date_range": [files[0].stem.replace("day_", ""), files[-1].stem.replace("day_", "")],
    }
    with open(model_dir / "meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    
    print(f"  Model saved: {model_path}")
    return model


# ═══════════════════════════════════════════════════════════════
# PREDICTION — Generate tomorrow's forecast
# ═══════════════════════════════════════════════════════════════

def predict_tomorrow(models_dir=MODELS_DIR, predictions_dir=PREDICTIONS_DIR):
    """Generate hourly predictions for tomorrow using trained model + weather forecast."""
    import pickle
    
    model_path = pathlib.Path(models_dir) / "predictor.pkl"
    if not model_path.exists():
        print("[PREDICT] No trained model found. Run 'python pipeline.py train' first.")
        return None
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    
    # Fetch tomorrow's weather
    tomorrow = datetime.now() + timedelta(days=1)
    date_str = tomorrow.strftime("%Y-%m-%d")
    weekday = tomorrow.weekday()
    is_weekend = 1 if weekday >= 5 else 0
    
    print(f"\n[PREDICT] Generating predictions for {date_str} ({tomorrow.strftime('%A')})...")
    
    hourly_weather = {}
    if REQ_OK:
        try:
            url = (f"https://api.open-meteo.com/v1/forecast?"
                   f"latitude={WEATHER_LAT}&longitude={WEATHER_LON}"
                   f"&hourly=temperature_2m,rain,cloud_cover,wind_speed_10m"
                   f"&timezone=Europe/Warsaw&forecast_days=2"
                   f"&start_date={date_str}&end_date={date_str}")
            r = requests.get(url, timeout=10)
            d = r.json().get("hourly", {})
            temps = d.get("temperature_2m", [10]*24)
            rains = d.get("rain", [0]*24)
            clouds = d.get("cloud_cover", [50]*24)
            winds = d.get("wind_speed_10m", [5]*24)
            for h in range(min(24, len(temps))):
                hourly_weather[h] = {
                    "temperature": temps[h], "rain": rains[h],
                    "cloud_cover": clouds[h], "wind_speed": winds[h]
                }
            print(f"  Weather fetched for {date_str}")
        except Exception as e:
            print(f"  Weather fetch failed: {e}, using defaults")
    
    # Generate predictions for each 10-min window
    predictions = []
    prev_count = 5  # reasonable default
    rolling = []
    
    for hour in range(24):
        for minute in range(0, 60, WINDOW_MINUTES):
            wx = hourly_weather.get(hour, {"temperature": 10, "rain": 0, "cloud_cover": 50, "wind_speed": 5})
            
            features = np.array([[
                hour, minute, weekday, is_weekend,
                wx["temperature"], wx["rain"], wx["cloud_cover"], wx["wind_speed"],
                prev_count,
                np.mean(rolling[-6:]) if rolling else prev_count,
                1.0,  # default speed
                2.0,  # default vehicles
            ]])
            
            if isinstance(model, dict) and model.get("type") == "linear":
                features_bias = np.append(features[0], 1)
                pred = float(np.dot(features_bias, model["coeffs"]))
            else:
                pred = float(model.predict(features)[0])
            
            pred = max(0, pred)
            predictions.append({
                "time": f"{hour:02d}:{minute:02d}",
                "hour": hour,
                "minute": minute,
                "predicted_count": round(pred, 1),
                "weather": wx,
            })
            
            prev_count = pred
            rolling.append(pred)
    
    # Save
    pred_dir = pathlib.Path(predictions_dir)
    pred_dir.mkdir(parents=True, exist_ok=True)
    pred_path = pred_dir / f"prediction_{date_str}.json"
    
    output = {
        "date": date_str,
        "generated_at": datetime.now().isoformat(),
        "windows": predictions,
    }
    
    with open(pred_path, "w") as f:
        json.dump(output, f, indent=2)
    
    # Summary
    hourly_avg = defaultdict(list)
    for p in predictions:
        hourly_avg[p["hour"]].append(p["predicted_count"])
    
    print(f"  Saved: {pred_path}")
    print(f"  Hourly predictions:")
    for h in range(6, 24):
        avg = np.mean(hourly_avg[h])
        bar = "█" * int(avg / 2)
        print(f"    {h:02d}:00  {avg:5.1f}  {bar}")
    
    return output


def load_prediction(date_str=None, predictions_dir=PREDICTIONS_DIR):
    """Load prediction file for a given date. Returns dict hour->predicted_count or None."""
    if date_str is None:
        date_str = datetime.now().strftime("%Y-%m-%d")
    
    pred_path = pathlib.Path(predictions_dir) / f"prediction_{date_str}.json"
    if not pred_path.exists():
        return None
    
    with open(pred_path) as f:
        data = json.load(f)
    
    # Index by (hour, minute)
    lookup = {}
    for w in data.get("windows", []):
        lookup[(w["hour"], w["minute"])] = w["predicted_count"]
    
    return lookup


def get_current_prediction(lookup):
    """Get the prediction for the current time window."""
    if lookup is None:
        return None
    now = datetime.now()
    minute_window = (now.minute // WINDOW_MINUTES) * WINDOW_MINUTES
    return lookup.get((now.hour, minute_window))


# ═══════════════════════════════════════════════════════════════
# STATUS
# ═══════════════════════════════════════════════════════════════

def show_status():
    """Show pipeline status."""
    print("\n╔══════════════════════════════════════╗")
    print("║     AXIS MUNDI — Pipeline Status     ║")
    print("╚══════════════════════════════════════╝\n")
    
    # Database
    db = pathlib.Path(DB_PATH)
    if db.exists():
        conn = sqlite3.connect(str(db))
        frames = conn.execute("SELECT COUNT(*), MIN(FRAMETIME), MAX(FRAMETIME) FROM frames").fetchone()
        weather = conn.execute("SELECT COUNT(*) FROM weather").fetchone()
        conn.close()
        print(f"Database: {db} ({db.stat().st_size/1024/1024:.1f} MB)")
        print(f"  Frames: {frames[0]}")
        if frames[1]:
            print(f"  Range: {datetime.fromtimestamp(frames[1]):%Y-%m-%d %H:%M} → {datetime.fromtimestamp(frames[2]):%Y-%m-%d %H:%M}")
        print(f"  Weather readings: {weather[0]}")
    else:
        print(f"Database: NOT FOUND ({db})")
    
    # History
    hist = pathlib.Path(HISTORY_DIR)
    if hist.exists():
        days = sorted(hist.glob("day_*.json"))
        print(f"\nHistory: {len(days)} days")
        for d in days[-5:]:
            print(f"  {d.name}")
    else:
        print(f"\nHistory: none yet")
    
    # Model
    model_path = pathlib.Path(MODELS_DIR) / "predictor.pkl"
    if model_path.exists():
        meta_path = pathlib.Path(MODELS_DIR) / "meta.json"
        if meta_path.exists():
            with open(meta_path) as f:
                meta = json.load(f)
            print(f"\nModel: trained {meta.get('trained_at', '?')}")
            print(f"  {meta.get('n_days', '?')} days, {meta.get('n_samples', '?')} samples")
        else:
            print(f"\nModel: exists (no metadata)")
    else:
        print(f"\nModel: NOT TRAINED")
    
    # Predictions
    pred_dir = pathlib.Path(PREDICTIONS_DIR)
    if pred_dir.exists():
        preds = sorted(pred_dir.glob("prediction_*.json"))
        print(f"\nPredictions: {len(preds)} files")
        for p in preds[-3:]:
            print(f"  {p.name}")
    else:
        print(f"\nPredictions: none yet")
    
    today = datetime.now().strftime("%Y-%m-%d")
    pred = load_prediction(today)
    if pred:
        current = get_current_prediction(pred)
        print(f"\n  Current prediction for now: {current}")
    
    print()


# ═══════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python pipeline.py [aggregate|train|predict|status]")
        print("       python pipeline.py aggregate 2026-02-27  # specific date")
        sys.exit(1)
    
    cmd = sys.argv[1]
    
    if cmd == "aggregate":
        date = sys.argv[2] if len(sys.argv) > 2 else None
        aggregate_day(date)
    elif cmd == "train":
        train_model()
    elif cmd == "predict":
        predict_tomorrow()
    elif cmd == "status":
        show_status()
    elif cmd == "all":
        # Run full nightly pipeline
        aggregate_day()
        train_model()
        predict_tomorrow()
    else:
        print(f"Unknown command: {cmd}")
