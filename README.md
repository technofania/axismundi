# AXIS MUNDI

**Urban Metabolism — A real-time art installation interpreting the city as a living organism.**

Axis Mundi watches Kraków's Main Square through a public camera, tracks people and vehicles with computer vision, predicts crowd patterns using weather-correlated machine learning, and translates the city's "nervous system" into visual art, data, and live generative music.

The installation runs 24/7. It collects behavioral data, learns daily rhythms, and compares its predictions against reality. When the city behaves unexpectedly — a sudden crowd, an empty square at rush hour — the system enters "arrhythmia": visuals shift from gold to magenta, music becomes dissonant, and the prediction error becomes viscerally felt.

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        LIVE LOOP (24/7)                         │
│                                                                 │
│  Camera Stream ──→ YOLO Tracking ──→ SQLite Database            │
│       │                  │                │                     │
│       │            People + Cars     frames + detections        │
│       │                  │                                      │
│       │         ┌────────┴──────────┐                           │
│       │         │                   │                           │
│       ▼         ▼                   ▼                           │
│   OpenCV     OSC Output      Weather API                       │
│   Display    (port 9000       (Open-Meteo                      │
│   1920x1080   + port 9001)     every 5min)                     │
│       │         │                   │                           │
│       │    ┌────┴────┐         weather table                   │
│       │    │         │         in SQLite                        │
│       │    ▼         ▼                                          │
│       │   Touch    Music.py                                    │
│       │   Designer  (MIDI→Ableton)                             │
│       │                                                         │
│       ▼                                                         │
│   SINGLE WINDOW OUTPUT                                          │
│   ┌──────────────────┬────────────┐                             │
│   │  Camera + Overlay │  Cosmos    │                            │
│   │  (NIGREDO)        │  (RUBEDO)  │                            │
│   ├──────┬────────────┼──────┬─────┤                            │
│   │Heart │ Timeline   │Weath.│ ROI │                            │
│   └──────┴────────────┴──────┴─────┘                            │
├─────────────────────────────────────────────────────────────────┤
│                    NIGHTLY PIPELINE (02:00)                      │
│                                                                 │
│  1. AGGREGATE: Raw frames → daily JSON (10-min windows)         │
│  2. TRAIN: History + weather → GradientBoosting model           │
│  3. PREDICT: Model + tomorrow's forecast → hourly predictions   │
│                                                                 │
│  Next day: live count vs prediction file = real error signal    │
└─────────────────────────────────────────────────────────────────┘
```

---

## Files

### Core (keep these)

| File | Purpose |
|------|---------|
| `axis_mundi_v3.py` | **Main script.** Camera + YOLO + display + OSC + DB. Run this. |
| `config.py` | All tunable parameters in one place. Edit this to adjust quality, performance, detection, visuals. |
| `pipeline.py` | Data pipeline: aggregation, training, prediction. Used automatically at 02:00 and manually via CLI. |
| `axis_mundi_music.py` | Standalone MIDI composer. Receives OSC, sends MIDI to Ableton. Run alongside v3. |
| `axis_mundi.db` | SQLite database. Frames, detections, and weather readings accumulate here. |
| `yolo26n.pt` | YOLOv26 nano model (5.4 MB). Fast, good enough for installation. |
| `yolo26s.pt` | YOLOv26 small model (20 MB). More accurate, slower. Change in config.py if you want. |
| `ws_bridge.py` | WebSocket bridge for browser-based dashboard (optional). |
| `axis_mundi_dashboard.html` | Browser dashboard with alchemical design (optional, needs ws_bridge). |
| `requirements.txt` | Python dependencies. |

### Generated directories

| Directory | Contents |
|-----------|----------|
| `history/` | Daily aggregated JSON files (one per day of operation) |
| `predictions/` | Prediction JSON files (hourly forecasts per day) |
| `models/` | Trained prediction model (`predictor.pkl`) |
| `output/` | Screenshots |

---

## Display Layout (1920×1080)

```
┌──────────────────────────┬────────────────┐
│                          │                │
│   CAMERA + OVERLAY       │   COSMOS       │
│   ~1190 × 756            │   ~730 × 756   │
│   (NIGREDO)              │   (RUBEDO)     │
│                          │                │
├────────┬─────────────────┼────────┬───────┤
│ HEART  │    TIMELINE     │WEATHER │  ROI  │
│ + ERR  │    + VELOCITY   │+ 24h   │+ FLOW │
│ ~230px │    ~770px       │~460px  │~460px │
│        │                 │        │       │
│  324px │    324px        │ 324px  │ 324px │
└────────┴─────────────────┴────────┴───────┘
```

### What each element shows

**NIGREDO — Camera + Overlay (top left)**

The live camera feed from Kraków Main Square with subtle neural overlay:

- **Gold/magenta dots** — Each detected person, color shifts gold→magenta during anomaly
- **Rings around dots** — Detection confidence (fuller ring = higher confidence)
- **Cyan arrows** — Velocity vectors showing walking direction and speed
- **Colored trails** — Movement history (last ~14 frames), fading behind each person
- **Thin lines between people** — "Synaptic connections" drawn between people within 160px of each other. Represents social proximity as neural connections.
- **Ghost trails** — Fading gray traces where people recently left the frame (city's "short-term memory")
- **Vehicle labels** — Cars, motorcycles, buses, trucks labeled with class name
- **Magenta scanline** — Horizontal line sweeping across frame during anomaly states
- **Bottom left:** "NIGREDO" label
- **Bottom right:** Person count + vehicle count (e.g., "13p 4v")

**RUBEDO — Cosmic Neural Field (top right)**

Abstract visualization of the same data without camera — pure data art:

- **Membrane potential field** — 8×12 grid of cells that glow based on spatial density. Gold normally, magenta during anomaly. Temporal sine waves create organic "breathing" tissue effect.
- **Neuron blooms** — Each person rendered as a layered glow (3 concentric circles with optional Gaussian blur). Size scales with speed + confidence + prediction error. Creates the "cosmic" pink/gold orbs you see.
- **Curved synapses** — Bezier curves between nearby people with traveling light pulses (signal dots moving along the connection).
- **Velocity lines** — Green directional lines showing movement.
- **Trailing paths** — Movement history rendered as gradient lines.
- **Vehicles** — Smaller, dimmer dots with class-specific colors.
- **Scanline** — Magenta sweep during anomaly.

**HEART + ERROR (bottom left panel)**

The city's vital signs:

- **Pulsing heart** — Concentric rings that beat at a rate proportional to prediction error. Calm (50 BPM) when prediction is accurate, rapid (180 BPM) during anomaly. Color: gold (healthy) / magenta (anomalous).
- **Number in center** — Current person count.
- **"X vehicles"** — Current vehicle count.
- **Error bar** — Horizontal bar showing prediction error as percentage. Color: cyan (STABLE, <25%) / amber (DRIFT, 25-50%) / magenta (ANOMALY, >50%).
- **Speed** — Average pedestrian speed in pixels/second.
- **Direction** — Dominant flow direction in degrees.
- **Groups** — Number of detected clusters (2+ people within 55px).
- **Compass** — Small directional indicator showing dominant flow.
- **"pred: FILE/EST"** — Whether prediction comes from trained model file or simple estimate.

**TIMELINE (bottom center-left panel)**

Historical trends over the last ~600 frames (~10 minutes at 60fps):

- **Gold line** — Actual person count over time.
- **Blue dashed line** — Predicted person count.
- **Glowing dot** — Current position on the timeline.
- **A: / P:** — Current actual and predicted values.
- **VELOCITY** — Speed chart showing average pedestrian speed over time (cyan line).
- **VEHICLES** — Vehicle count over time (amber line).
- **count~speed r=X.XX** — Pearson correlation between count and speed. Green if positive (>0.3), red if negative (<-0.3). Shows whether more people = faster/slower movement.
- **trend: +/-N** — Count change over last 30 readings. Green if growing, red if declining.

**WEATHER (bottom center-right panel)**

Real-time weather from Open-Meteo API (updated every 5 minutes):

- **Temperature** — Current °C (amber).
- **Description** — Weather code (Clear, Rain, Overcast, etc.).
- **H:XX% R:XXmm W:XX km/h** — Humidity, rain, wind speed.
- **24h TEMP** — Orange line chart of today's hourly temperature forecast.
- **RAIN** — Bar chart of hourly precipitation.
- **CLOUD** — Line chart of hourly cloud cover.
- **"RAIN suppresses" / "DRY baseline"** — Simple correlation indicator: rain typically reduces pedestrian count.

**ROI — Spatial Analysis (bottom right panel)**

Spatial distribution of people across the frame:

- **NW / NE / SW / SE** — Person count in each quadrant. Brighter blue = more people. Shows crowd distribution patterns (e.g., most people near Cloth Hall vs St. Mary's).
- **DIR** — Dot plot of dominant direction over last 50 readings. Shows direction consistency.
- **DIST** — Histogram of person counts over last 80 readings. Shows distribution shape (normal vs bimodal = consistent vs variable).
- **Anom: X/100** — How many of the last 100 readings had prediction error >30%. Red if >15.
- **Still: XX%** — Percentage of detected people with near-zero velocity (standing still vs walking).

---

## Setup

### Hardware requirements

- **GPU:** NVIDIA with CUDA (tested on RTX 4090). Any 20-series or newer works.
- **RAM:** 16 GB minimum, 64 GB recommended for long-term operation.
- **CPU:** Any modern multi-core. Detection runs on GPU, display on CPU.
- **Display:** 1920×1080 TV or projector for installation.
- **Network:** Stable internet for camera stream + weather API.

### Software setup

```bash
# Python 3.10+
pip install ultralytics opencv-python-headless numpy
pip install python-osc requests scikit-learn
pip install mido python-rtmidi    # only for music module

# For MIDI: install loopMIDI (Windows) or IAC Driver (Mac)
# Create a virtual MIDI port named "Axis Mundi"
```

### Running

```bash
# Terminal 1 — Main installation (vision + display + data + OSC)
python axis_mundi_v3.py

# Terminal 2 — Music (optional, needs Ableton + loopMIDI)
python axis_mundi_music.py

# That's it. Everything else is automatic.
```

### Keyboard controls (in the OpenCV window)

| Key | Action |
|-----|--------|
| `q` | Quit |
| `f` | Toggle fullscreen (for TV/projector) |
| `s` | Save screenshot to `output/` |

---

## MIDI Music Setup (Ableton)

### Port setup

1. Install [loopMIDI](https://www.tobias-erichsen.de/software/loopmidi.html) (Windows) or use IAC Driver (Mac).
2. Create a virtual MIDI port. Name it anything containing "Axis" (e.g., "Axis Mundi").
3. In Ableton, go to Preferences → MIDI and enable the port for Track input.

### Channel assignments

The music script sends on **channels 1–5 only** (Ableton shows 16 but 6–16 are unused):

| Channel | Role | Suggested Ableton instrument |
|---------|------|------------------------------|
| Ch 1 | **Pad / Drone** — Slow chord changes, lush sustained harmony | Wavetable pad with long attack, hall reverb |
| Ch 2 | **Arpeggio / Pluck** — Rhythmic pulses, density follows crowd | Short pluck synth, delay, filter mapped to CC74 |
| Ch 3 | **Solo / Bell** — Sparse melodic notes, only during calm periods | Crystalline bell/marimba, high register |
| Ch 4 | **Sub-Bass** — Deep weight from vehicles | Sine-wave sub-bass, simple |
| Ch 5 | **Texture / Atmosphere** — Active during anomaly, dissonant | Granular synth, noise-based |

### CC Automation

The script sends continuous control changes for real-time modulation:

| CC | Channel | Controls | Driven by |
|----|---------|----------|-----------|
| CC1 (Mod Wheel) | Ch 1, Ch 5 | Tension / filter intensity | Prediction error |
| CC11 (Expression) | Ch 1 | Overall pad volume | Crowd size + breathing sine wave |
| CC74 (Filter Cutoff) | Ch 2 | Brightness of arpeggios | Pedestrian speed + error |
| CC91 (Reverb) | Ch 1, Ch 3 | Reverb wet amount | Rain intensity |
| CC93 (Chorus) | Ch 1 | Spatial width | Rain / weather ambiguity |

### Musical behavior

| City state | Scale | Progression | Character |
|------------|-------|-------------|-----------|
| Few people, low error | Lydian | I-IV-V-I | Dreamy, expansive, minimal |
| Normal activity, low error | Mixolydian | I-vi-IV-V | Warm, flowing |
| Fast movement | Dorian | I-vi-IV-V | Cool, urban groove |
| Rain | Minor | i-iv-vi-iii | Melancholy |
| High prediction error (30-50%) | Phrygian | I-vi-ii-V | Dark tension |
| Very high error (>50%) | Diminished | Unpredictable | Chaotic dissonance |

When an **anomaly onset** is detected (error crosses the threshold), the solo voice plays a dramatic high note and a dissonant cluster is triggered on the texture channel.

---

## Data Pipeline

### What gets stored

The system writes to `axis_mundi.db` continuously:

**frames table** — One row per sampled frame:
- `FRAMEID`, `FRAMETIME` (unix timestamp), `PERSONCOUNT`, `VEHICLECOUNT`

**detections table** — One row per detected object per frame:
- `FRAMEID`, `PERSONID` (tracking ID), `CLASSID` (0=person, 2=car, etc.), `CONFIDENCE`, bounding box coordinates, `VELX`, `VELY`

**weather table** — One row every 5 minutes:
- `TIMESTAMP`, `TEMPERATURE`, `HUMIDITY`, `RAIN`, `CLOUD_COVER`, `WIND_SPEED`, `WEATHER_CODE`

### Nightly pipeline (automatic at 02:00)

The main script automatically runs the pipeline once per night:

1. **Aggregate** — Takes yesterday's raw frames, groups them into 10-minute windows, joins with weather data, saves to `history/day_YYYY-MM-DD.json`. Each window contains: average/min/max/std person count, vehicle count, speed stats, direction stats, and weather at that hour.

2. **Train** — After 2+ days of history, trains a GradientBoosting regression model. Features: hour, minute, weekday, is_weekend, temperature, rain, cloud_cover, wind_speed, previous_window_count, rolling_average. Saves to `models/predictor.pkl`.

3. **Predict** — Fetches tomorrow's weather forecast from Open-Meteo, runs it through the model, generates hourly predictions. Saves to `predictions/prediction_YYYY-MM-DD.json`.

### Manual pipeline operations

```bash
python pipeline.py status                    # Show what data you have
python pipeline.py aggregate                 # Aggregate yesterday
python pipeline.py aggregate 2026-02-27      # Aggregate specific date
python pipeline.py train                     # Train model from all history
python pipeline.py predict                   # Generate tomorrow's predictions
python pipeline.py all                       # Run aggregate + train + predict
```

### How predictions reach the display

When `axis_mundi_v3.py` starts, it looks for `predictions/prediction_YYYY-MM-DD.json` for today's date. If found, the predicted count for the current 10-minute window replaces the simple statistical estimate. The display shows "pred: FILE" (vs "pred: EST" for the fallback estimator). The prediction error — actual vs predicted — drives the anomaly signal.

The first few days use a simple rolling-average estimator. Once the model is trained, predictions become weather-aware and time-aware. The installation gets smarter over time.

---

## OSC Output

The main script sends OSC messages to all targets defined in `config.py` (default: ports 9000 and 9001).

| Address | Type | Description |
|---------|------|-------------|
| `/axis/count` | int | Current person count |
| `/axis/vehicles` | int | Current vehicle count |
| `/axis/speed` | float | Average pedestrian speed |
| `/axis/direction` | float | Dominant flow direction (degrees) |
| `/axis/error` | float | Prediction error (0.0–1.0+) |
| `/axis/anomaly` | int | 1 if error > 0.3, else 0 |
| `/axis/rain` | float | Current rainfall (mm) |
| `/axis/temp` | float | Current temperature (°C) |
| `/axis/clusters` | int | Number of detected groups |
| `/axis/roi` | [int×4] | Person count in [NW, NE, SW, SE] quadrants |

---

## Configuration Reference

All parameters in `config.py`:

### Performance tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `PROC_W` / `PROC_H` | 800×450 | YOLO input resolution. Higher = better detection, slower. |
| `BLOOM_BLUR_SIZE` | 7 | Gaussian blur on cosmos blooms. 0=off (fastest), 15=dreamy. Must be odd. |
| `COS_BLOOM_LAYERS` | 3 | Glow layers per neuron. 1=minimal, 5=lush. |
| `DB_WRITE_INTERVAL` | 5 | Write to DB every N frames. Higher = less disk IO. |
| `CAM_MAX_CONNECTIONS` | 20 | Max synapse lines on camera. Lower = faster. |
| `COS_MAX_CONNECTIONS` | 15 | Max synapse lines in cosmos. |

### Visual tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `CAM_DARKEN` | 0.80 | Camera brightness. 0.5=dark, 1.0=original. |
| `CAM_TRAIL_LEN` | 14 | Trail length behind each person. |
| `CAM_GLOW_RADIUS` | 8 | Person glow size on camera. |
| `CAM_VECTOR_SCALE` | 0.40 | Velocity arrow length multiplier. |
| `COS_BLOOM_BASE` | 10 | Base bloom radius in cosmos. |
| `COS_BLOOM_GROWTH` | 0.5 | How much each bloom layer grows. |
| `COS_MEMBRANE_OPACITY` | 1.0 | Membrane field strength. 0=off, 2=strong. |
| `COS_SCANLINE` | True | Anomaly scanline effect on/off. |
| `COS_SIGNAL_PULSES` | True | Traveling light dots on synapses. |

### Detection

| Parameter | Default | Effect |
|-----------|---------|--------|
| `YOLO_MODEL` | "yolo26n.pt" | Model file. "yolo26s.pt" for better accuracy. |
| `YOLO_CONF` | 0.25 | Detection confidence threshold. |
| `DETECT_CLASSES` | [0,2,3,5,7] | COCO classes: 0=person, 2=car, 3=motorcycle, 5=bus, 7=truck. |

---

## Startup Checklist

After a computer restart, you need to start:

1. **loopMIDI** — Virtual MIDI port (if using music)
2. **Ableton Live** — With 5 MIDI tracks on channels 1–5 (if using music)
3. **TouchDesigner** — Receiving OSC on port 9000 (if using visuals)
4. **Terminal 1:** `python axis_mundi_v3.py`
5. **Terminal 2:** `python axis_mundi_music.py` (if using music)

Press `f` in the OpenCV window for fullscreen on the installation TV.

Everything else — weather fetching, database writes, nightly aggregation, model training, prediction generation — happens automatically.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "table frames has 6 columns but 5 values" | You're running old `axis_mundi_unified.py`. Use `axis_mundi_v3.py`. |
| "SQLite objects created in a thread" | Fixed in v3 — weather uses separate connection per thread. |
| Cosmos flickering / scan lines | Fixed in v3 — cosmos redrawn fresh each frame, no persistence buffer. |
| Low FPS | Reduce `BLOOM_BLUR_SIZE` to 0, reduce `PROC_W` to 640, reduce `COS_BLOOM_LAYERS` to 1. |
| Pixelated output | Make sure `DISPLAY_W=1920, DISPLAY_H=1080` in config.py. |
| Camera disconnects | Auto-reconnects after `RECONNECT_DELAY` seconds. |
| No weather data | Check internet. Open-Meteo is free, no API key needed. |
| MIDI port not found | loopMIDI must be running. Port name must contain "Axis". |
| No predictions | Need 2+ days of data. Run `python pipeline.py status` to check. |

---

## Credits

- Camera: Kraków Main Square public webcam
- Detection: Ultralytics YOLO
- Weather: [Open-Meteo](https://open-meteo.com/) (free, no key)
- Concept: Urban metabolism as living organism — the city breathes, has a heartbeat, a nervous system, and sometimes arrhythmia.
