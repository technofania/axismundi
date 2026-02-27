"""
Axis Mundi — Master Configuration
All tunable parameters in one place.
Adjust these to balance quality vs performance on your hardware.
"""

# ─── HARDWARE PROFILE ───
# Your setup: RTX 4090, i9-14900k, 64GB RAM
# This config is tuned for that. For weaker hardware, reduce DISPLAY and increase PROC values.

# ─── DISPLAY ───
DISPLAY_W = 1920          # Output resolution width (TV)
DISPLAY_H = 1080          # Output resolution height (TV)

# Layout ratios (camera takes ~62% width, cosmos ~38%)
CAM_RATIO = 0.62          # Camera width as fraction of total
BOT_RATIO = 0.30          # Bottom panel height as fraction of total

# ─── PROCESSING ───
PROC_W = 800              # YOLO input width (higher = better detection, slower)
PROC_H = 450              # YOLO input height (keep 16:9)
YOLO_MODEL = "yolo26n.pt" # Model file (n=nano fast, s=small accurate)
YOLO_CONF = 0.25          # Detection confidence threshold
YOLO_DEVICE = 0           # GPU device (0 = first GPU)

# Detection classes (COCO):
# 0=person, 1=bicycle, 2=car, 3=motorcycle, 5=bus, 7=truck
DETECT_CLASSES = [0, 2, 3, 5, 7]  # People + vehicles

# ─── STREAM ───
STREAM_URL = "https://hoktastream1.webcamera.pl/krakow_cam_9a3b91/krakow_cam_9a3b91.stream/playlist.m3u8"
STREAM_BUFFER = 1         # Buffer size (1 = lowest latency)
RECONNECT_DELAY = 3       # Seconds to wait before reconnecting

# ─── VISUAL EFFECTS (turn these up/down to taste) ───
# Camera overlay
CAM_DARKEN = 0.80         # Camera brightness (0.5=very dark, 1.0=no darkening)
CAM_TRAIL_LEN = 14        # Trail length per person (more = longer tails)
CAM_MAX_CONNECTIONS = 20   # Max synapse lines drawn on camera
CAM_CONNECTION_DIST = 160  # Max pixel distance for connections
CAM_GLOW_RADIUS = 8       # Person glow circle radius
CAM_VECTOR_SCALE = 0.40   # Velocity arrow length multiplier

# Cosmos panel
COS_FADE = 0.90           # Persistence fade (0.8=fast fade, 0.98=long trails) 
COS_BLOOM_LAYERS = 3      # Glow layers per neuron (1-5, more=prettier, slower)
COS_BLOOM_BASE = 10       # Base bloom radius
COS_BLOOM_GROWTH = 0.5    # How much each layer grows
COS_MAX_CONNECTIONS = 15   # Max synapse lines in cosmos
COS_TRAIL_LEN = 16        # Trail length in cosmos
COS_MEMBRANE_OPACITY = 1.0 # Membrane field strength (0=off, 2=strong)
COS_SCANLINE = True       # Anomaly scanline effect
COS_SIGNAL_PULSES = True  # Traveling light pulses on synapses

# ─── BLUR (the expensive operation — set 0 to disable) ───
# GaussianBlur on cosmos bloom. Makes it gorgeous but costs CPU.
# 0 = no blur (fastest), 3 = subtle, 7 = soft glow, 15 = dreamy
BLOOM_BLUR_SIZE = 7       # Blur kernel (must be odd or 0)

# ─── DATA COLLECTION ───
DB_PATH = "./axis_mundi.db"
DB_WRITE_INTERVAL = 5     # Write to DB every N frames (higher = less IO)
WEATHER_INTERVAL = 300    # Fetch weather every N seconds

# Weather API (Open-Meteo, free, no key needed)
WEATHER_LAT = 50.0614     # Kraków
WEATHER_LON = 19.9372
WEATHER_URL = (
    f"https://api.open-meteo.com/v1/forecast?"
    f"latitude={WEATHER_LAT}&longitude={WEATHER_LON}"
    f"&current=temperature_2m,relative_humidity_2m,rain,cloud_cover,wind_speed_10m,weather_code"
    f"&hourly=temperature_2m,rain,cloud_cover,wind_speed_10m"
    f"&timezone=Europe/Warsaw&forecast_days=2"
)

# ─── OSC OUTPUT ───
OSC_ENABLED = True
OSC_TARGETS = [
    ("127.0.0.1", 9000),   # TouchDesigner
    ("127.0.0.1", 9001),   # Music module (axis_mundi_music.py)
]
OSC_SEND_INTERVAL = 3     # Send OSC every N frames

# ─── PIPELINE PATHS ───
HISTORY_DIR = "./history"
PREDICTIONS_DIR = "./predictions"
MODELS_DIR = "./models"
SCREENSHOTS_DIR = "./output"

# ─── AGGREGATION ───
AGGREGATE_HOUR = 2        # Hour to run nightly aggregation (2 = 2:00 AM)
WINDOW_MINUTES = 10       # Aggregation window size in minutes
