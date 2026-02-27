#!/usr/bin/env python3
"""
Axis Mundi — Unified Installation v3
1080p, cars+people, tunable effects, weather DB, predictions, OSC, nightly pipeline.
"""

import cv2
import time
import math
import pathlib
import sqlite3
import threading
import numpy as np
from collections import deque
from datetime import datetime
from ultralytics import YOLO

from config import *
from pipeline import init_db, store_weather, load_prediction, get_current_prediction, aggregate_day

try:
    from pythonosc.udp_client import SimpleUDPClient
    OSC_OK = True
except ImportError:
    OSC_OK = False

try:
    import requests
    REQ_OK = True
except ImportError:
    REQ_OK = False

# ─── Derived layout ───
CAM_W = int(DISPLAY_W * CAM_RATIO)
CAM_H = int(DISPLAY_H * (1 - BOT_RATIO))
COS_W = DISPLAY_W - CAM_W
COS_H = CAM_H
BOT_H = DISPLAY_H - CAM_H

# BGR colors
GOLD=(76,168,201); GOLD_B=(90,197,232); GOLD_D=(30,80,100)
CYAN=(180,220,0); CYAN_D=(80,110,0)
MAG=(107,26,255); MAG_D=(55,15,130)
ALBEDO=(255,136,51); DIM=(60,60,80); DARK=(14,14,22); VOID=(8,8,14)
AMBER=(50,130,230); GREEN=(80,180,60); RED=(60,60,200)

VEHICLE_COLORS = {  # class_id -> color
    0: GOLD,    # person
    2: CYAN,    # car
    3: CYAN_D,  # motorcycle
    5: ALBEDO,  # bus
    7: AMBER,   # truck
}
CLASS_NAMES = {0:"person", 2:"car", 3:"moto", 5:"bus", 7:"truck"}


class Weather:
    def __init__(self, db_path):
        self.db_path = db_path
        self.temp = self.humidity = None
        self.rain = 0.0; self.cloud = 0; self.wind = 0.0; self.code = 0
        self.hourly_temp = []; self.hourly_rain = []; self.hourly_cloud = []
        self.last = 0; self.ok = False

    def fetch(self):
        if not REQ_OK: return
        try:
            d = requests.get(WEATHER_URL, timeout=8).json()
            c = d.get("current", {})
            self.temp = c.get("temperature_2m")
            self.humidity = c.get("relative_humidity_2m")
            self.rain = c.get("rain", 0)
            self.cloud = c.get("cloud_cover", 0)
            self.wind = c.get("wind_speed_10m", 0)
            self.code = c.get("weather_code", 0)
            h = d.get("hourly", {})
            self.hourly_temp = (h.get("temperature_2m") or [])[:24]
            self.hourly_rain = (h.get("rain") or [])[:24]
            self.hourly_cloud = (h.get("cloud_cover") or [])[:24]
            self.ok = True; self.last = time.time()
            # Store to DB (thread-safe)
            store_weather(self.db_path, self.temp, self.humidity,
                         self.rain, self.cloud, self.wind, self.code)
        except Exception as e:
            print(f"[WX] {e}")

    def refresh(self):
        if time.time() - self.last > WEATHER_INTERVAL:
            threading.Thread(target=self.fetch, daemon=True).start()

    @property
    def desc(self):
        codes = {0:"Clear",1:"Clear",2:"Cloudy",3:"Overcast",45:"Fog",
                 51:"Drizzle",61:"Rain",63:"Rain",65:"Heavy rain",71:"Snow",80:"Showers",95:"Storm"}
        return codes.get(self.code, f"WMO{self.code}")


class OSC:
    def __init__(self):
        self.clients = []
        if OSC_OK and OSC_ENABLED:
            for ip, port in OSC_TARGETS:
                try:
                    self.clients.append(SimpleUDPClient(ip, port))
                    print(f"[OSC] → {ip}:{port}")
                except: pass
        self.c = len(self.clients) > 0  # compat flag

    def send(self, data):
        if not self.clients: return
        try:
            for k, v in data.items():
                msg = f"/axis/{k}"
                val = v if isinstance(v, list) else float(v) if isinstance(v, (int, float)) else v
                for client in self.clients:
                    client.send_message(msg, val)
        except: pass


class Tracker:
    def __init__(self):
        self.trails = {}; self.vels = {}; self.prev = {}; self.prev_t = None
        self.ghosts = []; self.classes = {}  # tid -> class_id
        self.spd_h = deque(maxlen=600); self.cnt_h = deque(maxlen=600)
        self.dir_h = deque(maxlen=600); self.pred_h = deque(maxlen=600)
        self.vcnt_h = deque(maxlen=600)  # vehicle count history
        self.density = np.zeros((8, 12), dtype=np.float32)
        self.clusters = []; self.err = 0.0; self.anom = 0.0
        self.roi = {"NW":0,"NE":0,"SW":0,"SE":0}
        self.person_count = 0; self.vehicle_count = 0
        self.prediction = None  # from file
        self.pred_file_count = None  # current prediction from file

    def update(self, ids, xyxys, confs, cls_ids, t):
        dt = max((t - self.prev_t) if self.prev_t else .033, .001)
        self.density *= .93
        cur = {}; speeds = []; roi = {"NW":0,"NE":0,"SW":0,"SE":0}
        mx, my = PROC_W/2, PROC_H/2
        pc, vc = 0, 0

        for i, tid in enumerate(ids):
            x1,y1,x2,y2 = xyxys[i]; cx,cy = (x1+x2)/2,(y1+y2)/2
            cls = cls_ids[i]
            self.classes[tid] = cls
            cur[tid] = (cx, cy)

            if cls == 0: pc += 1
            else: vc += 1

            if tid not in self.trails: self.trails[tid] = deque(maxlen=max(CAM_TRAIL_LEN, COS_TRAIL_LEN))
            self.trails[tid].append((cx, cy))
            vx, vy = 0, 0
            if tid in self.prev:
                px, py = self.prev[tid]; vx, vy = (cx-px)/dt, (cy-py)/dt
            self.vels[tid] = (vx, vy)
            if cls == 0:  # only person speeds
                speeds.append(math.sqrt(vx*vx+vy*vy))
            gx = min(11, max(0, int(cx/PROC_W*12)))
            gy = min(7, max(0, int(cy/PROC_H*8)))
            self.density[gy, gx] += .3
            # ROI (persons only)
            if cls == 0:
                if cx<mx and cy<my: roi["NW"] += 1
                elif cx>=mx and cy<my: roi["NE"] += 1
                elif cx<mx: roi["SW"] += 1
                else: roi["SE"] += 1

        self.roi = roi; self.person_count = pc; self.vehicle_count = vc
        active = set(ids)
        for tid in list(self.trails.keys()):
            if tid not in active:
                tr = list(self.trails[tid])
                if len(tr) > 2: self.ghosts.append((tr, self.classes.get(tid, 0), 35))
                del self.trails[tid]; self.vels.pop(tid, None); self.classes.pop(tid, None)
        self.ghosts = [(tr, cls, l-1) for tr, cls, l in self.ghosts if l > 0]

        avg = np.mean(speeds) if speeds else 0
        self.spd_h.append(avg); self.cnt_h.append(pc); self.vcnt_h.append(vc)
        vxs = sum(self.vels[tid][0] for tid in active if self.classes.get(tid) == 0 and tid in self.vels)
        vys = sum(self.vels[tid][1] for tid in active if self.classes.get(tid) == 0 and tid in self.vels)
        dom = math.degrees(math.atan2(vys, vxs)) % 360 if abs(vxs)+abs(vys) > .1 else 0
        self.dir_h.append(dom)

        # Clusters (people only)
        ppos = [((xyxys[i][0]+xyxys[i][2])/2,(xyxys[i][1]+xyxys[i][3])/2) for i in range(len(ids)) if cls_ids[i]==0]
        if len(ppos) > 2:
            cls2=[]; used=set()
            for i in range(len(ppos)):
                if i in used: continue
                cl=[i]; used.add(i)
                for j in range(i+1,len(ppos)):
                    if j in used: continue
                    d=math.sqrt((ppos[i][0]-ppos[j][0])**2+(ppos[i][1]-ppos[j][1])**2)
                    if d<55: cl.append(j); used.add(j)
                if len(cl)>=2: cls2.append((np.mean([ppos[k][0] for k in cl]),np.mean([ppos[k][1] for k in cl]),len(cl)))
            self.clusters=cls2
        else: self.clusters=[]

        # Prediction error
        if self.pred_file_count is not None:
            pred = self.pred_file_count
        else:
            base = np.mean(list(self.cnt_h)[-30:]) if len(self.cnt_h) > 5 else pc
            pred = max(0, int(base + np.random.normal(0, 1)))
        self.pred_h.append(pred)
        raw = abs(pc - pred) / max(pred, 1)
        self.anom = self.anom*.88 + raw*.12
        self.err = self.anom

        self.prev = cur; self.prev_t = t
        return pc, avg, dom


class Renderer:
    def __init__(self):
        self.out = np.zeros((DISPLAY_H, DISPLAY_W, 3), dtype=np.uint8)
        self.cam_buf = np.zeros((CAM_H, CAM_W, 3), dtype=np.uint8)
        self.cos_buf = np.zeros((COS_H, COS_W, 3), dtype=np.uint8)
        self.bot_buf = np.zeros((BOT_H, DISPLAY_W, 3), dtype=np.uint8)
        self.bloom_buf = np.zeros((COS_H, COS_W, 3), dtype=np.uint8)
        self.mem_small = np.zeros((8, 12, 3), dtype=np.uint8)
        self.f = 0
        self.csx, self.csy = CAM_W/PROC_W, CAM_H/PROC_H
        self.ksx, self.ksy = COS_W/PROC_W, COS_H/PROC_H

    def render(self, raw, trk, ids, xyxys, confs, cls_ids, wx):
        self.f += 1; f = self.f; err = trk.err; an = err > .3

        # Camera
        cv2.resize(raw, (CAM_W, CAM_H), dst=self.cam_buf)
        np.multiply(self.cam_buf, CAM_DARKEN, out=self.cam_buf, casting='unsafe')
        self._cam(self.cam_buf, trk, ids, xyxys, confs, cls_ids, f, an)

        # Cosmos — FRESH each frame (no persistence flicker)
        self.cos_buf[:] = VOID
        self.bloom_buf[:] = 0
        self._cosmos(self.cos_buf, self.bloom_buf, trk, ids, xyxys, confs, cls_ids, f, err, an)
        # Apply bloom blur if enabled
        if BLOOM_BLUR_SIZE > 0:
            k = BLOOM_BLUR_SIZE | 1  # ensure odd
            cv2.GaussianBlur(self.bloom_buf, (k, k), 0, dst=self.bloom_buf)
        # Add bloom on top of cosmos
        np.add(self.cos_buf, np.minimum(self.bloom_buf, 80), out=self.cos_buf, casting='unsafe')

        # Bottom
        self.bot_buf[:] = VOID
        self._bottom(self.bot_buf, trk, f, err, an, wx)

        # Combine
        o = self.out
        o[:CAM_H, :CAM_W] = self.cam_buf
        o[:COS_H, CAM_W:] = self.cos_buf
        o[CAM_H:, :] = self.bot_buf
        cv2.line(o, (CAM_W, 0), (CAM_W, CAM_H), (16,16,24), 1)
        cv2.line(o, (0, CAM_H), (DISPLAY_W, CAM_H), (16,16,24), 1)
        cv2.putText(o, "A X I S   M U N D I", (DISPLAY_W//2-95, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .5, GOLD, 1, cv2.LINE_AA)
        cv2.putText(o, time.strftime("%H:%M:%S"), (DISPLAY_W-85, 18),
                    cv2.FONT_HERSHEY_SIMPLEX, .4, DIM, 1, cv2.LINE_AA)
        return o

    def _cam(self, cam, trk, ids, xyxys, confs, cls_ids, f, an):
        sx, sy = self.csx, self.csy
        # Ghosts
        for trail, cls, life in trk.ghosts:
            a = life/35
            col_base = VEHICLE_COLORS.get(cls, DIM)
            for j in range(max(0,len(trail)-6), len(trail)):
                if j==0: continue
                p1=(int(trail[j-1][0]*sx),int(trail[j-1][1]*sy))
                p2=(int(trail[j][0]*sx),int(trail[j][1]*sy))
                g=a*(j/len(trail))
                cv2.line(cam,p1,p2,tuple(int(c*g*.3) for c in col_base),1,cv2.LINE_AA)

        # Connections (persons only)
        pctrs=[(int((xyxys[i][0]+xyxys[i][2])/2*sx),int((xyxys[i][1]+xyxys[i][3])/2*sy))
               for i in range(len(ids)) if cls_ids[i]==0]
        nd=0
        for i in range(len(pctrs)):
            for j in range(i+1,len(pctrs)):
                if nd>=CAM_MAX_CONNECTIONS: break
                dx,dy=pctrs[i][0]-pctrs[j][0],pctrs[i][1]-pctrs[j][1]
                dsq=dx*dx+dy*dy
                if dsq<CAM_CONNECTION_DIST**2:
                    s=1-math.sqrt(dsq)/CAM_CONNECTION_DIST
                    col=(int(35*s),int(10*s),int(80*s)) if an else (int(12*s),int(60*s),int(75*s))
                    cv2.line(cam,pctrs[i],pctrs[j],col,1,cv2.LINE_AA); nd+=1

        # All tracked objects
        for i, tid in enumerate(ids):
            x1,y1,x2,y2=xyxys[i]; cx,cy=int((x1+x2)/2*sx),int((y1+y2)/2*sy)
            cls=cls_ids[i]; col=VEHICLE_COLORS.get(cls, DIM)
            is_person = cls==0

            # Trail
            if tid in trk.trails:
                trail=list(trk.trails[tid])[-(CAM_TRAIL_LEN if is_person else 8):]
                for j in range(1,len(trail)):
                    p=j/len(trail)
                    t1=(int(trail[j-1][0]*sx),int(trail[j-1][1]*sy))
                    t2=(int(trail[j][0]*sx),int(trail[j][1]*sy))
                    c=tuple(int(v*p*.3) for v in col)
                    cv2.line(cam,t1,t2,c,1,cv2.LINE_AA)

            # Velocity vector (persons)
            if is_person and tid in trk.vels:
                vx,vy=trk.vels[tid]; spd2=vx*vx+vy*vy
                if spd2>.1:
                    spd=math.sqrt(spd2); vl=min(spd*sx*CAM_VECTOR_SCALE,55)
                    ang=math.atan2(vy,vx)
                    cv2.arrowedLine(cam,(cx,cy),(int(cx+math.cos(ang)*vl),int(cy+math.sin(ang)*vl)),
                                   CYAN_D,1,cv2.LINE_AA,tipLength=.25)

            # Core
            r = 4 if is_person else 3
            cv2.circle(cam,(cx,cy),r,col,-1,cv2.LINE_AA)
            cv2.circle(cam,(cx,cy),r+3,col,1,cv2.LINE_AA)

            # Vehicle label
            if not is_person:
                lbl=CLASS_NAMES.get(cls,"?")
                cv2.putText(cam,lbl,(cx+8,cy-5),cv2.FONT_HERSHEY_SIMPLEX,.28,col,1,cv2.LINE_AA)

        cv2.putText(cam,"NIGREDO",(10,CAM_H-10),cv2.FONT_HERSHEY_SIMPLEX,.32,DIM,1,cv2.LINE_AA)
        cv2.putText(cam,f"{trk.person_count}p {trk.vehicle_count}v",(CAM_W-80,CAM_H-10),
                    cv2.FONT_HERSHEY_SIMPLEX,.32,GOLD,1,cv2.LINE_AA)
        if an:
            sly=int((f*(1+trk.err*2))%CAM_H)
            cv2.line(cam,(0,sly),(CAM_W,sly),MAG_D,1)

    def _cosmos(self, cos, bloom, trk, ids, xyxys, confs, cls_ids, f, err, an):
        sx, sy = self.ksx, self.ksy

        # Membrane
        dm=trk.density; self.mem_small[:]=0
        for gy in range(8):
            for gx in range(12):
                d=min(dm[gy,gx],2.0)/2.0
                w=math.sin(gx*.5+f*.025)*math.cos(gy*.4+f*.02)*.2+.5
                v=max(0,d*.7+w*.3)*COS_MEMBRANE_OPACITY
                if v>.06:
                    if an: self.mem_small[gy,gx]=(int(35+45*v),int(8+10*v),int(60+80*v))
                    else: self.mem_small[gy,gx]=(int(10+15*v),int(22+35*v),int(35+55*v))
        mem_up=cv2.resize(self.mem_small,(COS_W,COS_H),interpolation=cv2.INTER_LINEAR)
        np.maximum(cos,mem_up,out=cos)

        # Person centers for connections
        pctrs=[]
        for i in range(len(ids)):
            if cls_ids[i]!=0: continue
            cx=int((xyxys[i][0]+xyxys[i][2])/2*sx); cy=int((xyxys[i][1]+xyxys[i][3])/2*sy)
            pctrs.append((cx,cy,ids[i]))

        # Synapses
        nd=0
        for i in range(len(pctrs)):
            for j in range(i+1,len(pctrs)):
                if nd>=COS_MAX_CONNECTIONS: break
                dx,dy=pctrs[i][0]-pctrs[j][0],pctrs[i][1]-pctrs[j][1]; dsq=dx*dx+dy*dy
                if dsq<28000:
                    s=1-math.sqrt(dsq)/167
                    mx2=(pctrs[i][0]+pctrs[j][0])//2+int(math.sin(f*.05+i*.3)*10*s)
                    my2=(pctrs[i][1]+pctrs[j][1])//2+int(math.cos(f*.04)*7*s)
                    pts=np.array([[pctrs[i][0],pctrs[i][1]],[mx2,my2],[pctrs[j][0],pctrs[j][1]]],np.int32)
                    col=(int(25*s),int(6*s),int(55*s)) if an else (int(35*s),int(45*s),int(8*s))
                    cv2.polylines(cos,[pts],False,col,1,cv2.LINE_AA)
                    if COS_SIGNAL_PULSES:
                        t=(f*.02+i*.15)%1
                        px=int(pctrs[i][0]*(1-t)**2+2*mx2*t*(1-t)+pctrs[j][0]*t**2)
                        py=int(pctrs[i][1]*(1-t)**2+2*my2*t*(1-t)+pctrs[j][1]*t**2)
                        pc2=(80,30,160) if an else (100,130,0)
                        cv2.circle(cos,(px,py),int(2+s*2),pc2,-1,cv2.LINE_AA)
                    nd+=1

        # Neurons — draw glow on bloom buffer, core on cos
        for i,tid in enumerate(ids):
            if cls_ids[i]!=0: continue
            x1,y1,x2,y2=xyxys[i]; cx,cy=int((x1+x2)/2*sx),int((y1+y2)/2*sy)
            conf=confs[i]; spd=0; vx=vy=0
            if tid in trk.vels: vx,vy=trk.vels[tid]; spd=math.sqrt(vx*vx+vy*vy)

            # Trail
            if tid in trk.trails:
                trail=list(trk.trails[tid])[-COS_TRAIL_LEN:]
                for j in range(1,len(trail)):
                    p=j/len(trail)
                    t1=(int(trail[j-1][0]*sx),int(trail[j-1][1]*sy))
                    t2=(int(trail[j][0]*sx),int(trail[j][1]*sy))
                    c=(int(20*p),int(8*p),int(50*p)) if an else (int(10*p),int(30*p),int(40*p))
                    cv2.line(cos,t1,t2,c,max(1,int(p*2)),cv2.LINE_AA)

            # Bloom layers (drawn to bloom buffer for blur)
            pulse=.5+.5*math.sin(f*.07+(tid%20)*.5)
            for layer in range(COS_BLOOM_LAYERS):
                r=int((COS_BLOOM_BASE+spd*5+err*8+conf*4+pulse*3)*(1+layer*COS_BLOOM_GROWTH))
                a=max(1,int((30-layer*8)*(1 if an else .7)))
                if an: gc=(int(a*.4),int(a*.1),a)
                else: gc=(int(a*.15),int(a*.5),int(a*.65))
                cv2.circle(bloom,(cx,cy),r,gc,-1,cv2.LINE_AA)

            # Core on main
            core=(110,40,220) if an else (50,180,220)
            cv2.circle(cos,(cx,cy),3,core,-1,cv2.LINE_AA)

            # Velocity
            if spd>.3:
                vl=min(spd*sx*.3,40); ang=math.atan2(vy,vx)
                cv2.line(cos,(cx,cy),(int(cx+math.cos(ang)*vl),int(cy+math.sin(ang)*vl)),(80,120,0),1,cv2.LINE_AA)

        # Vehicles as smaller dimmer dots
        for i,tid in enumerate(ids):
            if cls_ids[i]==0: continue
            cx,cy=int((xyxys[i][0]+xyxys[i][2])/2*sx),int((xyxys[i][1]+xyxys[i][3])/2*sy)
            col=VEHICLE_COLORS.get(cls_ids[i],DIM)
            cv2.circle(cos,(cx,cy),4,tuple(c//3 for c in col),-1,cv2.LINE_AA)

        # Ghost trails
        for trail,cls,life in trk.ghosts:
            if cls!=0: continue
            a=life/35
            for j in range(max(0,len(trail)-5),len(trail)):
                if j==0: continue
                p1=(int(trail[j-1][0]*sx),int(trail[j-1][1]*sy))
                p2=(int(trail[j][0]*sx),int(trail[j][1]*sy))
                g=int(20*a); cv2.line(cos,p1,p2,(int(g*.6),g,int(g*.4)),1,cv2.LINE_AA)

        # Scanline
        if COS_SCANLINE and err>.2:
            sly=int((f*(1+err*2))%COS_H)
            cv2.line(cos,(0,sly),(COS_W,sly),(30,8,55),1)

        cv2.putText(cos,"RUBEDO",(10,COS_H-10),cv2.FONT_HERSHEY_SIMPLEX,.32,DIM,1,cv2.LINE_AA)

    def _bottom(self, bot, trk, f, err, an, wx):
        # Panel widths
        p1w=int(DISPLAY_W*.12); p2w=int(DISPLAY_W*.40); p3w=int(DISPLAY_W*.24); p4w=DISPLAY_W-p1w-p2w-p3w
        p2x=p1w; p3x=p1w+p2w; p4x=p3x+p3w
        for x in [p1w, p2x+p2w, p3x+p3w]:
            cv2.line(bot,(x,0),(x,BOT_H),(16,16,24),1)

        cnt=trk.person_count; vc=trk.vehicle_count

        # ── P1: Heart ──
        hx,hy=p1w//2,50; col=MAG if an else GOLD
        bpm=50+int(err*130); phase=(f*bpm/60*.12)%(2*math.pi)
        for i in range(3):
            rp=(phase+i*.5)%(2*math.pi); exp=max(0,math.sin(rp))
            r=int((16+i*5)*(1+exp*.25)); a=max(0,int((50-i*14)*exp))
            if a>3: cv2.circle(bot,(hx,hy),r,tuple(min(255,int(c*a/50)) for c in col),1,cv2.LINE_AA)
        cv2.circle(bot,(hx,hy),14,VOID,-1); cv2.circle(bot,(hx,hy),14,col,1,cv2.LINE_AA)
        ts=cv2.getTextSize(str(cnt),cv2.FONT_HERSHEY_SIMPLEX,.55,2)[0]
        cv2.putText(bot,str(cnt),(hx-ts[0]//2,hy+ts[1]//2),cv2.FONT_HERSHEY_SIMPLEX,.55,col,2,cv2.LINE_AA)
        cv2.putText(bot,"PEOPLE",(hx-22,hy+28),cv2.FONT_HERSHEY_SIMPLEX,.2,DIM,1,cv2.LINE_AA)
        # Vehicles
        cv2.putText(bot,f"{vc} vehicles",(10,hy+50),cv2.FONT_HERSHEY_SIMPLEX,.28,CYAN,1,cv2.LINE_AA)
        # Error bar
        pct=min(err*100,100); ec=MAG if pct>50 else AMBER if pct>25 else CYAN
        el="ANOMALY" if pct>50 else "DRIFT" if pct>25 else "STABLE"
        cv2.rectangle(bot,(10,120),(p1w-10,123),DARK,-1)
        cv2.rectangle(bot,(10,120),(10+int((p1w-20)*pct/100),123),ec,-1)
        cv2.putText(bot,el,(10,117),cv2.FONT_HERSHEY_SIMPLEX,.2,ec,1,cv2.LINE_AA)
        cv2.putText(bot,f"{pct:.0f}%",(p1w-40,137),cv2.FONT_HERSHEY_SIMPLEX,.25,ec,1,cv2.LINE_AA)
        # Speed + compass
        spd=list(trk.spd_h)[-1] if trk.spd_h else 0
        d=list(trk.dir_h)[-1] if trk.dir_h else 0
        cv2.putText(bot,f"{spd:.1f}m/s",(10,165),cv2.FONT_HERSHEY_SIMPLEX,.32,CYAN,1,cv2.LINE_AA)
        cv2.putText(bot,f"{d:.0f}deg",(10,185),cv2.FONT_HERSHEY_SIMPLEX,.25,DIM,1,cv2.LINE_AA)
        nc=len(trk.clusters)
        cv2.putText(bot,f"{nc}grp",(10,205),cv2.FONT_HERSHEY_SIMPLEX,.25,DIM,1,cv2.LINE_AA)
        # Compass
        ccx,ccy,ccr=p1w-30,190,16
        cv2.circle(bot,(ccx,ccy),ccr,DIM,1,cv2.LINE_AA)
        ca=math.radians(d-90)
        cv2.arrowedLine(bot,(ccx,ccy),(int(ccx+math.cos(ca)*ccr*.7),int(ccy+math.sin(ca)*ccr*.7)),CYAN,1,cv2.LINE_AA,tipLength=.35)
        # Prediction source
        src = "FILE" if trk.pred_file_count is not None else "EST"
        cv2.putText(bot,f"pred:{src}",(10,BOT_H-10),cv2.FONT_HERSHEY_SIMPLEX,.2,DIM,1,cv2.LINE_AA)

        # ── P2: Timeline ──
        ch=list(trk.cnt_h); ph=list(trk.pred_h); sh=list(trk.spd_h); vh=list(trk.vcnt_h)
        ox=p2x+15; tw=p2w-30
        if len(ch)>2:
            cv2.putText(bot,"TIMELINE",(ox,14),cv2.FONT_HERSHEY_SIMPLEX,.25,DIM,1,cv2.LINE_AA)
            allv=ch+ph; mx2=max(allv); mn2=min(allv); rng=mx2-mn2 or 1
            th2=int(BOT_H*.35)
            def tp(vals,by=22):
                return[(ox+int(i/max(len(vals)-1,1)*tw),by+th2-int((v-mn2)/rng*th2)) for i,v in enumerate(vals)]
            pp=tp(ph)
            for i in range(1,len(pp)):
                if i%3: cv2.line(bot,pp[i-1],pp[i],ALBEDO,1,cv2.LINE_AA)
            ap=tp(ch)
            for i in range(1,len(ap)): cv2.line(bot,ap[i-1],ap[i],GOLD,1,cv2.LINE_AA)
            if ap: cv2.circle(bot,ap[-1],3,GOLD_B,-1,cv2.LINE_AA)
            cv2.putText(bot,f"A:{ch[-1]}",(ox+tw-80,14),cv2.FONT_HERSHEY_SIMPLEX,.25,GOLD,1,cv2.LINE_AA)
            cv2.putText(bot,f"P:{ph[-1]}",(ox+tw-30,14),cv2.FONT_HERSHEY_SIMPLEX,.25,ALBEDO,1,cv2.LINE_AA)
        # Velocity
        vy_off=int(BOT_H*.45)
        if len(sh)>2:
            cv2.putText(bot,"VELOCITY",(ox,vy_off),cv2.FONT_HERSHEY_SIMPLEX,.2,DIM,1,cv2.LINE_AA)
            smx=max(sh) or 1
            for i in range(1,len(sh)):
                x1=ox+int((i-1)/max(len(sh)-1,1)*tw); x2=ox+int(i/max(len(sh)-1,1)*tw)
                y1=vy_off+8+40-int(sh[i-1]/smx*40); y2=vy_off+8+40-int(sh[i]/smx*40)
                cv2.line(bot,(x1,y1),(x2,y2),CYAN_D,1,cv2.LINE_AA)
        # Vehicle chart
        if len(vh)>2:
            cv2.putText(bot,"VEHICLES",(ox+tw//2,vy_off),cv2.FONT_HERSHEY_SIMPLEX,.2,DIM,1,cv2.LINE_AA)
            vmx=max(vh) or 1
            for i in range(1,len(vh)):
                x1=ox+tw//2+int((i-1)/max(len(vh)-1,1)*(tw//2)); x2=ox+tw//2+int(i/max(len(vh)-1,1)*(tw//2))
                y1=vy_off+8+40-int(vh[i-1]/vmx*40); y2=vy_off+8+40-int(vh[i]/vmx*40)
                cv2.line(bot,(x1,y1),(x2,y2),(140,100,30),1,cv2.LINE_AA)
        # Correlations
        cy2=BOT_H-15
        if len(ch)>20 and len(sh)>20:
            rc=np.array(ch[-50:],dtype=float); rs=np.array(sh[-50:],dtype=float)
            if np.std(rc)>0 and np.std(rs)>0:
                corr=np.corrcoef(rc,rs)[0,1]
                ccol=GREEN if corr>.3 else RED if corr<-.3 else DIM
                cv2.putText(bot,f"count~speed r={corr:.2f}",(ox,cy2),cv2.FONT_HERSHEY_SIMPLEX,.22,ccol,1,cv2.LINE_AA)
        if len(ch)>30:
            trend=ch[-1]-ch[-min(30,len(ch))]
            tc=GREEN if trend>2 else RED if trend<-2 else DIM
            cv2.putText(bot,f"trend:{'+'if trend>0 else''}{trend}",(ox+200,cy2),cv2.FONT_HERSHEY_SIMPLEX,.22,tc,1,cv2.LINE_AA)

        # ── P3: Weather ──
        wx_x=p3x+12
        cv2.putText(bot,"WEATHER",(wx_x,14),cv2.FONT_HERSHEY_SIMPLEX,.25,DIM,1,cv2.LINE_AA)
        if wx.ok:
            cv2.putText(bot,f"{wx.temp:.1f}C",(wx_x,38),cv2.FONT_HERSHEY_SIMPLEX,.45,AMBER,1,cv2.LINE_AA)
            cv2.putText(bot,wx.desc,(wx_x+80,38),cv2.FONT_HERSHEY_SIMPLEX,.28,DIM,1,cv2.LINE_AA)
            cv2.putText(bot,f"H:{wx.humidity}% R:{wx.rain}mm W:{wx.wind}km/h",(wx_x,55),
                        cv2.FONT_HERSHEY_SIMPLEX,.22,RED if (wx.rain or 0)>0 else DIM,1,cv2.LINE_AA)
            ht=wx.hourly_temp; cw2=p3w-25
            if len(ht)>2:
                cv2.putText(bot,"24h TEMP",(wx_x,72),cv2.FONT_HERSHEY_SIMPLEX,.18,DIM,1,cv2.LINE_AA)
                tmx,tmn=max(ht),min(ht); trng=tmx-tmn or 1
                for i in range(1,len(ht)):
                    x1=wx_x+int((i-1)/(len(ht)-1)*cw2); x2=wx_x+int(i/(len(ht)-1)*cw2)
                    y1=80+35-int((ht[i-1]-tmn)/trng*35); y2=80+35-int((ht[i]-tmn)/trng*35)
                    cv2.line(bot,(x1,y1),(x2,y2),AMBER,1,cv2.LINE_AA)
            hr=wx.hourly_rain
            if len(hr)>2 and max(hr)>0:
                cv2.putText(bot,"RAIN",(wx_x,125),cv2.FONT_HERSHEY_SIMPLEX,.18,DIM,1,cv2.LINE_AA)
                rmx=max(hr) or 1
                for i in range(len(hr)):
                    x=wx_x+int(i/max(len(hr)-1,1)*cw2); h2=int(hr[i]/rmx*25)
                    if h2>0: cv2.line(bot,(x,155),(x,155-h2),(200,120,50),2)
            hc=wx.hourly_cloud
            if len(hc)>2:
                cv2.putText(bot,"CLOUD",(wx_x,168),cv2.FONT_HERSHEY_SIMPLEX,.18,DIM,1,cv2.LINE_AA)
                cmx=max(hc) or 1
                for i in range(1,len(hc)):
                    x1=wx_x+int((i-1)/(len(hc)-1)*cw2); x2=wx_x+int(i/(len(hc)-1)*cw2)
                    y1=178+25-int(hc[i-1]/cmx*25); y2=178+25-int(hc[i]/cmx*25)
                    cv2.line(bot,(x1,y1),(x2,y2),(90,90,70),1,cv2.LINE_AA)
            impact="RAIN->fewer" if (wx.rain or 0)>.5 else "DRY baseline"
            cv2.putText(bot,impact,(wx_x,BOT_H-12),cv2.FONT_HERSHEY_SIMPLEX,.22,
                        RED if (wx.rain or 0)>.5 else GREEN,1,cv2.LINE_AA)
        else:
            cv2.putText(bot,"Fetching...",(wx_x,38),cv2.FONT_HERSHEY_SIMPLEX,.3,DIM,1,cv2.LINE_AA)

        # ── P4: ROI + Stats ──
        rx=p4x+12
        cv2.putText(bot,"SPATIAL",(rx,14),cv2.FONT_HERSHEY_SIMPLEX,.25,DIM,1,cv2.LINE_AA)
        roi=trk.roi; mxr=max(roi.values()) or 1
        for idx,(k,v) in enumerate(roi.items()):
            bx=rx+(idx%2)*50; by=22+(idx//2)*40; n=v/mxr
            bg=(int(140*n*.1+8),int(90*n*.1+8),int(18*n*.1+8))
            cv2.rectangle(bot,(bx,by),(bx+46,by+36),bg,-1)
            cv2.rectangle(bot,(bx,by),(bx+46,by+36),(int(140*n*.2+12),int(90*n*.2+12),18),1)
            cv2.putText(bot,k,(bx+3,by+11),cv2.FONT_HERSHEY_SIMPLEX,.2,DIM,1,cv2.LINE_AA)
            cv2.putText(bot,str(v),(bx+13,by+30),cv2.FONT_HERSHEY_SIMPLEX,.4,ALBEDO,1,cv2.LINE_AA)
        # Histogram
        ch2=list(trk.cnt_h)[-80:]
        if len(ch2)>10:
            cv2.putText(bot,"DIST",(rx,112),cv2.FONT_HERSHEY_SIMPLEX,.18,DIM,1,cv2.LINE_AA)
            bins=np.histogram(ch2,bins=10)[0]; bmx=max(bins) or 1
            for i,b in enumerate(bins):
                bh=int(b/bmx*28); bx2=rx+i*9
                if bh>0: cv2.rectangle(bot,(bx2,148-bh),(bx2+7,148),GOLD_D,-1)
        # Stationarity
        if trk.vels:
            still=sum(1 for tid,v in trk.vels.items() if trk.classes.get(tid)==0 and math.sqrt(v[0]**2+v[1]**2)<.5)
            tot=sum(1 for tid in trk.vels if trk.classes.get(tid)==0)
            pct_still=still/max(tot,1)*100
            cv2.putText(bot,f"Still:{pct_still:.0f}%",(rx,168),cv2.FONT_HERSHEY_SIMPLEX,.22,DIM,1,cv2.LINE_AA)
        # Anomaly rate
        if len(trk.cnt_h)>10 and len(trk.pred_h)>10:
            errs=[abs(c-p)/max(p,1) for c,p in zip(list(trk.cnt_h)[-100:],list(trk.pred_h)[-100:])]
            na=sum(1 for e in errs if e>.3)
            cv2.putText(bot,f"Anom:{na}/100",(rx,188),cv2.FONT_HERSHEY_SIMPLEX,.22,MAG if na>15 else DIM,1,cv2.LINE_AA)


def nightly_job(last_date):
    """Run nightly aggregation + training + prediction."""
    today = datetime.now().strftime("%Y-%m-%d")
    if today == last_date:
        return last_date
    now = datetime.now()
    if now.hour == AGGREGATE_HOUR and now.minute < 5:
        print(f"\n[NIGHTLY] Running pipeline at {now:%H:%M}...")
        try:
            aggregate_day()  # yesterday
            from pipeline import train_model, predict_tomorrow
            train_model()
            predict_tomorrow()
        except Exception as e:
            print(f"[NIGHTLY] Error: {e}")
        return today
    return last_date


def main():
    for d in [HISTORY_DIR, PREDICTIONS_DIR, MODELS_DIR, SCREENSHOTS_DIR]:
        pathlib.Path(d).mkdir(parents=True, exist_ok=True)

    print("Loading model..."); model = YOLO(YOLO_MODEL)
    print("Connecting..."); cap = cv2.VideoCapture(STREAM_URL)
    if not cap.isOpened(): print("ERROR!"); return
    cap.set(cv2.CAP_PROP_BUFFERSIZE, STREAM_BUFFER); print("Stream OK!")

    conn = init_db(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT IFNULL(MAX(FRAMEID),0) FROM frames"); fid = c.fetchone()[0]

    trk = Tracker(); ren = Renderer()
    osc = OSC(); wx = Weather(DB_PATH); wx.fetch()
    fps_buf = deque(maxlen=30); osc_tick = 0
    last_nightly = ""

    # Load today's predictions if available
    pred_lookup = load_prediction()
    if pred_lookup:
        print(f"[PRED] Loaded predictions for today ({len(pred_lookup)} windows)")

    cv2.namedWindow("Axis Mundi", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Axis Mundi", DISPLAY_W, DISPLAY_H)
    print(f"\n=== AXIS MUNDI v3 | {DISPLAY_W}x{DISPLAY_H} | q:quit f:fullscreen s:screenshot ===")
    print(f"=== OSC:{'ON' if osc.c else 'OFF'} | WX:{'OK' if wx.ok else 'wait'} | "
          f"Classes:{[CLASS_NAMES.get(c,c) for c in DETECT_CLASSES]} | Blur:{BLOOM_BLUR_SIZE} ===\n")

    try:
        while True:
            t0 = time.time()
            if not cap.grab():
                cap.release(); time.sleep(RECONNECT_DELAY)
                cap = cv2.VideoCapture(STREAM_URL)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, STREAM_BUFFER); continue
            ret, raw = cap.retrieve()
            if not ret: continue
            ft = time.time()
            small = cv2.resize(raw, (PROC_W, PROC_H))

            results = model.track(small, classes=DETECT_CLASSES, conf=YOLO_CONF,
                                  persist=True, verbose=False, tracker="bytetrack.yaml",
                                  device=YOLO_DEVICE)
            boxes = results[0].boxes
            if boxes.id is not None:
                ids = boxes.id.int().cpu().tolist()
                confs = boxes.conf.float().cpu().tolist()
                xyxys = boxes.xyxy.float().cpu().tolist()
                cls_ids = boxes.cls.int().cpu().tolist()
            else:
                ids, confs, xyxys, cls_ids = [], [], [], []

            # Update prediction from file
            if pred_lookup:
                trk.pred_file_count = get_current_prediction(pred_lookup)

            cnt, spd, dr = trk.update(ids, xyxys, confs, cls_ids, ft)
            out = ren.render(raw, trk, ids, xyxys, confs, cls_ids, wx)

            fps_buf.append(time.time()-t0)
            fps = 1/(sum(fps_buf)/len(fps_buf))
            cv2.putText(out, f"FPS:{fps:.0f}", (5, DISPLAY_H-5),
                        cv2.FONT_HERSHEY_SIMPLEX, .28, DIM, 1, cv2.LINE_AA)

            # DB
            fid += 1
            if fid % DB_WRITE_INTERVAL == 0 and (cnt > 0 or trk.vehicle_count > 0):
                c.execute("INSERT INTO frames VALUES(?,?,?,?,?,?)",
                          (fid, "live", "Krakow", ft, cnt, trk.vehicle_count))
                for i, tid in enumerate(ids):
                    vx, vy = trk.vels.get(tid, (0,0))
                    c.execute("INSERT OR IGNORE INTO detections VALUES(?,?,?,?,?,?,?,?,?,?)",
                              (fid, tid, cls_ids[i], confs[i], *xyxys[i], vx, vy))
                conn.commit()

            # OSC
            osc_tick += 1
            if osc_tick % OSC_SEND_INTERVAL == 0:
                osc.send({"count": cnt, "vehicles": trk.vehicle_count,
                          "speed": spd, "direction": dr, "error": trk.err,
                          "anomaly": 1 if trk.err>.3 else 0,
                          "rain": wx.rain or 0, "temp": wx.temp or 0,
                          "clusters": len(trk.clusters),
                          "roi": [trk.roi.get(k,0) for k in ("NW","NE","SW","SE")]})

            wx.refresh()

            # Nightly pipeline
            last_nightly = nightly_job(last_nightly)
            # Reload predictions at midnight
            if fid % 1000 == 0:
                pred_lookup = load_prediction() or pred_lookup

            cv2.imshow("Axis Mundi", out)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'): break
            elif key == ord('f'):
                p = cv2.getWindowProperty("Axis Mundi", cv2.WND_PROP_FULLSCREEN)
                cv2.setWindowProperty("Axis Mundi", cv2.WND_PROP_FULLSCREEN,
                    cv2.WINDOW_NORMAL if p == cv2.WINDOW_FULLSCREEN else cv2.WINDOW_FULLSCREEN)
            elif key == ord('s'):
                fn = f"{SCREENSHOTS_DIR}/screenshot_{int(time.time())}.jpg"
                cv2.imwrite(fn, out); print(f"Saved: {fn}")

            if fid % 200 == 0:
                print(f"[{time.strftime('%H:%M:%S')}] #{fid} | {cnt}p {trk.vehicle_count}v | "
                      f"{fps:.0f}fps | err:{trk.err:.0%} | {wx.desc} {wx.temp}C")

    except KeyboardInterrupt: pass
    finally:
        cap.release(); cv2.destroyAllWindows(); conn.close()
        print(f"\n{fid} frames → {DB}")


if __name__ == "__main__":
    main()
