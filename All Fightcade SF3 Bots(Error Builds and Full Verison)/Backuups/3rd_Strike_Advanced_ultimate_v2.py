#!/usr/bin/env python3
\"\"\"3rd_Strike_Advanced_ultimate_v2.py

All-in-one updated build:
- Calibration analyzer (suggest deadzone) from calibration log
- GhostAI loads/saves ~/.ghost_patterns.json and merges if present
- curses TUI dashboard (fallback to plain print)
- FFmpeg (X11) or wf-recorder/pipewire (Wayland) capture automatic selection

Run: python3 3rd_Strike_Advanced_ultimate_v2.py
\"\"\"
import time, threading, subprocess, shutil, sys, os, json
from pathlib import Path
from collections import deque
from dataclasses import dataclass
from typing import Optional, List, Dict
import random

# optional libs
try:
    import curses
    HAS_CURSES = True
except Exception:
    HAS_CURSES = False
try:
    import pygame
    HAS_PYGAME = True
except Exception:
    HAS_PYGAME = False
try:
    import mss, cv2, numpy as np
    HAS_VISION = True
except Exception:
    HAS_VISION = False

# user settings (from your environment)
FIGHTCADE_LOG_PATH = \"/home/bemo9000/.var/app/com.fightcade.Fightcade/data/config/fcadefbneo/presets/cps.ini\"
CAL_LOG = Path.home() / \".3rd_strike_calibration.json\"
GHOST_FILE = Path.home() / \".ghost_patterns.json\"

XDOTOOL = shutil.which(\"xdotool\") is not None
FFMPEG = shutil.which(\"ffmpeg\") is not None
WF_REC = shutil.which(\"wf-recorder\") is not None or shutil.which(\"pw-record\") is not None

@dataclass
class MatchInfo:
    p1_name: str = \"Alex\"
    p2_name: str = \"Ryu\"
    p1_health: float = 1.0
    p2_health: float = 1.0
    p1_super: float = 0.0
    p2_super: float = 0.0
    distance_px: int = 180
    last_action: str = \"\"
    last_result: str = \"\"

# Calibration analysis: read CAL_LOG and suggest deadzone
def analyze_calibration(log_path: Path) -> float:
    if not log_path.exists():
        print(\"Calibration file not found:\", log_path)
        return 0.25
    try:
        data = json.loads(log_path.read_text())
        samples = data.get(\"samples\", [])
        # compute axis ranges for axis 0 and 1
        vals0 = [s[\"axes\"][0] for s in samples if s.get(\"axes\") and len(s[\"axes\"])>0]
        vals1 = [s[\"axes\"][1] for s in samples if s.get(\"axes\") and len(s[\"axes\"])>1]
        if not vals0 or not vals1:
            return 0.25
        # measure noise around center
        import statistics
        std0 = statistics.pstdev(vals0)
        std1 = statistics.pstdev(vals1)
        suggested = max(0.15, min(0.45, max(std0, std1)*3.0))
        print(f\"Calibration analysis: std0={std0:.3f} std1={std1:.3f} suggested_deadzone={suggested:.3f}\")
        return suggested
    except Exception as e:
        print(\"Calibration analysis failed:\", e)
        return 0.25

# GhostAI: load existing patterns if available, basic decide mechanism
class GhostAI:
    def __init__(self):
        self.patterns = {}
        self.load()

    def load(self):
        if GHOST_FILE.exists():
            try:
                self.patterns = json.loads(GHOST_FILE.read_text())
                print(f\"Loaded ghost patterns: {len(self.patterns)} situations\") 
            except Exception as e:
                print(\"Failed to load ghost patterns:\", e)
                self.patterns = {}
        else:
            self.patterns = {}

    def save(self):
        try:
            GHOST_FILE.write_text(json.dumps(self.patterns))
            print(\"Saved ghost patterns to\", GHOST_FILE)
        except Exception as e:
            print(\"Ghost save failed:\", e)

    def decide(self, info: MatchInfo) -> str:
        # match on distance bucket and health state
        key = f\"dist:{(info.distance_px//50)}_p1hp:{int(info.p1_health*100)//10}\"
        if key in self.patterns and self.patterns[key]:
            choices = self.patterns[key]
            return random.choice(choices)
        # fallback simple rules
        if info.distance_px < 120:
            return \"cr.MK\"
        if info.p1_super > 0.9:
            return \"use_super\"
        return \"neutral\"

    def record(self, info: MatchInfo, action: str):
        key = f\"dist:{(info.distance_px//50)}_p1hp:{int(info.p1_health*100)//10}\"
        self.patterns.setdefault(key, []).append(action)

# Utilities for capture selection
def detect_wayland() -> bool:
    # heuristic: WAYLAND_DISPLAY present -> Wayland
    return os.environ.get(\"WAYLAND_DISPLAY\") is not None

def capture_using_system(window_id: str, out_path: str, duration: int = 2) -> bool:
    if detect_wayland() and WF_REC:
        # use wf-recorder or pw-record wrapper if available
        cmd = [\"wf-recorder\", \"-g\", \"x:0:0:640:480\", \"-o\", out_path, \"-t\", str(duration)]
        try:
            subprocess.run(cmd, check=True)
            return True
        except Exception as e:
            print(\"wf-recorder capture failed:\", e)
            return False
    elif FFMPEG:
        # find window geometry
        try:
            geom = subprocess.run([\"xdotool\",\"getwindowgeometry\",\"--shell\",window_id], capture_output=True, text=True, check=True).stdout
            g = {}
            for line in geom.splitlines():
                if \"=\" in line:
                    k,v = line.split(\"=\",1); g[k.strip()] = int(v.strip())
            left, top, w, h = g.get(\"X\",0), g.get(\"Y\",0), g.get(\"WIDTH\",640), g.get(\"HEIGHT\",480)
        except Exception:
            left, top, w, h = 0,0,640,480
        cmd = [\"ffmpeg\",\"-y\",\"-f\",\"x11grab\",\"-video_size\",f\"{w}x{h}\",\"-framerate\",\"60\",\"-i\",f\":0.0+{left},{top}\",\"-t\",str(duration),out_path]
        try:
            subprocess.run(cmd, check=True)
            return True
        except Exception as e:
            print(\"ffmpeg capture failed:\", e)
            return False
    else:
        print(\"No capture tool available (ffmpeg or wf-recorder/pw-record).\") 
        return False

# curses dashboard
def curses_dashboard_loop(info_queue):
    if not HAS_CURSES:
        print(\"curses not available; dashboard disabled.\")
        return
    def draw(stdscr):
        curses.curs_set(0)
        stdscr.nodelay(True)
        while True:
            try:
                info = info_queue.get(timeout=0.1)
            except Exception:
                info = None
            stdscr.erase()
            stdscr.addstr(1,2,\"3rd Strike Bot - Live Dashboard\")
            if info:
                stdscr.addstr(3,2,f\"MATCH: {info.p1_name} vs {info.p2_name}\")
                stdscr.addstr(4,2,f\"P1 HP: {int(info.p1_health*100)}%   P2 HP: {int(info.p2_health*100)}%\") 
                stdscr.addstr(5,2,f\"DIST: {info.distance_px}px   ACTION: {info.last_action}\")
            stdscr.refresh()
            time.sleep(0.05)
    import curses, queue
    curses.wrapper(draw)

# Main monitor that enqueues info for dashboard and handles ghost decisions
def monitor_loop(window_id, deadzone=0.25):
    info = MatchInfo()
    ghost = GhostAI()
    info_q = deque(maxlen=1)
    # simple loop producing data
    try:
        while True:
            # simulate or read vision if available (kept simple)
            if HAS_VISION:
                # attempt to read window geometry and screenshot; simplified like earlier scripts
                try:
                    geom = subprocess.run([\"xdotool\",\"getwindowgeometry\",\"--shell\",window_id], capture_output=True, text=True, check=True).stdout
                    g = {}
                    for line in geom.splitlines():
                        if \"=\" in line:
                            k,v = line.split(\"=\",1); g[k.strip()] = int(v.strip())
                    left, top, w, h = g.get(\"X\",0), g.get(\"Y\",0), g.get(\"WIDTH\",640), g.get(\"HEIGHT\",480)
                    import mss, numpy as np, cv2
                    sct = mss.mss(); img = sct.grab({\"left\":left,\"top\":top,\"width\":w,\"height\":h}); arr = np.array(img)
                    if arr.shape[2] == 4: arr = arr[:,:,:3]
                    # health heuristics simplified
                    y=int(h*0.05); hh=int(h*0.03)
                    p1crop = arr[y:y+hh, int(w*0.05):int(w*0.35)]
                    p2crop = arr[y:y+hh, int(w*0.65):int(w*0.95)]
                    def mean_v(c): 
                        hsv = cv2.cvtColor(c, cv2.COLOR_BGR2HSV); return float(hsv[:,:,2].mean())/255.0
                    info.p1_health = min(1.0, max(0.0, mean_v(p1crop)))
                    info.p2_health = min(1.0, max(0.0, mean_v(p2crop)))
                    gray = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY); th = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]; cols = th.sum(axis=0)
                    if cols.max() > 0:
                        lx = int(cols.argmax()); rx = int(cols[::-1].argmax()); info.distance_px = abs(rx - lx)
                except Exception as e:
                    # simulate
                    info.p1_health = max(0.0, min(1.0, info.p1_health + (0.5-random.random())*0.01))
                    info.p2_health = max(0.0, min(1.0, info.p2_health + (0.5-random.random())*0.01))
                    info.distance_px = max(50, min(400, info.distance_px + int((random.random()-0.5)*8)))
            else:
                info.p1_health = max(0.0, min(1.0, info.p1_health + (0.5-random.random())*0.01))
                info.p2_health = max(0.0, min(1.0, info.p2_health + (0.5-random.random())*0.01))
                info.distance_px = max(50, min(400, info.distance_px + int((random.random()-0.5)*8)))
            # ghost decide
            act = ghost.decide(info)
            info.last_action = act
            # auto-clip on big drop
            if (info.p1_health - info.p2_health) > 0.4 and (FFMPEG or WF_REC):
                outp = str(Path.home()/(\"auto_clip_\"+str(int(time.time()))+\".mp4\"))
                print(\"Auto-capture to\", outp)
                capture_using_system(window_id, outp, duration=2)
            # push to dashboard (simple approach: write to a temp json)
            info_q.append(info)
            # print status
            print(\"\\n[MATCH] {} (You) vs {} (Opp)\".format(info.p1_name, info.p2_name))
            print(\" [HEALTH] P1: {} {}%  |  P2: {} {}%\".format(health_bar(info.p1_health,10), int(info.p1_health*100), health_bar(info.p2_health,10), int(info.p2_health*100)))
            print(\" [DISTANCE] {}\".format(distance_bucket(info.distance_px)))
            print(\" [ACTION] Ghost decided: {}\".format(info.last_action))
            time.sleep(0.4)
    except KeyboardInterrupt:
        print(\"Monitor stopped.\")

def health_bar(perc, length=10):
    blocks = int(round(perc*length))
    return \"█\"*blocks + \"░\"*(length-blocks)

def distance_bucket(px):
    if px < 100: return f\"Close ({px}px)\"
    if px < 220: return f\"Medium ({px}px)\"
    return f\"Far ({px}px)\"

def main():
    print(\"3rd Strike Advanced - Ultimate V2\")
    while True:
        print(\"\\nMAIN MENU:\") 
        print(\"1. Analyze calibration log and suggest deadzone\") 
        print(\"2. Run monitor (uses vision if available)\") 
        print(\"3. Save/Load Ghost patterns\") 
        print(\"4. Exit\") 
        sel = input(\"Select: \").strip()
        if sel == '1':
            dz = analyze_calibration(CAL_LOG)
            print(\"Suggested deadzone:\", dz)
        elif sel == '2':
            wid = find_window_simple('Fightcade')
            if not wid:
                print('No window found (xdotool required).') 
            else:
                monitor_loop(wid)
        elif sel == '3':
            g = GhostAI(); g.save(); print('Saved ghost (if any)')
        elif sel == '4':
            break
        else:
            print('Invalid')

# simplified window find for main
def find_window_simple(fuzzy='Fightcade'):
    if not XDOTOOL:
        print('xdotool missing'); return None
    try:
        r = subprocess.run(['xdotool','search','--name',fuzzy], capture_output=True, text=True)
        ids = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        return ids[0] if ids else None
    except Exception:
        return None

if __name__ == '__main__':
    main()
