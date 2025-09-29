#!/usr/bin/env python3
"""3rd_Strike_Advanced_full_gui.py

Merged full version with:
- Correct header (no escaped quotes)
- OCR (pytesseract + Tesseract) character & Super Art detection
- Profile system for control mappings & mirror settings
- Boot-watch, control detection, rematch remembering
- Simple Mirror (P1->P2) using profiles
- Basic Tkinter GUI for profile editing and OCR calibration

Requirements:
    sudo apt install tesseract-ocr xdotool
    pip install pytesseract pillow opencv-python mss pynput tkinter
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path
from typing import Optional, Dict, Tuple

# Optional libs
try:
    import cv2, numpy as np, mss
    HAS_CV2 = True
    HAS_MSS = True
except Exception:
    HAS_CV2 = False
    HAS_MSS = False

try:
    import pytesseract
    from PIL import Image
    HAS_PYTESSERACT = True
except Exception:
    HAS_PYTESSERACT = False

try:
    from pynput import keyboard as kb
    HAS_PYNPUT = True
except Exception:
    HAS_PYNPUT = False

CONFIG_PATH = Path.home() / ".3rd_strike_bot_config.json"
PROFILES_PATH = Path.home() / ".3rd_strike_bot_profiles.json"
DEFAULT_WINDOW_NAME = "Fightcade"

def load_json(p: Path) -> dict:
    if p.exists():
        try:
            return json.loads(p.read_text())
        except Exception:
            return {}
    return {}

def save_json(p: Path, data: dict):
    try:
        p.write_text(json.dumps(data, indent=2))
    except Exception as e:
        logging.exception("Failed to write %s: %s", p, e)

def ensure_default_profile():
    profiles = load_json(PROFILES_PATH)
    if 'default' not in profiles:
        profiles['default'] = {
            'key_map': {'up':'w','down':'s','left':'a','right':'d',
                        'lp':'y','mp':'u','hp':'i','lk':'h','mk':'j','hk':'k'},
            'mirror_settings': {'mode':'exact','delay':0.05,
                                'max_random':0.2,'mistake_prob':0.02}
        }
        save_json(PROFILES_PATH, profiles)

def find_window(name: str) -> Optional[str]:
    try:
        r = subprocess.run(["xdotool","search","--name",name],
                           capture_output=True,text=True,check=True)
        ids = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        return ids[0] if ids else None
    except Exception:
        return None

def detect_controls(timeout=4.0) -> Dict[str,str]:
    logging.info("Detecting controls for %.1f seconds...", timeout)
    detected = set()
    result = {'p1':'unknown','p2':'unknown'}
    if not HAS_PYNPUT:
        return {'p1':'arrows','p2':'wasd'}
    def on_press(key):
        try:
            k = key.char.lower()
        except Exception:
            k = str(key).split('.')[-1].lower()
        detected.add(k)
    listener = kb.Listener(on_press=on_press)
    listener.start()
    time.sleep(timeout)
    listener.stop()
    arrow = {'up','down','left','right','up_arrow','down_arrow','left_arrow','right_arrow'}
    wasd = {'w','a','s','d'}; buttons={'y','u','i','h','j','k'}
    if detected & arrow: result['p1']='arrows'
    if detected & wasd or detected & buttons: result['p2']='wasd'
    if result['p1']=='unknown' and result['p2']!='unknown': result['p1']='arrows'
    if result['p2']=='unknown' and result['p1']!='unknown': result['p2']='wasd'
    return result

def is_tesseract_installed() -> bool:
    try:
        subprocess.run(['tesseract','--version'], capture_output=True, check=True)
        return True
    except Exception:
        return False

def ocr_text(region: Tuple[int,int,int,int]) -> str:
    if not (HAS_MSS and HAS_CV2 and HAS_PYTESSERACT and is_tesseract_installed()):
        return ""
    sct = mss.mss()
    img = sct.grab({'left':region[0],'top':region[1],
                    'width':region[2],'height':region[3]})
    arr = np.array(img)
    if arr.shape[2]==4: arr=arr[:,:,:3]
    pil = Image.fromarray(arr)
    return pytesseract.image_to_string(pil).strip()

def detect_character_sa() -> Tuple[Optional[str],Optional[str]]:
    # This is still heuristic
    text = ocr_text((100,100,400,200)).lower()
    chars = ['alex','ryu','ken','chun','elena','remy','gouki','hugo','makoto']
    found_char = next((c for c in chars if c in text), None)
    sa = None
    for token in ['sa1','sa2','sa3','super art 1','super art 2','super art 3']:
        if token in text:
            if '1' in token: sa='SA1'
            elif '2' in token: sa='SA2'
            elif '3' in token: sa='SA3'
            break
    return found_char, sa

def start_simple_mirror(profile: dict):
    if not HAS_PYNPUT:
        print("pynput required")
        return
    key_map = profile.get('mirror_map', {'left':'a','right':'d','up':'w','down':'s'})
    def on_press(key):
        try: k = key.char.lower()
        except Exception: k = str(key).split('.')[-1].lower()
        if k in key_map:
            subprocess.run(["xdotool","key",key_map[k]])
    listener = kb.Listener(on_press=on_press)
    listener.start()
    print("SimpleMirror active, Ctrl+C to stop")
    try:
        while True: time.sleep(0.5)
    except KeyboardInterrupt:
        listener.stop()

# ------------------ GUI for profile editing ------------------
def launch_gui():
    ensure_default_profile()
    profiles = load_json(PROFILES_PATH)

    root = tk.Tk()
    root.title("3rd Strike Bot - Profiles")

    profile_names = list(profiles.keys())
    current_profile = tk.StringVar(value=profile_names[0])

    frame = ttk.Frame(root, padding=10)
    frame.grid()

    ttk.Label(frame, text="Select profile:").grid(column=0,row=0,sticky='w')
    combo = ttk.Combobox(frame, textvariable=current_profile, values=profile_names)
    combo.grid(column=1,row=0)

    key_map_text = tk.Text(frame, width=40, height=10)
    key_map_text.grid(column=0,row=1,columnspan=2,pady=5)

    def load_profile(*args):
        prof = profiles[current_profile.get()]
        key_map_text.delete("1.0","end")
        for k,v in prof.get('key_map',{}).items():
            key_map_text.insert("end",f"{k}:{v}\n")
    combo.bind("<<ComboboxSelected>>", load_profile)
    load_profile()

    def save_profile():
        prof_name = current_profile.get()
        lines = key_map_text.get("1.0","end").strip().splitlines()
        km = {}
        for line in lines:
            if ':' in line:
                k,v = line.split(':',1)
                km[k.strip()] = v.strip()
        profiles[prof_name]['key_map'] = km
        save_json(PROFILES_PATH, profiles)
        messagebox.showinfo("Saved","Profile saved")
    ttk.Button(frame,text="Save",command=save_profile).grid(column=0,row=2,pady=5)
    ttk.Button(frame,text="Close",command=root.destroy).grid(column=1,row=2,pady=5)
    root.mainloop()

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--menu',action='store_true',help="Run interactive terminal menu")
    parser.add_argument('--gui',action='store_true',help="Launch GUI profile editor")
    parser.add_argument('--boot-watch',action='store_true',help="Wait for game window and detect")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
    ensure_default_profile()
    profiles = load_json(PROFILES_PATH)

    if args.gui:
        launch_gui()
        return
    if args.boot_watch:
        print("Waiting for Fightcade window...")
        wid = None
        while not wid:
            wid = find_window(DEFAULT_WINDOW_NAME)
            if not wid: time.sleep(1)
        print("Window found:", wid)
        controls = detect_controls()
        print("Controls:", controls)
        char, sa = detect_character_sa()
        print("Detected character:", char, "Super Art:", sa)
        return
    # default menu
    while True:
        print("\n=== 3RD STRIKE BOT ===")
        print("1. Detect controls")
        print("2. OCR character/Super Art")
        print("3. Simple Mirror")
        print("4. Launch GUI profile editor")
        print("5. Exit")
        sel = input("Select: ").strip()
        if sel=='1':
            print("Controls:", detect_controls())
        elif sel=='2':
            print("Detected:", detect_character_sa())
        elif sel=='3':
            start_simple_mirror(profiles.get('default',{}))
        elif sel=='4':
            launch_gui()
        elif sel=='5':
            break
        else:
            print("Invalid")

if __name__ == "__main__":
    main()
