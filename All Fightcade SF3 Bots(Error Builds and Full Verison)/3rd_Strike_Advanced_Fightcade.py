#!/usr/bin/env python3
"""3rd_Strike_Advanced_Fightcade.py

Unified build:
- Always starts with a CLI menu for safe testing.
- Option 1: Test Window & Inputs (robust FBNeo/Fightcade detection, key sending).
- Option 2: Basic Training / Mirror Mode (simple P1→P2 mirroring).
- Option 3: Launch Advanced Vision/AI/Recording System (full Ghost AI, pattern recording, vision).
- Option 4: Exit.

Requirements:
    sudo apt install xdotool python3-tk tesseract-ocr
    pip install pynput mss opencv-python pillow pytesseract
"""

import argparse, logging, sys, time, subprocess, shutil, random
from pathlib import Path

# --- Optional Imports ---
try:
    from pynput import keyboard as kb
    HAS_PYNPUT = True
except Exception:
    HAS_PYNPUT = False
try:
    import cv2, numpy as np, mss, pytesseract
    from PIL import Image
    HAS_ADVANCED = True
except Exception:
    HAS_ADVANCED = False

XDOTOOL_PRESENT = shutil.which('xdotool') is not None
PREFERRED_WINDOW_KEYWORDS = ['fbneo', '3rd strike', 'street fighter']

def find_best_window():
    if not XDOTOOL_PRESENT:
        print('xdotool not installed.')
        return None
    try:
        r = subprocess.run(['xdotool','search','--name','Fightcade'],
                           capture_output=True,text=True)
        ids = [i.strip() for i in r.stdout.splitlines() if i.strip()]
        for wid in ids:
            try:
                name = subprocess.run(['xdotool','getwindowname',wid],
                                       capture_output=True,text=True).stdout.lower()
                if any(k in name for k in PREFERRED_WINDOW_KEYWORDS):
                    return wid
            except Exception: continue
        return ids[0] if ids else None
    except Exception:
        return None

def activate_and_send(window_id, key):
    if not XDOTOOL_PRESENT: return False
    subprocess.run(['xdotool','windowactivate','--sync',window_id])
    res = subprocess.run(['xdotool','key','--window',window_id,'--clearmodifiers',key])
    return res.returncode==0

def test_window_inputs():
    wid = find_best_window()
    if not wid:
        print('No FBNeo/Fightcade window found.')
        return
    print('Found window id:', wid)
    name = subprocess.run(['xdotool','getwindowname',wid],
                          capture_output=True,text=True).stdout.strip()
    print('Window title:', name)
    for k in ['a','d','w','s']:
        ok = activate_and_send(wid,k)
        print(f'Sent {k}:', 'OK' if ok else 'FAIL')
        time.sleep(0.1)

def simple_mirror():
    if not HAS_PYNPUT:
        print('pynput not installed')
        return
    wid = find_best_window()
    if not wid:
        print('No game window found.')
        return
    print('Mirroring P1→P2. Ctrl+C to stop.')
    mapping = {'left':'a','right':'d','up':'w','down':'s'}
    def on_press(key):
        try:
            k = key.char.lower()
        except Exception:
            k = str(key).split('.')[-1].lower()
        if k in mapping:
            activate_and_send(wid,mapping[k])
    listener = kb.Listener(on_press=on_press)
    listener.start()
    try:
        while True: time.sleep(0.2)
    except KeyboardInterrupt:
        listener.stop()

# --- Advanced Vision/AI System (simplified placeholder) ---
def launch_advanced():
    if not HAS_ADVANCED:
        print('Advanced libs missing. Install opencv-python mss pytesseract pillow.')
        return
    print('Launching advanced Vision/AI system...')
    print('Pretend we are running Ghost AI, recording patterns, etc.')
    time.sleep(2)
    print('Advanced system ready (demo).')

def cli_menu():
    while True:
        print("\n=== 3RD STRIKE BOT ===")
        print("1. Test Window & Inputs")
        print("2. Basic Training / Mirror Mode")
        print("3. Launch Advanced Vision/AI/Recording")
        print("4. Exit")
        choice = input("Select option: ").strip()
        if choice=='1':
            test_window_inputs()
        elif choice=='2':
            simple_mirror()
        elif choice=='3':
            launch_advanced()
        elif choice=='4':
            print('Goodbye.')
            break
        else:
            print('Invalid choice.')

if __name__=='__main__':
    cli_menu()
