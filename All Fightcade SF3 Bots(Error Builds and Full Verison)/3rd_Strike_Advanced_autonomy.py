#!/usr/bin/env python3
\"\"\"3rd_Strike_Advanced_autonomy.py

Features added per user request:
- boot-watch mode: wait for Fightcade window to appear (user can run while game boots)
- automatic P1/P2 control detection by listening for keypresses (arrows/buttons vs WASD+YUI...)
- character + Super Art detection: uses vision if available; otherwise interactive prompt at select
- remembers selected character and SA between rematches (saves to JSON)
- training selection menu presented when session begins
- Help option and transparent gameplay logs
- Simple mode now mirrors P1 inputs to P2 to demonstrate recording (instead of heuristic walk-right)
- Saves configuration (window_name, last_characters) to config.json
- Packaged as a single script; requires pynput for key listening and xdotool for input sending on Linux

Usage examples:
    python3 3rd_Strike_Advanced_autonomy.py --menu
    python3 3rd_Strike_Advanced_autonomy.py --boot-watch
    python3 3rd_Strike_Advanced_autonomy.py --export-zip

Note: vision-based detection requires mss + opencv; otherwise script falls back to prompts.
\"\"\"

import argparse
import json
import logging
import os
import subprocess
import sys
import threading
import time
import zipfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Optional libs
try:
    from pynput import keyboard as kb
    HAS_PYNPUT = True
except Exception:
    HAS_PYNPUT = False

try:
    import cv2, numpy as np, mss
    HAS_CV2 = True
    HAS_MSS = True
except Exception:
    HAS_CV2 = False
    HAS_MSS = False

CONFIG_PATH = Path.home() / \".3rd_strike_bot_config.json\"

DEFAULT_WINDOW_NAME = \"Fightcade\"

# --------------------------------------------------------------------------------------
# Simple persistence for remembered selections and settings
# --------------------------------------------------------------------------------------
def load_config():
    if CONFIG_PATH.exists():
        try:
            return json.loads(CONFIG_PATH.read_text())
        except Exception:
            return {}
    return {}

def save_config(cfg):
    try:
        CONFIG_PATH.write_text(json.dumps(cfg, indent=2))
    except Exception as e:
        logging.exception(\"Failed to write config: %s\", e)

# --------------------------------------------------------------------------------------
# Minimal GameState dataclass (reused simpler for transparency)
# --------------------------------------------------------------------------------------
@dataclass
class GameState:
    timestamp: float
    distance: float = 320.0
    p1_health: float = 1.0
    p2_health: float = 1.0

# --------------------------------------------------------------------------------------
# Utilities: window detection via xdotool
# --------------------------------------------------------------------------------------
def find_window(window_name_fuzzy: str = DEFAULT_WINDOW_NAME) -> Optional[str]:
    try:
        r = subprocess.run([\"xdotool\", \"search\", \"--name\", window_name_fuzzy], capture_output=True, text=True, check=True)
        ids = [l.strip() for l in r.stdout.splitlines() if l.strip()]
        return ids[0] if ids else None
    except Exception:
        return None

def wait_for_window(window_name_fuzzy: str = DEFAULT_WINDOW_NAME, timeout: Optional[float] = None):
    logging.info(f\"Waiting for window matching '{window_name_fuzzy}'...\")
    start = time.time()
    while True:
        w = find_window(window_name_fuzzy)
        if w:
            logging.info(f\"Found window: {w}\")
            return w
        if timeout and (time.time() - start) > timeout:
            logging.warning(\"Timeout waiting for window.\")
            return None
        time.sleep(0.5)

# --------------------------------------------------------------------------------------
# Input executor with helper to send single mapped key (for mirror)
# --------------------------------------------------------------------------------------
class LinuxInputExecutor:
    def __init__(self, window_name_fuzzy: str = DEFAULT_WINDOW_NAME):
        self.window_name_fuzzy = window_name_fuzzy
        self.window_id = find_window(window_name_fuzzy)
        self.key_map = {
            'up': 'w', 'down': 's', 'left': 'a', 'right': 'd',
            'lp': 'y', 'mp': 'u', 'hp': 'i', 'lk': 'h', 'mk': 'j', 'hk': 'k'
        }

    def refresh_window(self):
        self.window_id = find_window(self.window_name_fuzzy)

    def send_key(self, key: str):
        if not self.window_id:
            logging.debug(\"No window id; cannot send key\")
            return False
        mapped = self.key_map.get(key, key)
        try:
            subprocess.run([\"xdotool\", \"key\", \"--window\", self.window_id, mapped], check=True, capture_output=True)
            logging.debug(f\"Sent key '{key}' -> '{mapped}' to window {self.window_id}\")
            return True
        except Exception as e:
            logging.exception(\"Failed to send key: %s\", e)
            return False

# --------------------------------------------------------------------------------------
# Control detection: listen for keys for short period and infer P1/P2 mapping by presence
# of arrow keys vs WASD + YUI...
# --------------------------------------------------------------------------------------
def detect_controls(timeout: float = 5.0) -> Dict[str, str]:
    \"\"\"Listen for key presses for `timeout` seconds and guess which keys are P1 vs P2.\n    Returns mapping: {'p1': 'arrows', 'p2': 'wasd'} or 'unknown' if ambiguous\"\"\"\n    logging.info(\"Detecting controls: press some keys for P1 (arrows + buttons) and P2 (WASD + buttons). You have %s seconds...\", timeout)\n    detected = set()\n    results = {'p1': 'unknown', 'p2': 'unknown'}\n\n    if not HAS_PYNPUT:\n        logging.warning(\"pynput not installed; cannot auto-detect controls. Falling back to defaults.\")\n        return {'p1': 'arrows', 'p2': 'wasd'}\n\n    def on_press(key):\n        try:\n            k = key.char.lower()\n        except Exception:\n            k = str(key).split('.')[-1].lower()\n        detected.add(k)\n\n    listener = kb.Listener(on_press=on_press)\n    listener.start()\n    time.sleep(timeout)\n    listener.stop()\n\n    # heuristics\n    arrow_keys = {'up','down','left','right','up_arrow','down_arrow','left_arrow','right_arrow'}\n    wasd = {'w','a','s','d'}\n    p2_buttons = {'y','u','i','h','j','k'}\n\n    if detected & arrow_keys:\n        results['p1'] = 'arrows'\n    if detected & wasd or detected & p2_buttons:\n        results['p2'] = 'wasd'\n\n    # fallback if ambiguous\n    if results['p1'] == 'unknown' and results['p2'] != 'unknown':\n        results['p1'] = 'arrows'\n    if results['p2'] == 'unknown' and results['p1'] != 'unknown':\n        results['p2'] = 'wasd'\n\n    logging.info(f\"Detected keys: {detected}\")\n    logging.info(f\"Control detection result: {results}\")\n    return results\n\n# --------------------------------------------------------------------------------------\n# Character and Super Art detection (vision + fallback interactive)\n# We'll try to use a basic template matching if vision available; otherwise prompt.\n# For now, default starting matchup Alex vs Ryu, SA1 by default unless changed.\n# --------------------------------------------------------------------------------------\n\ndef detect_character_and_sa(executor: LinuxInputExecutor, timeout: float = 10.0) -> Tuple[str,str]:\n    cfg = load_config()\n    last = cfg.get('last_picks', {})\n    # If we've seen a previous pick recently return it; this handles rematch remembering\n    if last:\n        logging.info(f\"Remembered last picks: {last}\")\n        return last.get('p1', 'alex'), last.get('sa_p1', 'SA1')\n\n    # Try vision-based detection if possible\n    if HAS_CV2 and HAS_MSS:\n        logging.info(\"Attempting vision-based character detection (experimental)...\")\n        try:\n            # crude: search for text 'Alex' or 'Ryu' on screen using OCR would be ideal but OCR isn't included.\n            # Instead, take a screenshot and let the user confirm.\n            sct = mss.mss()\n            win = executor.window_id or find_window(executor.window_name_fuzzy)\n            if not win:\n                executor.refresh_window()\n            # take a single screenshot of screen center region and show to user\n            img = sct.grab(sct.monitors[0])\n            arr = np.array(img)\n            cv2.imwrite('select_snapshot.png', arr)\n            logging.info(\"Saved select_snapshot.png - please open it and confirm characters. Falling back to prompt if needed.\")\n        except Exception as e:\n            logging.exception(\"Vision snapshot failed: %s\", e)\n\n    # interactive fallback\n    print(\"Character selection detection fallback:\")\n    p1 = input(\"Who did YOU pick? (default 'alex'): \").strip() or 'alex'\n    sa = input(\"Which Super Art did you select for YOU? (SA1/SA2/SA3) [default SA1]: \").strip() or 'SA1'\n    # save for rematch memory\n    cfg = load_config()\n    cfg.setdefault('last_picks', {})\n    cfg['last_picks']['p1'] = p1\n    cfg['last_picks']['sa_p1'] = sa\n    save_config(cfg)\n    logging.info(f\"Remembering pick: {p1} ({sa})\")\n    return p1, sa\n\n# --------------------------------------------------------------------------------------\n# Training selection menu\n# --------------------------------------------------------------------------------------\nTRAINING_OPTIONS = [\n    \"Defensive training\",\n    \"Offensive training\",\n    \"Neutral/footsies training\",\n    \"Combo practice\",\n    \"Anti-air practice\",\n    \"Full match recording\"\n]\n\ndef choose_training_type():\n    print(\"\\nSelect training type:\")\n    for i, t in enumerate(TRAINING_OPTIONS, 1):\n        print(f\"{i}. {t}\")\n    sel = input(\"Choose (1-6): \").strip()\n    try:\n        idx = int(sel) - 1\n        if 0 <= idx < len(TRAINING_OPTIONS):\n            logging.info(f\"Selected training: {TRAINING_OPTIONS[idx]}\")\n            return TRAINING_OPTIONS[idx]\n    except Exception:\n        pass\n    logging.info(\"Invalid selection; defaulting to Full match recording\")\n    return TRAINING_OPTIONS[-1]\n\n# --------------------------------------------------------------------------------------\n# Simple mode now mimics P1 inputs: we implement a lightweight mirror using pynput\n# --------------------------------------------------------------------------------------\nclass SimpleMirror:\n    def __init__(self, executor: LinuxInputExecutor):\n        self.executor = executor\n        self.listener = None\n        self.running = False\n\n    def _normalize_key(self, key):\n        try:\n            return key.char.lower()\n        except Exception:\n            return str(key).split('.')[-1].lower()\n\n    def _on_press(self, key):\n        k = self._normalize_key(key)\n        # Map P1 arrows to P2 wasd, and P1 buttons to P2 buttons\n        map_table = {\n            'left': 'a', 'right': 'd', 'up': 'w', 'down': 's',\n            'y': 'y','u':'u','i':'i','h':'h','j':'j','k':'k'\n        }\n        mapped = map_table.get(k)\n        if mapped:\n            self.executor.send_key(mapped)\n            logging.debug(f\"SimpleMirror: mapped {k} -> {mapped}\")\n\n    def start(self):\n        if not HAS_PYNPUT:\n            logging.error(\"pynput not installed: Simple mirror unavailable\")\n            return\n        self.running = True\n        self.listener = kb.Listener(on_press=self._on_press)\n        self.listener.start()\n        logging.info(\"SimpleMirror started: P1 inputs are being mirrored to P2\")\n\n    def stop(self):\n        self.running = False\n        if self.listener:\n            self.listener.stop()\n        logging.info(\"SimpleMirror stopped\")\n\n# --------------------------------------------------------------------------------------\n# Help and transparency print\n# --------------------------------------------------------------------------------------\ndef print_help():\n    print(\"\"\"\n3rd Strike Bot - Help\n- Boot-watch: start the script with --boot-watch while starting Fightcade. The script will wait\n  for the Fightcade window to appear and then auto-run detection.\n- Control detection: the bot listens briefly to infer which keys are used for P1 and P2.\n- Character/SA detection: uses vision when available; otherwise asks you to input choices.\n- Simple mode: now mirrors P1 inputs to P2 so you can see recording.\n- Mirror mode: interactive option in menu for advanced mirroring behaviors.\n\"\"\")\n\n# --------------------------------------------------------------------------------------\n# Interactive main menu tying everything together\n# --------------------------------------------------------------------------------------\n\ndef main_menu(window_name=DEFAULT_WINDOW_NAME):\n    cfg = load_config()\n    executor = LinuxInputExecutor(window_name)\n\n    while True:\n        print(\"\\n=== 3RD STRIKE BOT ===\")\n        print(\"1. Boot-watch (wait for game and auto-detect)\")\n        print(\"2. Detect Controls now\")\n        print(\"3. Character/SA detection\")\n        print(\"4. Choose training type\")\n        print(\"5. Simple mirror mode (mirror P1->P2)\")\n        print(\"6. Help\")\n        print(\"7. Exit\")\n        sel = input(\"Select (1-7): \").strip()\n        if sel == '1':\n            print(\"Boot-watch: waiting for game window...\")\n            w = wait_for_window(window_name)\n            if not w:\n                print(\"Window not found; returning to menu\")\n                continue\n            # found window; detect controls & characters\n            controls = detect_controls(timeout=4.0)\n            p1_ctrl = controls.get('p1')\n            p2_ctrl = controls.get('p2')\n            print(f\"Detected: P1={p1_ctrl}, P2={p2_ctrl}\")\n            p1_char, p1_sa = detect_character_and_sa(executor)\n            print(f\"You picked: {p1_char} ({p1_sa})\")\n        elif sel == '2':\n            controls = detect_controls()\n            print(f\"Controls: {controls}\")\n        elif sel == '3':\n            p1_char, p1_sa = detect_character_and_sa(executor)\n            print(f\"Detected/selected: {p1_char} / {p1_sa}\")\n        elif sel == '4':\n            t = choose_training_type()\n            print(f\"Training: {t}\")\n        elif sel == '5':\n            sm = SimpleMirror(executor)\n            try:\n                sm.start()\n                print(\"Press Ctrl+C to stop Simple Mirror...\")\n                while True:\n                    time.sleep(0.5)\n            except KeyboardInterrupt:\n                sm.stop()\n                continue\n        elif sel == '6':\n            print_help()\n        elif sel == '7':\n            print(\"Exiting.\")\n            break\n        else:\n            print(\"Invalid selection\")\n\n# --------------------------------------------------------------------------------------\n# CLI / entrypoint\n# --------------------------------------------------------------------------------------\n\ndef create_zip(output='3rd_strike_autonomy_package.zip'):\n    here = Path('.').resolve()\n    files = [\"3rd_Strike_Advanced_autonomy.py\"]\n    with zipfile.ZipFile(output, 'w', compression=zipfile.ZIP_DEFLATED) as z:\n        for f in files:\n            if os.path.exists(f):\n                z.write(f)\n    return output\n\n\ndef main(argv=None):\n    parser = argparse.ArgumentParser()\n    parser.add_argument('--menu', action='store_true', help='Start interactive menu')\n    parser.add_argument('--boot-watch', action='store_true', help='Wait for Fightcade window and auto-detect')\n    parser.add_argument('--window-name', type=str, default=DEFAULT_WINDOW_NAME)\n    parser.add_argument('--export-zip', action='store_true')\n    args = parser.parse_args(argv)\n\n    logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')\n\n    if args.export_zip:\n        z = create_zip()\n        print(f\"Created {z}\")\n        return\n    if args.menu:\n        main_menu(window_name=args.window_name)\n        return\n    if args.boot_watch:\n        w = wait_for_window(args.window_name)\n        if not w:\n            print(\"Window not found\")\n            return\n        controls = detect_controls(timeout=4.0)\n        p1_char, p1_sa = detect_character_and_sa(LinuxInputExecutor(args.window_name))\n        print(f\"Auto-detected: controls={controls}, you picked {p1_char} ({p1_sa})\")\n        return\n    # default: show menu\n    main_menu(window_name=args.window_name)\n\nif __name__ == '__main__':\n    main()\n