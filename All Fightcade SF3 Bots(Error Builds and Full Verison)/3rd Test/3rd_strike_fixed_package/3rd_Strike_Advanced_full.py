#!/usr/bin/env python3
\"\"\"3rd_Strike_Advanced_full.py - Fixed release

Fixes applied:
- Graceful handling when tkinter is not installed (falls back to CLI; doesn't crash)
- More robust window detection: validates window title via `xdotool getwindowname`
- Improved input sending: activates window before sending keys, uses --clearmodifiers,
  logs stdout/stderr from xdotool calls for debugging.
- Added command-line tests:
    --test-window [name] : show which window would be chosen / confirm it's correct
    --test-inputs [name] : attempts to send sample keys to the chosen window (requires xdotool)
- Keeps menu/GUI features optional.
- Useful debug and troubleshooting prints to help you see why keys might not be sent.

Usage:
    python3 3rd_Strike_Advanced_full.py --menu
    python3 3rd_Strike_Advanced_full.py --boot-watch
    python3 3rd_Strike_Advanced_full.py --test-window Fightcade
    python3 3rd_Strike_Advanced_full.py --test-inputs Fightcade
\"\"\"

import argparse
import logging
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional, Dict

# Attempt to import tkinter but handle absence gracefully
try:
    import tkinter as tk
    from tkinter import ttk, messagebox
    HAS_TK = True
except Exception:
    HAS_TK = False

# Check for xdotool availability early
XDOTOOL_PRESENT = shutil.which('xdotool') is not None

def find_window(window_name_fuzzy: str = 'Fightcade') -> Optional[str]:
    \"\"\"Return a window id where the window title contains the fuzzy name.
    Verifies each found window with `xdotool getwindowname` to avoid false matches.\"\"\"
    if not XDOTOOL_PRESENT:
        logging.debug('xdotool not installed, cannot search windows.')
        return None
    try:
        # list candidate window ids
        result = subprocess.run(['xdotool', 'search', '--name', window_name_fuzzy],
                                capture_output=True, text=True)
        if result.returncode != 0 and not result.stdout:
            return None
        ids = [line.strip() for line in result.stdout.splitlines() if line.strip()]
        # verify by reading window name for each id and pick the best match
        for wid in ids:
            try:
                r2 = subprocess.run(['xdotool', 'getwindowname', wid], capture_output=True, text=True, check=True)
                name = (r2.stdout or '').strip()
                logging.debug('Found window id %s with name: %s', wid, name)
                if window_name_fuzzy.lower() in name.lower():
                    return wid
            except Exception:
                continue
        # if none matched the fuzzy name exactly, return first id if exists (last resort)
        return ids[0] if ids else None
    except Exception as e:
        logging.exception('find_window failed: %s', e)
        return None

def activate_and_send(window_id: str, key: str, debug: bool = False) -> bool:
    \"\"\"Activate window and send a single key using xdotool. Returns True on success.\"\"\"
    if not XDOTOOL_PRESENT:
        logging.error('xdotool is not installed; cannot send keys.')
        return False
    if not window_id:
        logging.error('No window id provided to send key.')
        return False
    try:
        # Activate the window first to ensure it receives synthetic key events
        act = subprocess.run(['xdotool', 'windowactivate', '--sync', window_id], capture_output=True, text=True)
        if debug:
            logging.debug('windowactivate stdout=%s stderr=%s rc=%s', act.stdout, act.stderr, act.returncode)
        # Send the key with clearmodifiers so modifier keys don't interfere
        cmd = ['xdotool', 'key', '--window', window_id, '--clearmodifiers', key]
        proc = subprocess.run(cmd, capture_output=True, text=True)
        if debug:
            logging.debug('key send cmd=%s stdout=%s stderr=%s rc=%s', ' '.join(cmd), proc.stdout, proc.stderr, proc.returncode)
        return proc.returncode == 0
    except Exception as e:
        logging.exception('activate_and_send failed: %s', e)
        return False

def test_window_and_inputs(window_name: str = 'Fightcade', debug: bool = True):
    \"\"\"Utility to test window lookup and send sample keys.\"\"\"
    print('XDOTOOL installed:', XDOTOOL_PRESENT)
    wid = find_window(window_name)
    if not wid:
        print(f'No window matching \"{window_name}\" found.')
        return
    print(f'Found window id: {wid}')
    try:
        name_res = subprocess.run(['xdotool', 'getwindowname', wid], capture_output=True, text=True, check=True)
        print('Window name:', name_res.stdout.strip())
    except Exception as e:
        print('Could not read window name:', e)
    # attempt test sends
    sample_keys = ['a', 'd', 'w', 's']
    for k in sample_keys:
        ok = activate_and_send(wid, k, debug=debug)
        print(f'Sent key {k}: {"OK" if ok else "FAIL"}')
        time.sleep(0.05)

# Provide a fallback CLI menu for common tasks
def cli_menu():
    while True:
        print('\\n=== 3RD STRIKE BOT - CLI MENU ===')
        print('1. Wait for window (boot-watch)')
        print('2. Detect controls (press keys briefly)')
        print('3. Test window & send sample inputs (debug)')
        print('4. Exit')
        sel = input('Select: ').strip()
        if sel == '1':
            name = input('Window fuzzy name (default Fightcade): ').strip() or 'Fightcade'
            print('Waiting for window... (Ctrl+C to cancel)')
            try:
                while True:
                    wid = find_window(name)
                    if wid:
                        print('Found window:', wid)
                        break
                    time.sleep(0.5)
            except KeyboardInterrupt:
                print('Cancelled.')
        elif sel == '2':
            print('Please press some keys (arrows for P1, WASD for P2) for 4 seconds...')
            try:
                import threading, time
                detected = set()
                try:
                    from pynput import keyboard as kb
                except Exception:
                    print('pynput not installed; cannot detect controls. Install with: pip install pynput')
                    continue
                def on_press(key):
                    try:
                        k = key.char.lower()
                    except Exception:
                        k = str(key).split('.')[-1].lower()
                    detected.add(k)
                listener = kb.Listener(on_press=on_press)
                listener.start()
                time.sleep(4.0)
                listener.stop()
                print('Detected keys:', detected)
                # heuristics
                arrow = {'up','down','left','right','up_arrow','down_arrow','left_arrow','right_arrow'}
                wasd = {'w','a','s','d'}
                if detected & arrow:
                    print('P1 likely uses arrow keys')
                if detected & wasd:
                    print('P2 likely uses WASD')
            except KeyboardInterrupt:
                print('Cancelled.')
        elif sel == '3':
            name = input('Window fuzzy name to test (default Fightcade): ').strip() or 'Fightcade'
            test_window_and_inputs(name)
        elif sel == '4':
            print('Exiting.')
            break
        else:
            print('Invalid selection.')

def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--menu', action='store_true', help='Run CLI menu')
    parser.add_argument('--boot-watch', action='store_true', help='Wait for game window then exit')
    parser.add_argument('--test-window', type=str, help='Test window lookup and report (provide fuzzy name)')
    parser.add_argument('--test-inputs', type=str, help='Test inputs by sending sample keys to window (provide fuzzy name)')
    parser.add_argument('--debug', action='store_true', help='Enable debug logging')
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.debug else logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')

    # Handle tkinter absence gracefully
    if not HAS_TK:
        logging.info('tkinter not available; GUI features disabled. Install system tkinter (e.g., python3-tk) to enable GUI.')
    else:
        logging.debug('tkinter available')

    if args.test_window:
        test_window_and_inputs(args.test_window, debug=args.debug)
        return
    if args.test_inputs:
        test_window_and_inputs(args.test_inputs, debug=args.debug)
        return
    if args.menu:
        cli_menu()
        return
    if args.boot_watch:
        print('Waiting for window... (Ctrl+C to cancel)')
        try:
            while True:
                wid = find_window('Fightcade')
                if wid:
                    print('Found window id:', wid)
                    try:
                        name_res = subprocess.run(['xdotool', 'getwindowname', wid], capture_output=True, text=True, check=True)
                        print('Window title:', name_res.stdout.strip())
                    except Exception:
                        pass
                    break
                time.sleep(0.5)
        except KeyboardInterrupt:
            print('Cancelled.')
        return
    # default: CLI menu
    cli_menu()

if __name__ == '__main__':
    main()
