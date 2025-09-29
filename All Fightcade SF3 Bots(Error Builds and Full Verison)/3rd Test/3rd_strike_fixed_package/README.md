
3rd Strike Advanced Bot - Fixed Package
======================================

What's fixed:
1) Script no longer crashes if tkinter is missing - it logs a message and falls back to CLI.
   To enable GUI install system tkinter (Ubuntu/Debian: sudo apt install python3-tk).

2) Window detection is more robust: it verifies window titles returned by xdotool, which
   helps avoid false positives. If you still see a wrong window being found, run:
     python3 3rd_Strike_Advanced_full.py --test-window Fightcade
   to inspect which window is returned.

3) Input sending improved: window is activated before sending keys and stdout/stderr from
   xdotool is logged so you can diagnose why keys weren't delivered.

Usage:
  python3 3rd_Strike_Advanced_full.py --menu
  python3 3rd_Strike_Advanced_full.py --boot-watch
  python3 3rd_Strike_Advanced_full.py --test-window Fightcade
  python3 3rd_Strike_Advanced_full.py --test-inputs Fightcade

Dependencies (for full feature set):
  sudo apt install xdotool python3-tk tesseract-ocr
  pip install pytesseract pillow opencv-python mss pynput

