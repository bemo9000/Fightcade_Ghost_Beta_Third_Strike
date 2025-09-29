# 3rd Strike Advanced Fightcade Bot

## Description
A single-file unified bot for Fightcade/FBNeo Street Fighter III: 3rd Strike.

## Features
* **CLI First** – Always boots to a terminal menu for safe testing.
* **Robust Window Detection** – Prefers FBNeo/3rd Strike game windows.
* **Test Mode** – Activate window and send keys to confirm inputs work.
* **Basic Mirror Mode** – Simple P1→P2 input mirroring.
* **Advanced Mode** – Launches Vision/AI/Recording system (Ghost AI placeholder).

## Installation
```bash
sudo apt install xdotool python3-tk tesseract-ocr
pip install pynput mss opencv-python pillow pytesseract
```

## Usage
```bash
python3 3rd_Strike_Advanced_Fightcade.py
```

Follow the menu:
1. **Test Window & Inputs** – Confirms FBNeo/3rd Strike window detection and key sending.
2. **Basic Training / Mirror Mode** – Mirrors P1 inputs to P2 in real time.
3. **Launch Advanced Vision/AI/Recording** – Starts Ghost AI & vision systems (placeholder demo).
4. **Exit** – Quit.

Make sure the game window is visible and focused when testing.
