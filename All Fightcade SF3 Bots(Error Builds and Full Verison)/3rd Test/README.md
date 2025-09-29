# 3rd Strike Advanced Bot (Full GUI + OCR Package)

This package provides an advanced automation and training bot for **Street Fighter III: 3rd Strike** on Fightcade.

## Features
- **Boot-watch**: Waits for the Fightcade game window and detects when 3rd Strike is running.
- **Automatic Control Detection**: Identifies Player 1 and Player 2 control schemes (arrows vs WASD).
- **Character & Super Art OCR**: Uses Tesseract OCR to detect selected character and super art.
- **Mirror Mode**: Mirrors Player 1 inputs to Player 2 for training and recording.
- **Profiles**: Persistent configuration for control mappings and mirror settings (`~/.3rd_strike_bot_profiles.json`).
- **GUI Editor**: Tkinter GUI to easily view and edit profiles or calibrate OCR regions.

## Requirements
- Linux with `xdotool` installed for sending inputs
- Python 3.8+
- System Tesseract OCR binary
- Python packages:
  ```bash
  pip install pytesseract pillow opencv-python mss pynput
  sudo apt install python3-tk   # for Tkinter GUI if not already installed
  ```

## Usage
Extract the ZIP and run:
```bash
python3 3rd_Strike_Advanced_full_gui.py --menu      # Interactive terminal menu
python3 3rd_Strike_Advanced_full_gui.py --gui       # GUI profile editor
python3 3rd_Strike_Advanced_full_gui.py --boot-watch  # Wait for Fightcade window and auto-detect
```

### Terminal Menu Options
1. Detect controls  
2. OCR character / Super Art  
3. Simple Mirror (mirrors Player 1 inputs to Player 2)  
4. Launch GUI profile editor  
5. Exit  

### GUI Profile Editor
The GUI allows you to:
- Select and edit key mappings
- Save profiles for different setups
- Adjust mirror settings (delay, random delay, mistake probability)

## Tips
- Make sure Fightcade window name contains the word **Fightcade** (default).  
- OCR detection works best if the game window is not obscured.  
- Profiles are saved automatically in `~/.3rd_strike_bot_profiles.json`.

## Notes
This bot is for personal training and experimentation only. Use responsibly.
