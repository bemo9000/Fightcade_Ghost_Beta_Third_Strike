#!/bin/bash
# Launcher for 3rd Strike Bot
# Opens a new terminal window and runs the bot
cd "$(dirname "$0")"
gnome-terminal -- bash -c "python3 3rd_strike_bot.py; exec bash"
