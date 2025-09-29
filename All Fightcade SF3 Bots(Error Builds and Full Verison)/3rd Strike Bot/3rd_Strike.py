import cv2
import numpy as np
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional
import queue
import subprocess

@dataclass
class GameState:
    timestamp: float
    player1_health: float
    player2_health: float
    player1_super: float
    player2_super: float
    player1_position: Tuple[int, int]
    player2_position: Tuple[int, int]
    player1_state: str
    player2_state: str
    distance: float
    frame_advantage: int
    round_timer: float

class ActionCommand:
    def __init__(self, inputs: List[str], timing: float, duration: float, priority: int):
        self.inputs = inputs
        self.timing = timing
        self.duration = duration
        self.priority = priority

class GameStateExtractor:
    def extract_state(self) -> Optional[GameState]:
        raise NotImplementedError

class FightcadeExtractor(GameStateExtractor):
    def extract_state(self) -> Optional[GameState]:
        return GameState(
            timestamp=time.time(),
            player1_health=1.0,
            player2_health=1.0,
            player1_super=0.0,
            player2_super=0.0,
            player1_position=(160, 400),
            player2_position=(480, 400),
            player1_state="idle",
            player2_state="idle",
            distance=320,
            frame_advantage=0,
            round_timer=99.0
        )

class FightingGameAI:
    def decide_action(self, game_state: GameState, history: List[GameState]) -> Optional[ActionCommand]:
        raise NotImplementedError

class Fightcade3rdStrikeAI(FightingGameAI):
    def __init__(self, character: str = "ken"):
        self.character = character
        self.last_action_time = 0
        self.action_cooldown = 0.016
        self.combos = {
            'hadoken': ['down', 'down-right', 'right', 'hp'],
            'dp': ['right', 'down', 'down-right', 'hp']
        }

    def decide_action(self, game_state: GameState, history: List[GameState]) -> Optional[ActionCommand]:
        current_time = game_state.timestamp
        if current_time - self.last_action_time < self.action_cooldown:
            return None

        distance = game_state.distance
        if distance > 250:
            return ActionCommand(inputs=['right'], timing=current_time, duration=0.1, priority=1)
        if 150 < distance < 300:
            return ActionCommand(inputs=self.combos['hadoken'], timing=current_time, duration=0.2, priority=5)
        if distance < 100:
            return ActionCommand(inputs=self.combos['dp'], timing=current_time, duration=0.2, priority=10)

        self.last_action_time = current_time
        return None

class InputExecutor:
    def __init__(self):
        self.action_queue = queue.Queue()
        self.running = True

    def queue_action(self, action):
        self.action_queue.put(action)

    def execute_inputs(self):
        while self.running:
            try:
                action = self.action_queue.get(timeout=0.1)
                self._execute_action(action)
            except queue.Empty:
                continue

    def _execute_action(self, action):
        raise NotImplementedError

    def stop(self):
        self.running = False

class LinuxInputExecutor(InputExecutor):
    def __init__(self):
        super().__init__()
        # P2 controls mapping
        self.key_mappings = {
            "up": "w",
            "down": "s",
            "left": "a",
            "right": "d",
            "lp": "y",   # Weak Punch
            "mp": "u",   # Medium Punch
            "hp": "i",   # Strong Punch
            "lk": "h",   # Weak Kick
            "mk": "j",   # Medium Kick
            "hk": "k",   # Heavy Kick
            "coin": "6", # Select/Coin
            "start": "2"
        }
        
        # Auto-detect 3rd Strike window
        self.window_id = self._find_3rd_strike_window()
        self.window_name = None
        
        if self.window_id:
            print(f"✓ Found 3rd Strike window: {self.window_id}")
        else:
            print("❌ Could not find 3rd Strike window!")
            print("   Make sure Fightcade is running with 3rd Strike loaded")

    def _find_3rd_strike_window(self):
        """Automatically find the 3rd Strike window"""
        try:
            # First, find all Fightcade windows
            result = subprocess.run(
                ["xdotool", "search", "--name", "Fightcade"],
                capture_output=True, text=True, check=True
            )
            
            window_ids = result.stdout.strip().split('\n')
            
            # Check each window for 3rd Strike
            for wid in window_ids:
                if not wid:
                    continue
                    
                # Get window name
                name_result = subprocess.run(
                    ["xdotool", "getwindowname", wid],
                    capture_output=True, text=True, check=True
                )
                window_name = name_result.stdout.strip()
                
                # Check if it's 3rd Strike
                if "3rd Strike" in window_name or "Street Fighter III" in window_name:
                    self.window_name = window_name
                    print(f"🎮 Detected: {window_name}")
                    return wid
            
            # If no specific 3rd Strike window found, try the simpler search
            result = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "--name", "3rd Strike"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip().split("\n")[0]
            
        except subprocess.CalledProcessError:
            return None

    def _execute_action(self, action: ActionCommand):
        if not self.window_id:
            return

        for key in action.inputs:
            mapped = self.key_mappings.get(key, key)
            try:
                subprocess.run(
                    ["xdotool", "key", "--window", self.window_id, mapped],
                    check=True,
                    capture_output=True
                )
                time.sleep(action.duration / len(action.inputs))
            except Exception as e:
                print(f"xdotool failed: {e}")

class FightingGameBot:
    def __init__(self, extractor: GameStateExtractor, ai: FightingGameAI):
        self.extractor = extractor
        self.ai = ai
        self.executor = LinuxInputExecutor()
        self.running = False
        self.history = []

    def start(self):
        if not self.executor.window_id:
            print("\n❌ Cannot start - no window detected!")
            print("Troubleshooting:")
            print("  1. Make sure Fightcade is running")
            print("  2. Make sure 3rd Strike is loaded (not just in menu)")
            print("  3. Try running: xdotool search --name 'Fightcade'")
            return
            
        print("Starting bot...")
        self.running = True
        executor_thread = threading.Thread(target=self.executor.execute_inputs, daemon=True)
        executor_thread.start()
        self.run_loop()

    def run_loop(self):
        while self.running:
            state = self.extractor.extract_state()
            if state:
                self.history.append(state)
                if len(self.history) > 100:
                    self.history.pop(0)
                action = self.ai.decide_action(state, self.history)
                if action:
                    self.executor.queue_action(action)
            time.sleep(0.01)

    def stop(self):
        print("Stopping bot...")
        self.running = False
        self.executor.stop()

if __name__ == "__main__":
    print("="*60)
    print("🥋 Fightcade SFIII: 3rd Strike AI Bot")
    print("="*60)
    print()
    
    extractor = FightcadeExtractor()
    ai = Fightcade3rdStrikeAI(character="ken")
    bot = FightingGameBot(extractor, ai)

    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()
        print("\nBot stopped by user")
