import cv2
import numpy as np
import time
import threading
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import queue
import subprocess
import pickle
from pathlib import Path
from collections import defaultdict
from pynput import keyboard as kb
import random

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
    
    def to_situation_key(self) -> str:
        """Convert state to situation key for pattern matching"""
        health_bucket = round(self.player2_health * 10) / 10
        opp_health_bucket = round(self.player1_health * 10) / 10
        dist_bucket = int(self.distance / 50) * 50
        super_bucket = round(self.player2_super * 2) / 2
        return f"{health_bucket:.1f}_{opp_health_bucket:.1f}_{dist_bucket}_{super_bucket:.1f}"

class ActionCommand:
    def __init__(self, inputs: List[str], timing: float, duration: float, priority: int):
        self.inputs = inputs
        self.timing = timing
        self.duration = duration
        self.priority = priority

@dataclass
class RecordedAction:
    """Action recorded from player"""
    situation: str
    keys_pressed: List[str]
    duration: float
    timestamp: float

class InputRecorder:
    """Records player inputs in real-time"""
    
    def __init__(self):
        self.recording = False
        self.recorded_actions: List[RecordedAction] = []
        self.current_keys_held = {}
        self.current_game_state = None
        self.listener = None
        self.watch_keys = {'w', 'a', 's', 'd', 'y', 'u', 'i', 'h', 'j', 'k'}
        
    def start_recording(self):
        self.recording = True
        self.recorded_actions = []
        print("🔴 RECORDING - Play and I'll learn from you...")
        self.listener = kb.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        
    def stop_recording(self):
        self.recording = False
        if self.listener:
            self.listener.stop()
        print(f"⏹️  Recorded {len(self.recorded_actions)} actions")
        return self.recorded_actions
    
    def update_game_state(self, state: GameState):
        self.current_game_state = state
    
    def _normalize_key(self, key) -> Optional[str]:
        try:
            if hasattr(key, 'char') and key.char:
                return key.char.lower()
            return None
        except:
            return None
    
    def _on_press(self, key):
        if not self.recording:
            return
        key_name = self._normalize_key(key)
        if key_name and key_name in self.watch_keys:
            if key_name not in self.current_keys_held:
                self.current_keys_held[key_name] = time.time()
    
    def _on_release(self, key):
        if not self.recording:
            return
        key_name = self._normalize_key(key)
        if key_name and key_name in self.watch_keys:
            if key_name in self.current_keys_held and self.current_game_state:
                press_time = self.current_keys_held[key_name]
                duration = time.time() - press_time
                action = RecordedAction(
                    situation=self.current_game_state.to_situation_key(),
                    keys_pressed=[key_name],
                    duration=duration,
                    timestamp=press_time
                )
                self.recorded_actions.append(action)
                del self.current_keys_held[key_name]

class GhostAI:
    """AI that learns from watching player"""
    
    def __init__(self, character: str = "ken"):
        self.character = character
        self.last_action_time = 0
        self.action_cooldown = 0.016
        self.patterns: Dict[str, List[RecordedAction]] = defaultdict(list)
        self.situation_counts: Dict[str, int] = defaultdict(int)
        self.min_examples = 2
        self.randomness = 0.15
        self.save_dir = Path("ghost_data")
        self.save_dir.mkdir(exist_ok=True)
        
    def learn_from_recording(self, recorded_actions: List[RecordedAction]):
        print(f"\n🧠 Learning from {len(recorded_actions)} actions...")
        for action in recorded_actions:
            self.patterns[action.situation].append(action)
            self.situation_counts[action.situation] += 1
        print(f"✓ Learned {len(self.patterns)} unique situations")
        
    def decide_action(self, game_state: GameState, history: List[GameState]) -> Optional[ActionCommand]:
        current_time = game_state.timestamp
        if current_time - self.last_action_time < self.action_cooldown:
            return None
            
        situation = game_state.to_situation_key()
        
        if situation in self.patterns and len(self.patterns[situation]) >= self.min_examples:
            if random.random() > self.randomness:
                action = random.choice(self.patterns[situation])
            else:
                all_actions = [a for acts in self.patterns.values() for a in acts]
                if all_actions:
                    action = random.choice(all_actions)
                else:
                    return None
            
            self.last_action_time = current_time
            return ActionCommand(
                inputs=action.keys_pressed,
                timing=current_time,
                duration=min(action.duration, 0.2),
                priority=5
            )
        
        # Fuzzy match similar situations
        best_match = self._find_similar(game_state)
        if best_match:
            self.last_action_time = current_time
            return best_match
            
        return None
    
    def _find_similar(self, game_state: GameState) -> Optional[ActionCommand]:
        current_health = game_state.player2_health
        current_dist = game_state.distance
        best_match = None
        best_score = float('inf')
        
        for situation_key, actions in self.patterns.items():
            if len(actions) < self.min_examples:
                continue
            try:
                parts = situation_key.split('_')
                health, opp_health, dist, super_meter = map(float, parts)
                score = abs(health - current_health) * 2 + abs(dist - current_dist) / 100
                if score < best_score and score < 0.5:
                    best_score = score
                    best_match = actions
            except:
                continue
        
        if best_match:
            action = random.choice(best_match)
            return ActionCommand(
                inputs=action.keys_pressed,
                timing=time.time(),
                duration=min(action.duration, 0.2),
                priority=3
            )
        return None
    
    def save_patterns(self, name: str = None):
        if name is None:
            name = f"{self.character}_ghost_{int(time.time())}"
        filepath = self.save_dir / f"{name}.pkl"
        data = {
            'patterns': dict(self.patterns),
            'counts': dict(self.situation_counts),
            'character': self.character
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        print(f"💾 Saved to {filepath.name}")
        
    def load_patterns(self, filename: str):
        filepath = self.save_dir / filename
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return False
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.patterns = defaultdict(list, data['patterns'])
        self.situation_counts = defaultdict(int, data['counts'])
        print(f"✅ Loaded {len(self.patterns)} patterns from {filename}")
        return True
    
    def list_ghosts(self):
        files = sorted(list(self.save_dir.glob("*.pkl")))
        if not files:
            print("📁 No saved ghosts")
            return []
        print("\n📁 SAVED GHOSTS:")
        for i, f in enumerate(files, 1):
            try:
                data = pickle.load(open(f, 'rb'))
                print(f"  {i}. {f.name} - {len(data['patterns'])} situations")
            except:
                print(f"  {i}. {f.name} (corrupted)")
        return files

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
    def __init__(self, extractor: GameStateExtractor, ai):
        self.extractor = extractor
        self.ai = ai
        self.executor = LinuxInputExecutor()
        self.recorder = InputRecorder()
        self.running = False
        self.history = []
        self.mode = 'playback'  # 'record' or 'playback'

    def start_recording(self):
        if not self.executor.window_id:
            print("\n❌ Cannot start - no window detected!")
            return
        print("\n📼 RECORDING MODE")
        print("Play matches on P2 controls (WASD + YUIHJK)")
        print("Press Ctrl+C when done\n")
        self.mode = 'record'
        self.recorder.start_recording()
        self.running = True
        try:
            while self.running:
                state = self.extractor.extract_state()
                if state:
                    self.recorder.update_game_state(state)
                    self.history.append(state)
                    if len(self.history) > 100:
                        self.history.pop(0)
                time.sleep(0.016)
        except KeyboardInterrupt:
            print("\n⏹️  Recording stopped!")
        recorded = self.recorder.stop_recording()
        if recorded:
            if isinstance(self.ai, GhostAI):
                self.ai.learn_from_recording(recorded)
                save = input("\nSave this session? (y/n): ").lower()
                if save == 'y':
                    name = input("Name (or enter for auto): ").strip()
                    self.ai.save_patterns(name if name else None)

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
    print("\nMODE SELECTION:")
    print("  1. Simple AI (play now)")
    print("  2. Ghost AI - Record session")
    print("  3. Ghost AI - Play with learned patterns")
    print("  4. Ghost AI - Load saved ghost")
    print("  5. Ghost AI - View saved ghosts")
    
    choice = input("\nChoice (1-5): ").strip()
    
    extractor = FightcadeExtractor()
    
    if choice == "1":
        # Original simple AI
        ai = Fightcade3rdStrikeAI(character="ken")
        bot = FightingGameBot(extractor, ai)
        try:
            bot.start()
        except KeyboardInterrupt:
            bot.stop()
            print("\nBot stopped")
            
    elif choice == "2":
        # Record new ghost session
        ghost_ai = GhostAI(character="ken")
        bot = FightingGameBot(extractor, ghost_ai)
        bot.start_recording()
        
    elif choice == "3":
        # Play with ghost
        ghost_ai = GhostAI(character="ken")
        bot = FightingGameBot(extractor, ghost_ai)
        if len(ghost_ai.patterns) == 0:
            print("⚠️  No patterns loaded! Record or load first.")
        else:
            print(f"🎮 Playing with {len(ghost_ai.patterns)} situations")
            try:
                bot.start()
            except KeyboardInterrupt:
                bot.stop()
                print("\nBot stopped")
                
    elif choice == "4":
        # Load saved ghost
        ghost_ai = GhostAI(character="ken")
        files = ghost_ai.list_ghosts()
        if files:
            idx = input("Select number: ").strip()
            try:
                ghost_ai.load_patterns(files[int(idx)-1].name)
                bot = FightingGameBot(extractor, ghost_ai)
                print("\n🎮 Ready to play!")
                input("Press Enter to start...")
                try:
                    bot.start()
                except KeyboardInterrupt:
                    bot.stop()
                    print("\nBot stopped")
            except (ValueError, IndexError):
                print("❌ Invalid selection")
                
    elif choice == "5":
        # View ghosts
        ghost_ai = GhostAI(character="ken")
        ghost_ai.list_ghosts()
        
    else:
        print("❌ Invalid choice")
