import cv2
import numpy as np
import time
import threading
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
import queue
import subprocess
import json
import pickle
from collections import defaultdict
from pathlib import Path
from pynput import keyboard as kb

@dataclass
class GameState:
    timestamp: float
    player1_health: float
    player2_health: float
    player1_super: float
    player2_super: float
    player1_position: Tuple[int, int]
    player2_position: Tuple[int, int]
    distance: float
    round_timer: float
    
    def to_situation_key(self) -> str:
        """Convert state to a situation key for pattern matching"""
        # Discretize for pattern matching
        health_bucket = round(self.player1_health * 10) / 10  # 0.0, 0.1, 0.2 ... 1.0
        opp_health_bucket = round(self.player2_health * 10) / 10
        dist_bucket = int(self.distance / 50) * 50  # 0, 50, 100, 150, 200...
        super_bucket = round(self.player1_super * 2) / 2  # 0, 0.5, 1.0
        
        return f"{health_bucket:.1f}_{opp_health_bucket:.1f}_{dist_bucket}_{super_bucket:.1f}"

@dataclass
class RecordedAction:
    """Action with context from when it was performed"""
    situation: str  # The game situation key
    keys_pressed: List[str]  # What keys were pressed
    duration: float  # How long held
    timestamp: float
    outcome_score: float = 0.0  # Did this lead to damage/success?

class ActionCommand:
    def __init__(self, inputs: List[str], timing: float, duration: float, priority: int):
        self.inputs = inputs
        self.timing = timing
        self.duration = duration
        self.priority = priority

class InputRecorder:
    """Records keyboard inputs in real-time and associates them with game state"""
    
    def __init__(self):
        self.recording = False
        self.recorded_actions: List[RecordedAction] = []
        self.current_keys_held = {}  # key -> press_timestamp
        self.current_game_state = None
        self.listener = None
        
        # P2 controls only (what we're learning from and will use)
        self.p2_keys = {'w', 'a', 's', 'd', 'y', 'u', 'i', 'h', 'j', 'k', '6', '2'}
        self.watch_keys = self.p2_keys
        
    def start_recording(self):
        """Start recording inputs"""
        self.recording = True
        self.recorded_actions = []
        print("🔴 RECORDING - Play your matches, I'm watching...")
        
        # Start keyboard listener
        self.listener = kb.Listener(on_press=self._on_key_press, on_release=self._on_key_release)
        self.listener.start()
        
    def stop_recording(self):
        """Stop recording"""
        self.recording = False
        if self.listener:
            self.listener.stop()
        print(f"⏹️  Recording stopped - captured {len(self.recorded_actions)} actions")
        return self.recorded_actions
    
    def update_game_state(self, state: GameState):
        """Update current game state (called from main loop)"""
        self.current_game_state = state
    
    def _normalize_key(self, key) -> Optional[str]:
        """Convert pynput key to our key name"""
        try:
            # Handle special keys
            if hasattr(key, 'char') and key.char:
                return key.char.lower()
            elif hasattr(key, 'name'):
                return key.name.lower()
            return None
        except:
            return None
    
    def _on_key_press(self, key):
        """Callback when key is pressed"""
        if not self.recording:
            return
            
        key_name = self._normalize_key(key)
        if not key_name or key_name not in self.watch_keys:
            return
        
        # Record when key was first pressed
        if key_name not in self.current_keys_held:
            self.current_keys_held[key_name] = time.time()
    
    def _on_key_release(self, key):
        """Callback when key is released"""
        if not self.recording:
            return
            
        key_name = self._normalize_key(key)
        if not key_name or key_name not in self.watch_keys:
            return
        
        # Record the full action when key is released
        if key_name in self.current_keys_held and self.current_game_state:
            press_time = self.current_keys_held[key_name]
            duration = time.time() - press_time
            
            # Create recorded action with game context
            action = RecordedAction(
                situation=self.current_game_state.to_situation_key(),
                keys_pressed=[key_name],
                duration=duration,
                timestamp=press_time
            )
            self.recorded_actions.append(action)
            
            del self.current_keys_held[key_name]

class GhostAI:
    """AI that learns by watching and copying player inputs"""
    
    def __init__(self, character: str = "ken"):
        self.character = character
        self.last_action_time = 0
        self.action_cooldown = 0.016  # ~1 frame
        
        # Pattern database: situation -> list of actions
        self.patterns: Dict[str, List[RecordedAction]] = defaultdict(list)
        self.situation_frequency: Dict[str, int] = defaultdict(int)
        
        # Learning parameters
        self.min_examples = 2  # Only need 2 examples to start using a pattern
        self.randomness = 0.15  # 15% chance to pick random learned action
        
        # Stats
        self.total_actions_learned = 0
        self.unique_situations = 0
        
        # Save directory
        self.save_dir = Path("ghost_data")
        self.save_dir.mkdir(exist_ok=True)
    
    def learn_from_recording(self, recorded_actions: List[RecordedAction]):
        """Learn patterns from recorded session"""
        print(f"\n🧠 LEARNING FROM RECORDING...")
        print(f"   Processing {len(recorded_actions)} actions...")
        
        for action in recorded_actions:
            self.patterns[action.situation].append(action)
            self.situation_frequency[action.situation] += 1
        
        self.total_actions_learned = len(recorded_actions)
        self.unique_situations = len(self.patterns)
        
        print(f"   ✓ Learned {self.unique_situations} unique situations")
        print(f"   ✓ Total actions in memory: {self.total_actions_learned}")
        
        # Show most common situations
        print(f"\n📊 TOP SITUATIONS LEARNED:")
        top_situations = sorted(self.situation_frequency.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        for i, (situation, count) in enumerate(top_situations, 1):
            print(f"   {i}. {situation} → {count} times")
    
    def decide_action(self, game_state: GameState, history: List[GameState]) -> Optional[ActionCommand]:
        """Decide what to do based on learned patterns"""
        current_time = game_state.timestamp
        
        # Cooldown
        if current_time - self.last_action_time < self.action_cooldown:
            return None
        
        # Get current situation
        situation = game_state.to_situation_key()
        
        # Check if we have learned this exact situation
        if situation in self.patterns and len(self.patterns[situation]) >= self.min_examples:
            import random
            
            # Pick action (with some randomness for variety)
            if random.random() > self.randomness:
                action = random.choice(self.patterns[situation])
            else:
                # Pick from any learned action
                all_actions = [a for actions in self.patterns.values() for a in actions]
                action = random.choice(all_actions)
            
            self.last_action_time = current_time
            
            # Keys are already P2 controls, use them directly
            return ActionCommand(
                inputs=action.keys_pressed,
                timing=current_time,
                duration=min(action.duration, 0.2),  # Cap duration
                priority=5
            )
        
        # Try fuzzy matching for similar situations
        similar_action = self._find_similar_situation(game_state)
        if similar_action:
            self.last_action_time = current_time
            return similar_action
        
        # No pattern found - do nothing (safer than random)
        return None
    
    def _find_similar_situation(self, game_state: GameState) -> Optional[ActionCommand]:
        """Find actions from similar game states"""
        current_health = game_state.player1_health
        current_dist = game_state.distance
        
        # Look for situations within tolerance
        best_match = None
        best_score = float('inf')
        
        for situation_key, actions in self.patterns.items():
            if len(actions) < self.min_examples:
                continue
                
            # Parse situation
            try:
                parts = situation_key.split('_')
                health, opp_health, dist, super_meter = map(float, parts)
                
                # Calculate similarity score (lower is better)
                score = (
                    abs(health - current_health) * 2 +  # Health difference weighted more
                    abs(dist - current_dist) / 100 +     # Distance difference
                    abs(opp_health - game_state.player2_health)
                )
                
                if score < best_score and score < 0.5:  # Only if reasonably similar
                    best_score = score
                    best_match = actions
                    
            except:
                continue
        
        if best_match:
            import random
            action = random.choice(best_match)
            
            # Keys are already P2 controls
            return ActionCommand(
                inputs=action.keys_pressed,
                timing=time.time(),
                duration=min(action.duration, 0.2),
                priority=3
            )
        
        return None
    
    def save_patterns(self, session_name: str = None):
        """Save learned patterns"""
        if session_name is None:
            session_name = f"{self.character}_session_{int(time.time())}"
        
        filepath = self.save_dir / f"{session_name}.pkl"
        
        data = {
            'patterns': dict(self.patterns),
            'frequency': dict(self.situation_frequency),
            'character': self.character,
            'total_actions': self.total_actions_learned,
            'unique_situations': self.unique_situations
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"\n💾 Saved patterns to: {filepath.name}")
        return filepath.name
    
    def load_patterns(self, filename: str):
        """Load saved patterns"""
        filepath = self.save_dir / filename
        
        if not filepath.exists():
            print(f"❌ File not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.patterns = defaultdict(list, data['patterns'])
        self.situation_frequency = defaultdict(int, data['frequency'])
        self.character = data.get('character', 'unknown')
        self.total_actions_learned = data.get('total_actions', 0)
        self.unique_situations = data.get('unique_situations', 0)
        
        print(f"\n✅ LOADED: {filename}")
        print(f"   Character: {self.character}")
        print(f"   Situations: {self.unique_situations}")
        print(f"   Total actions: {self.total_actions_learned}")
        
        return True
    
    def list_saved_ghosts(self):
        """List available saved ghosts"""
        pattern_files = sorted(list(self.save_dir.glob("*.pkl")))
        
        if not pattern_files:
            print("\n📁 No saved ghosts found")
            return []
        
        print("\n📁 SAVED GHOSTS:")
        for i, pfile in enumerate(pattern_files, 1):
            try:
                with open(pfile, 'rb') as f:
                    data = pickle.load(f)
                char = data.get('character', '?')
                situations = data.get('unique_situations', 0)
                print(f"   {i}. {pfile.name}")
                print(f"      └─ {char} | {situations} situations")
            except:
                print(f"   {i}. {pfile.name} (corrupted)")
        
        return pattern_files

class GameStateExtractor:
    def extract_state(self) -> Optional[GameState]:
        raise NotImplementedError

class FightcadeExtractor(GameStateExtractor):
    """Extract game state - placeholder until CV is implemented"""
    def extract_state(self) -> Optional[GameState]:
        # TODO: Implement actual screen reading
        # For now, return dummy data so recording/playback works
        return GameState(
            timestamp=time.time(),
            player1_health=1.0,
            player2_health=1.0,
            player1_super=0.0,
            player2_super=0.0,
            player1_position=(160, 400),
            player2_position=(480, 400),
            distance=320,
            round_timer=99.0
        )

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
        # P2 controls mapping - exactly as you specified
        self.key_mappings = {
            "w": "w",      # Up
            "s": "s",      # Down
            "a": "a",      # Left
            "d": "d",      # Right
            "y": "y",      # Weak Punch
            "u": "u",      # Medium Punch
            "i": "i",      # Strong Punch
            "h": "h",      # Weak Kick
            "j": "j",      # Medium Kick
            "k": "k",      # Heavy Kick
            "6": "6",      # Coin/Select
            "2": "2",      # Start
        }
        self.window_id = self._get_window_id()

    def _get_window_id(self):
        """Find 3rd Strike window specifically"""
        try:
            # Look for 3rd Strike specifically
            result = subprocess.run(
                ["xdotool", "search", "--name", "Street Fighter III.*3rd Strike"],
                capture_output=True, text=True, check=True
            )
            window_ids = result.stdout.strip().split("\n")
            if window_ids:
                print(f"✓ Found 3rd Strike window: {window_ids[0]}")
                return window_ids[0]
        except subprocess.CalledProcessError:
            print("❌ Could not find 3rd Strike window")
            print("   Make sure Fightcade is running with 3rd Strike loaded!")
        
        return None

    def _execute_action(self, action: ActionCommand):
        if not self.window_id:
            return

        for key in action.inputs:
            mapped = self.key_mappings.get(key, key)
            try:
                # Press and hold for duration
                subprocess.run(
                    ["xdotool", "key", "--window", self.window_id, 
                     "--delay", str(int(action.duration * 1000)), mapped],
                    check=True, capture_output=True
                )
            except Exception as e:
                print(f"Input error: {e}")

class GhostBot:
    """Main bot that orchestrates everything"""
    
    def __init__(self):
        self.extractor = FightcadeExtractor()
        self.ghost_ai = GhostAI()
        self.executor = LinuxInputExecutor()
        self.recorder = InputRecorder()
        
        self.running = False
        self.mode = None  # 'record' or 'play'
        self.history = []
    
    def record_session(self):
        """Record a training session"""
        print("\n" + "="*60)
        print("📼 RECORDING SESSION")
        print("="*60)
        print("Play your matches normally on P2 controls:")
        print("  Movement: WASD")
        print("  Punches: Y (Weak) U (Medium) I (Strong)")
        print("  Kicks: H (Weak) J (Medium) K (Heavy)")
        print("\nThe ghost will watch and learn your patterns")
        print("Press Ctrl+C when finished\n")
        
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
                time.sleep(0.016)  # ~60 FPS
                
        except KeyboardInterrupt:
            print("\n\n⏹️  Recording stopped!")
        
        # Process recording
        recorded = self.recorder.stop_recording()
        if recorded:
            self.ghost_ai.learn_from_recording(recorded)
            
            save = input("\nSave this session? (y/n): ").lower()
            if save == 'y':
                name = input("Session name (or press enter for auto): ").strip()
                self.ghost_ai.save_patterns(name if name else None)
    
    def play_session(self):
        """Let the ghost play"""
        print("\n" + "="*60)
        print("🎮 GHOST PLAYING")
        print("="*60)
        
        if self.ghost_ai.unique_situations == 0:
            print("❌ No patterns loaded! Record or load a session first.")
            return
        
        print(f"Ghost is playing with {self.ghost_ai.unique_situations} learned situations")
        print("Press Ctrl+C to stop\n")
        
        self.mode = 'play'
        self.running = True
        
        # Start executor thread
        executor_thread = threading.Thread(target=self.executor.execute_inputs, daemon=True)
        executor_thread.start()
        
        try:
            while self.running:
                state = self.extractor.extract_state()
                if state:
                    self.history.append(state)
                    if len(self.history) > 100:
                        self.history.pop(0)
                    
                    action = self.ghost_ai.decide_action(state, self.history)
                    if action:
                        self.executor.queue_action(action)
                
                time.sleep(0.016)
                
        except KeyboardInterrupt:
            print("\n⏹️  Stopped")
        
        self.executor.stop()

def main():
    """Main menu"""
    print("="*60)
    print("🥋 STREET FIGHTER III: 3RD STRIKE - GHOST AI")
    print("="*60)
    print("Tekken-style ghost that learns from watching you play\n")
    
    bot = GhostBot()
    
    while True:
        print("\n" + "─"*60)
        print("MENU:")
        print("  1. Record new session (play and teach the ghost)")
        print("  2. Ghost plays (uses learned patterns)")
        print("  3. Load saved ghost")
        print("  4. View saved ghosts")
        print("  5. Exit")
        print("─"*60)
        
        choice = input("\nChoice: ").strip()
        
        if choice == "1":
            bot.record_session()
            
        elif choice == "2":
            bot.play_session()
            
        elif choice == "3":
            files = bot.ghost_ai.list_saved_ghosts()
            if files:
                try:
                    idx = int(input("\nSelect number: ")) - 1
                    if 0 <= idx < len(files):
                        bot.ghost_ai.load_patterns(files[idx].name)
                except (ValueError, IndexError):
                    print("❌ Invalid selection")
                    
        elif choice == "4":
            bot.ghost_ai.list_saved_ghosts()
            
        elif choice == "5":
            print("\n👋 Later!")
            break
            
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    main()
