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
        """Convert state to a situation key for pattern matching"""
        # Discretize continuous values for pattern matching
        health_bucket = round(self.player1_health * 4) / 4  # 0, 0.25, 0.5, 0.75, 1.0
        opp_health_bucket = round(self.player2_health * 4) / 4
        dist_bucket = int(self.distance / 100) * 100  # 0, 100, 200, 300, etc.
        super_bucket = round(self.player1_super * 4) / 4
        
        return f"{health_bucket}_{opp_health_bucket}_{dist_bucket}_{super_bucket}"

@dataclass
class PlayerAction:
    """Represents what a player actually did"""
    timestamp: float
    keys_pressed: List[str]
    duration: float
    game_state: GameState

class ActionCommand:
    def __init__(self, inputs: List[str], timing: float, duration: float, priority: int):
        self.inputs = inputs
        self.timing = timing
        self.duration = duration
        self.priority = priority

class InputRecorder:
    """Records player inputs in real-time"""
    
    def __init__(self):
        self.recording = False
        self.recorded_actions = []
        self.key_states = {}
        self.last_key_change = time.time()
        
    def start_recording(self):
        """Start recording inputs"""
        self.recording = True
        self.recorded_actions = []
        print("🔴 Recording started - play some matches!")
        
    def stop_recording(self):
        """Stop recording and save"""
        self.recording = False
        print(f"⏹️  Recording stopped - captured {len(self.recorded_actions)} actions")
        return self.recorded_actions
    
    def record_keypress(self, key: str, pressed: bool, game_state: GameState):
        """Record a key press/release event"""
        if not self.recording:
            return
            
        current_time = time.time()
        
        if pressed:
            self.key_states[key] = current_time
        else:
            if key in self.key_states:
                start_time = self.key_states[key]
                duration = current_time - start_time
                
                # Record the action
                action = PlayerAction(
                    timestamp=start_time,
                    keys_pressed=[key],
                    duration=duration,
                    game_state=game_state
                )
                self.recorded_actions.append(action)
                
                del self.key_states[key]

class GhostAI:
    """AI that learns from recorded player sessions"""
    
    def __init__(self, character: str = "ken"):
        self.character = character
        self.last_action_time = 0
        self.action_cooldown = 0.016
        
        # Pattern database: situation -> list of actions taken in that situation
        self.patterns: Dict[str, List[PlayerAction]] = defaultdict(list)
        self.situation_counts: Dict[str, int] = defaultdict(int)
        
        # Learning parameters
        self.min_confidence = 3  # Need at least 3 examples to use a pattern
        self.exploration_rate = 0.1  # 10% chance to try something random
        
        # Save directory
        self.save_dir = Path("ghost_data")
        self.save_dir.mkdir(exist_ok=True)
        
    def learn_from_recording(self, recorded_actions: List[PlayerAction]):
        """Learn patterns from recorded gameplay"""
        print(f"🧠 Learning from {len(recorded_actions)} recorded actions...")
        
        for action in recorded_actions:
            situation_key = action.game_state.to_situation_key()
            self.patterns[situation_key].append(action)
            self.situation_counts[situation_key] += 1
        
        print(f"📊 Learned {len(self.patterns)} unique situations")
        print(f"🎯 Most common situations:")
        
        # Show top 5 most common situations
        top_situations = sorted(self.situation_counts.items(), 
                              key=lambda x: x[1], reverse=True)[:5]
        for situation, count in top_situations:
            print(f"   {situation}: {count} times")
        
        self.save_patterns()
    
    def decide_action(self, game_state: GameState, history: List[GameState]) -> Optional[ActionCommand]:
        """Decide action based on learned patterns"""
        current_time = game_state.timestamp
        
        # Cooldown check
        if current_time - self.last_action_time < self.action_cooldown:
            return None
        
        # Get current situation
        situation_key = game_state.to_situation_key()
        
        # Check if we have learned patterns for this situation
        if situation_key in self.patterns:
            confidence = self.situation_counts[situation_key]
            
            # Only use pattern if we have enough examples
            if confidence >= self.min_confidence:
                # Pick a random action from similar situations (adds variety)
                import random
                
                # Occasionally explore (try something different)
                if random.random() > self.exploration_rate:
                    action_data = random.choice(self.patterns[situation_key])
                    
                    self.last_action_time = current_time
                    return ActionCommand(
                        inputs=action_data.keys_pressed,
                        timing=current_time,
                        duration=action_data.duration,
                        priority=5
                    )
        
        # Fallback: check similar situations (fuzzy match)
        similar_action = self._find_similar_situation(game_state)
        if similar_action:
            self.last_action_time = current_time
            return similar_action
        
        # Last resort: basic behavior
        return self._fallback_behavior(game_state, current_time)
    
    def _find_similar_situation(self, game_state: GameState) -> Optional[ActionCommand]:
        """Find actions from similar game states"""
        current_health = game_state.player1_health
        current_distance = game_state.distance
        
        # Look for situations within tolerance ranges
        tolerance_health = 0.25
        tolerance_distance = 150
        
        matching_actions = []
        
        for situation_key, actions in self.patterns.items():
            # Parse situation key
            parts = situation_key.split('_')
            if len(parts) != 4:
                continue
                
            health, opp_health, dist, super_meter = map(float, parts)
            
            # Check if situation is similar
            if (abs(health - current_health) <= tolerance_health and
                abs(dist - current_distance) <= tolerance_distance):
                matching_actions.extend(actions)
        
        if matching_actions:
            import random
            action_data = random.choice(matching_actions)
            return ActionCommand(
                inputs=action_data.keys_pressed,
                timing=time.time(),
                duration=action_data.duration,
                priority=3
            )
        
        return None
    
    def _fallback_behavior(self, game_state: GameState, current_time: float) -> Optional[ActionCommand]:
        """Basic fallback when no patterns match"""
        distance = game_state.distance
        
        # Very basic spacing
        if distance > 250:
            return ActionCommand(inputs=['right'], timing=current_time, duration=0.1, priority=1)
        elif distance < 50:
            return ActionCommand(inputs=['left'], timing=current_time, duration=0.1, priority=1)
        
        return None
    
    def save_patterns(self, filename: str = None):
        """Save learned patterns to disk"""
        if filename is None:
            filename = f"{self.character}_ghost_{int(time.time())}.pkl"
        
        filepath = self.save_dir / filename
        
        data = {
            'patterns': dict(self.patterns),
            'situation_counts': dict(self.situation_counts),
            'character': self.character
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        
        print(f"💾 Patterns saved to {filepath}")
    
    def load_patterns(self, filename: str):
        """Load previously learned patterns"""
        filepath = self.save_dir / filename
        
        if not filepath.exists():
            print(f"❌ Pattern file not found: {filepath}")
            return False
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.patterns = defaultdict(list, data['patterns'])
        self.situation_counts = defaultdict(int, data['situation_counts'])
        self.character = data['character']
        
        print(f"✅ Loaded {len(self.patterns)} patterns from {filepath}")
        return True
    
    def list_saved_patterns(self):
        """List all saved pattern files"""
        pattern_files = list(self.save_dir.glob("*.pkl"))
        
        if not pattern_files:
            print("No saved patterns found")
            return []
        
        print("\n📁 Available ghost patterns:")
        for i, pfile in enumerate(pattern_files, 1):
            size = len(pickle.load(open(pfile, 'rb'))['patterns'])
            print(f"   {i}. {pfile.name} ({size} situations)")
        
        return pattern_files

class GameStateExtractor:
    def extract_state(self) -> Optional[GameState]:
        raise NotImplementedError

class FightcadeExtractor(GameStateExtractor):
    """
    TODO: Implement actual screen capture and CV detection
    For now returns dummy data - you'll need to implement:
    - Health bar detection
    - Character position detection  
    - Super meter reading
    - Distance calculation
    """
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
        self.key_mappings = {
            "up": "w",
            "down": "s",
            "left": "a",
            "right": "d",
            "lp": "y",
            "mp": "u",
            "hp": "i",
            "lk": "h",
            "mk": "j",
            "hk": "k",
            "coin": "6",
            "start": "2"
        }
        self.window_id = self._get_window_id()

    def _get_window_id(self):
        try:
            result = subprocess.run(
                ["xdotool", "search", "--onlyvisible", "name", "3rd Strike"],
                capture_output=True, text=True, check=True
            )
            return result.stdout.strip().split("\n")[0]
        except Exception as e:
            print(f"Could not find window: {e}")
            return None

    def _execute_action(self, action: ActionCommand):
        if not self.window_id:
            return

        for key in action.inputs:
            mapped = self.key_mappings.get(key, key)
            try:
                subprocess.run(
                    ["xdotool", "windowfocus", self.window_id, "key", mapped],
                    check=True
                )
                time.sleep(action.duration / len(action.inputs))
            except Exception as e:
                print(f"xdotool failed: {e}")

class FightingGameBot:
    def __init__(self, extractor: GameStateExtractor, ai: GhostAI):
        self.extractor = extractor
        self.ai = ai
        self.executor = LinuxInputExecutor()
        self.running = False
        self.history = []
        self.recorder = InputRecorder()
        
        # Mode: 'record' or 'playback'
        self.mode = 'playback'

    def start_recording_mode(self):
        """Start bot in recording mode to learn from player"""
        print("📼 RECORDING MODE")
        print("Play some matches - the bot will watch and learn!")
        print("Press Ctrl+C when done recording")
        
        self.mode = 'record'
        self.recorder.start_recording()
        self.running = True
        
        # Start monitoring (without executing actions)
        self.run_loop()
    
    def start_playback_mode(self):
        """Start bot in playback mode using learned patterns"""
        print("🎮 PLAYBACK MODE")
        print("Bot is now playing using learned patterns!")
        
        self.mode = 'playback'
        self.running = True
        
        # Start executor thread
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
                
                if self.mode == 'playback':
                    # AI makes decisions
                    action = self.ai.decide_action(state, self.history)
                    if action:
                        self.executor.queue_action(action)
                elif self.mode == 'record':
                    # Just observe (recording happens via keyboard hooks in real implementation)
                    pass
                    
            time.sleep(0.01)

    def stop(self):
        print("Stopping bot...")
        self.running = False
        self.executor.stop()
        
        if self.mode == 'record':
            recorded = self.recorder.stop_recording()
            if recorded:
                self.ai.learn_from_recording(recorded)

def interactive_menu():
    """Interactive menu for bot control"""
    print("=" * 60)
    print("🥋 STREET FIGHTER III: 3RD STRIKE - GHOST AI")
    print("=" * 60)
    
    ghost_ai = GhostAI(character="ken")
    extractor = FightcadeExtractor()
    bot = FightingGameBot(extractor, ghost_ai)
    
    while True:
        print("\n📋 MENU:")
        print("  1. Record new session (learn from your gameplay)")
        print("  2. Play with ghost AI (use learned patterns)")
        print("  3. Load existing ghost patterns")
        print("  4. View saved patterns")
        print("  5. Exit")
        
        choice = input("\nChoice (1-5): ").strip()
        
        if choice == "1":
            print("\n" + "="*60)
            try:
                bot.start_recording_mode()
            except KeyboardInterrupt:
                bot.stop()
                print("\n✅ Recording saved!")
                
        elif choice == "2":
            print("\n" + "="*60)
            pattern_count = len(ghost_ai.patterns)
            if pattern_count == 0:
                print("⚠️  No patterns loaded! Record a session first or load existing patterns.")
                continue
            
            print(f"🎯 Using {pattern_count} learned situations")
            try:
                bot.start_playback_mode()
            except KeyboardInterrupt:
                bot.stop()
                print("\n⏹️  Playback stopped")
                
        elif choice == "3":
            files = ghost_ai.list_saved_patterns()
            if files:
                idx = input("Select file number: ").strip()
                try:
                    ghost_ai.load_patterns(files[int(idx)-1].name)
                except (ValueError, IndexError):
                    print("❌ Invalid selection")
                    
        elif choice == "4":
            ghost_ai.list_saved_patterns()
            
        elif choice == "5":
            print("👋 Goodbye!")
            break
        else:
            print("❌ Invalid choice")

if __name__ == "__main__":
    interactive_menu()
