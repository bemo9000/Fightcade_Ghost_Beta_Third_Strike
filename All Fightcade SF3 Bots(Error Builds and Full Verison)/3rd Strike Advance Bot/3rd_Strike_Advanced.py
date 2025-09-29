
#!/usr/bin/env python3
"""
3rd_Strike_Advanced.py
Advanced Fightcade SFIII bot: CLI flags, debug logging, vision-capable extractor (optional),
smart save names, and matchup-aware ghost storage.

Notes:
 - Vision extractor requires `mss` (for screenshots) and `numpy`/`opencv-python` for processing.
 - If `mss` or window geometry detection fails, the script falls back to a dummy extractor (useful for testing).
 - Interacting with the real game requires correct window name matching (Fightcade / 3rd Strike).

Usage examples:
  python3 3rd_Strike_Advanced.py --character ken --opponent chunli --mode record --debug
  python3 3rd_Strike_Advanced.py --mode play --debug
"""

import argparse
import logging
import time
import threading
import queue
import subprocess
import pickle
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from collections import defaultdict
import random
import datetime

# Optional vision libs
try:
    import cv2
    import numpy as np
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False

try:
    import mss
    import mss.tools
    HAS_MSS = True
except Exception:
    HAS_MSS = False

# Optional keyboard listener (for recording)
try:
    from pynput import keyboard as kb
    HAS_PYNPUT = True
except Exception:
    HAS_PYNPUT = False


# ---------------------- Basic data classes ----------------------
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
    situation: str
    keys_pressed: List[str]
    duration: float
    timestamp: float


# ---------------------- Input recorder (pynput optional) ----------------------
class InputRecorder:
    def __init__(self, watch_keys: Optional[set] = None):
        self.recording = False
        self.recorded_actions: List[RecordedAction] = []
        self.current_keys_held = {}
        self.current_game_state = None
        self.listener = None
        self.watch_keys = watch_keys or {'w', 'a', 's', 'd', 'y', 'u', 'i', 'h', 'j', 'k'}
        if not HAS_PYNPUT:
            logging.warning("pynput not available: live recording will not work. Install pynput to enable it.")

    def start_recording(self):
        if not HAS_PYNPUT:
            raise RuntimeError("pynput not available - cannot start input recording.")
        self.recording = True
        self.recorded_actions = []
        logging.info("🔴 RECORDING - Play and I'll learn from you...")
        self.listener = kb.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()

    def stop_recording(self):
        self.recording = False
        if self.listener:
            self.listener.stop()
        logging.info(f"⏹️  Recorded {len(self.recorded_actions)} actions")
        return self.recorded_actions

    def update_game_state(self, state: GameState):
        self.current_game_state = state

    def _normalize_key(self, key) -> Optional[str]:
        try:
            if hasattr(key, 'char') and key.char:
                return key.char.lower()
            return None
        except Exception:
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


# ---------------------- Ghost AI (matchup-aware) ----------------------
class GhostAI:
    def __init__(self, character: str = "ken", opponent: str = "ryu", min_examples: int = 2, randomness: float = 0.15, save_root: Path = Path("ghost_data")):
        self.character = character.lower()
        self.opponent = opponent.lower()
        self.matchup = f"{self.character}_vs_{self.opponent}"
        self.last_action_time = 0.0
        self.action_cooldown = 0.016
        self.patterns: Dict[str, List[RecordedAction]] = defaultdict(list)
        self.situation_counts: Dict[str, int] = defaultdict(int)
        self.min_examples = min_examples
        self.randomness = randomness
        self.save_root = Path(save_root)
        self.save_dir = self.save_root / self.matchup
        self.save_dir.mkdir(parents=True, exist_ok=True)

    def learn_from_recording(self, recorded_actions: List[RecordedAction]):
        logging.info(f"🧠 Learning from {len(recorded_actions)} actions...")
        for action in recorded_actions:
            self.patterns[action.situation].append(action)
            self.situation_counts[action.situation] += 1
        logging.info(f"✓ Learned {len(self.patterns)} unique situations for {self.matchup}")

    def decide_action(self, game_state: GameState, history: List[GameState]) -> Optional[ActionCommand]:
        current_time = game_state.timestamp
        if current_time - self.last_action_time < self.action_cooldown:
            return None
        situation = game_state.to_situation_key()

        # exact or near-exact match
        if situation in self.patterns and len(self.patterns[situation]) >= self.min_examples:
            if random.random() > self.randomness:
                action = random.choice(self.patterns[situation])
                logging.debug(f"Exact match: {situation} -> {action.keys_pressed}")
            else:
                all_actions = [a for acts in self.patterns.values() for a in acts]
                if all_actions:
                    action = random.choice(all_actions)
                else:
                    return None
            self.last_action_time = current_time
            return ActionCommand(inputs=action.keys_pressed, timing=current_time, duration=min(action.duration, 0.2), priority=5)

        # fuzzy
        best = self._find_similar(game_state)
        if best:
            self.last_action_time = current_time
            logging.debug(f"Fuzzy match used for situation {situation}")
            return best
        return None

    def _find_similar(self, game_state: GameState) -> Optional[ActionCommand]:
        curr_health = game_state.player2_health
        curr_dist = game_state.distance
        best_match = None
        best_score = float('inf')
        for situation_key, actions in self.patterns.items():
            if len(actions) < self.min_examples:
                continue
            try:
                parts = situation_key.split('_')
                health, opp_health, dist, super_meter = map(float, parts)
                score = abs(health - curr_health) * 2 + abs(dist - curr_dist) / 100
                if score < best_score and score < 0.5:
                    best_score = score
                    best_match = actions
            except Exception:
                continue
        if best_match:
            action = random.choice(best_match)
            return ActionCommand(inputs=action.keys_pressed, timing=time.time(), duration=min(action.duration, 0.2), priority=3)
        return None

    def _smart_name(self, base_name: str = None, extra: str = "") -> str:
        now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        count = sum(1 for _ in self.save_dir.glob("*.pkl"))
        base = base_name or f"{self.matchup}_ghost_{now}"
        if extra:
            base = f"{base}_{extra}"
        return f"{base}_{count+1}.pkl"

    def save_patterns(self, name: str = None, extra_note: str = "", records_count: Optional[int] = None):
        filename = name or self._smart_name(extra=extra_note)
        filepath = self.save_dir / filename
        data = {
            'patterns': dict(self.patterns),
            'counts': dict(self.situation_counts),
            'character': self.character,
            'opponent': self.opponent,
            'saved_at': time.time(),
            'records_count': records_count or sum(len(v) for v in self.patterns.values())
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
        logging.info(f"💾 Saved matchup data to {filepath}")

    def load_patterns(self, filename: str) -> bool:
        filepath = self.save_dir / filename
        if not filepath.exists():
            logging.error(f"❌ File not found: {filepath}")
            return False
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        self.patterns = defaultdict(list, data.get('patterns', {}))
        self.situation_counts = defaultdict(int, data.get('counts', {}))
        logging.info(f"✅ Loaded {len(self.patterns)} patterns from {filename}")
        return True

    def list_ghosts(self) -> List[Path]:
        files = sorted(list(self.save_dir.glob("*.pkl")))
        if not files:
            logging.info(f"📁 No saved ghosts for {self.matchup}")
            return []
        logging.info(f"\n📁 SAVED GHOSTS for {self.matchup}:")
        for i, f in enumerate(files, 1):
            try:
                data = pickle.load(open(f, 'rb'))
                logging.info(f"  {i}. {f.name} - {len(data.get('patterns', {}))} situations")
            except Exception:
                logging.info(f"  {i}. {f.name} (corrupted)")
        return files


# ---------------------- Extractors ----------------------
class GameStateExtractor:
    def extract_state(self) -> Optional[GameState]:
        raise NotImplementedError


class DummyExtractor(GameStateExtractor):
    def extract_state(self) -> Optional[GameState]:
        # Basic dummy data so the bot can run without vision: static centered positions.
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


class VisionExtractor(GameStateExtractor):
    """
    Attempt to grab Fightcade/3rd Strike window using xdotool to get window geometry and mss to screenshot.
    Then parse a few indicators (health bars, super meters, simple x positions).

    This implementation is intentionally conservative and robust:
      - If any required library isn't present, it raises/returns None and user falls back to DummyExtractor.
      - Parsing routines are simple color/threshold heuristics; you'll likely need to tune regions for your setup.
    """
    def __init__(self, window_name_fuzzy: str = "Fightcade"):
        if not HAS_MSS or not HAS_CV2:
            raise RuntimeError("Vision extractor requires mss and cv2 (opencv).")
        self.window_name_fuzzy = window_name_fuzzy
        self.sct = mss.mss()
        self.window_id = self._find_window_id()
        self.geom = self._get_window_geometry() if self.window_id else None
        if not self.geom:
            raise RuntimeError("Could not determine window geometry for vision extractor.")

    def _find_window_id(self) -> Optional[str]:
        try:
            result = subprocess.run(["xdotool", "search", "--name", self.window_name_fuzzy], capture_output=True, text=True, check=True)
            ids = [i for i in result.stdout.strip().splitlines() if i.strip()]
            if not ids:
                return None
            return ids[0]
        except Exception:
            return None

    def _get_window_geometry(self) -> Optional[Dict[str, int]]:
        # Uses xwd/xwininfo via xdotool getwindowgeometry for geometry
        try:
            r = subprocess.run(["xdotool", "getwindowgeometry", "--shell", self.window_id], capture_output=True, text=True, check=True)
            lines = r.stdout.splitlines()
            geom = {}
            for line in lines:
                if "=" in line:
                    k, v = line.split("=", 1)
                    geom[k.strip()] = int(v.strip())
            # geom now contains WIDTH, HEIGHT, X, Y if available
            return {"left": geom.get("X", 0), "top": geom.get("Y", 0), "width": geom.get("WIDTH", 640), "height": geom.get("HEIGHT", 480)}
        except Exception:
            return None

    def _grab(self):
        r = {"top": self.geom["top"], "left": self.geom["left"], "width": self.geom["width"], "height": self.geom["height"]}
        img = self.sct.grab(r)
        arr = np.array(img)
        # convert BGRA -> BGR
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    def _read_health_percentage(self, img, left_region=True) -> float:
        """
        Heuristic: assume health bar is a solid colored horizontal bar at a known relative Y.
        This is very dependent on your window/layout; tune these values for accuracy.
        Returns 0.0 - 1.0
        """
        h, w, _ = img.shape
        # relative regions (these might need tuning)
        y = int(h * 0.05)  # top small margin
        hbar_h = int(h * 0.03)
        if left_region:
            x1, x2 = int(w * 0.05), int(w * 0.35)
        else:
            x1, x2 = int(w * 0.65), int(w * 0.95)
        crop = img[y:y + hbar_h, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        # pick red-ish / green-ish depending on healthbar design - generic approach: compute mean non-black
        v = hsv[:, :, 2].astype(float)
        filled = np.mean(v) / 255.0
        # clamp
        return max(0.0, min(1.0, filled))

    def extract_state(self) -> Optional[GameState]:
        try:
            img = self._grab()
            # crude estimates: left health = player1, right health = player2
            p1 = self._read_health_percentage(img, left_region=True)
            p2 = self._read_health_percentage(img, left_region=False)
            # positions: try to find bright sprites via simple threshold (placeholder)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]
            cols = np.sum(thresh, axis=0)
            if np.max(cols) > 0:
                left_x = int(np.argmax(cols))
                right_x = int(len(cols) - np.argmax(cols[::-1]))
                dist = abs(right_x - left_x)
            else:
                left_x = int(img.shape[1] * 0.25)
                right_x = int(img.shape[1] * 0.75)
                dist = right_x - left_x
            return GameState(
                timestamp=time.time(),
                player1_health=float(p1),
                player2_health=float(p2),
                player1_super=0.0,
                player2_super=0.0,
                player1_position=(left_x, int(img.shape[0] * 0.5)),
                player2_position=(right_x, int(img.shape[0] * 0.5)),
                player1_state="unknown",
                player2_state="unknown",
                distance=float(dist),
                frame_advantage=0,
                round_timer=99.0
            )
        except Exception as e:
            logging.exception("Vision extractor failed: %s", e)
            return None


# ---------------------- Simple AI (hardcoded heuristic) ----------------------
class Fightcade3rdStrikeAI:
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


# ---------------------- Input executor (xdotool) ----------------------
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
    def __init__(self, window_name_fuzzy: str = "Fightcade"):
        super().__init__()
        self.key_mappings = {
            "up": "w", "down": "s", "left": "a", "right": "d",
            "lp": "y", "mp": "u", "hp": "i", "lk": "h", "mk": "j", "hk": "k",
            "coin": "6", "start": "2"
        }
        self.window_name_fuzzy = window_name_fuzzy
        self.window_id = self._find_3rd_strike_window()
        if self.window_id:
            logging.info(f"✓ Found 3rd Strike window: {self.window_id}")
        else:
            logging.warning("❌ Could not find 3rd Strike window!")

    def _find_3rd_strike_window(self):
        try:
            result = subprocess.run(["xdotool", "search", "--name", self.window_name_fuzzy], capture_output=True, text=True, check=True)
            ids = [i for i in result.stdout.strip().splitlines() if i.strip()]
            return ids[0] if ids else None
        except Exception:
            return None

    def _execute_action(self, action: ActionCommand):
        if not self.window_id:
            return
        for key in action.inputs:
            mapped = self.key_mappings.get(key, key)
            try:
                subprocess.run(["xdotool", "key", "--window", self.window_id, mapped], check=True, capture_output=True)
                time.sleep(action.duration / max(1, len(action.inputs)))
            except Exception as e:
                logging.debug("xdotool failed: %s", e)


# ---------------------- Bot controller ----------------------
class FightingGameBot:
    def __init__(self, extractor: GameStateExtractor, ai, executor: InputExecutor = None, recorder: InputRecorder = None, debug: bool = False):
        self.extractor = extractor
        self.ai = ai
        self.executor = executor or LinuxInputExecutor()
        self.recorder = recorder or InputRecorder()
        self.running = False
        self.history: List[GameState] = []
        self.mode = 'playback'
        self.debug = debug

    def start_recording(self):
        if not self.executor.window_id:
            logging.error("❌ Cannot start - no window detected!")
            return
        logging.info("\n📼 RECORDING MODE")
        logging.info("Play matches on P2 controls (WASD + YUIHJK)")
        logging.info("Press Ctrl+C when done\n")
        self.mode = 'record'
        if not HAS_PYNPUT:
            logging.error("pynput not available - cannot record inputs.")
            return
        self.recorder.start_recording()
        self.running = True
        try:
            while self.running:
                state = self.extractor.extract_state()
                if state:
                    self.recorder.update_game_state(state)
                    self.history.append(state)
                    if len(self.history) > 500:
                        self.history.pop(0)
                time.sleep(0.016)
        except KeyboardInterrupt:
            logging.info("\n⏹️  Recording stopped!")
        recorded = self.recorder.stop_recording()
        if recorded and isinstance(self.ai, GhostAI):
            self.ai.learn_from_recording(recorded)
            save = input("\nSave this session? (y/n): ").lower()
            if save == 'y':
                extra = input("Optional note for filename (or enter): ").strip()
                self.ai.save_patterns(extra_note=extra, records_count=len(recorded))

    def start(self):
        if not self.executor.window_id:
            logging.error("❌ Cannot start - no window detected!")
            return
        logging.info("Starting bot...")
        self.running = True
        executor_thread = threading.Thread(target=self.executor.execute_inputs, daemon=True)
        executor_thread.start()
        self.run_loop()

    def run_loop(self):
        while self.running:
            state = self.extractor.extract_state()
            if state:
                self.history.append(state)
                if len(self.history) > 500:
                    self.history.pop(0)
                action = self.ai.decide_action(state, self.history)
                if action:
                    if self.debug:
                        logging.debug(f"DECIDE [{state.to_situation_key()}] -> {action.inputs} dur={action.duration} pri={action.priority}")
                    self.executor.queue_action(action)
            time.sleep(0.01)

    def stop(self):
        logging.info("Stopping bot...")
        self.running = False
        self.executor.stop()


# ---------------------- CLI / Entrypoint ----------------------
def build_arg_parser():
    p = argparse.ArgumentParser(description="3rd Strike Advanced Bot")
    p.add_argument("--character", type=str, default="ken", help="Your character (e.g. ken)")
    p.add_argument("--opponent", type=str, default="ryu", help="Opponent character")
    p.add_argument("--mode", type=str, default="simple", choices=["simple", "record", "play", "load", "view"], help="Mode to run")
    p.add_argument("--debug", action="store_true", help="Enable debug logging")
    p.add_argument("--vision", action="store_true", help="Enable vision-based extractor (requires mss + opencv)")
    p.add_argument("--window-name", type=str, default="Fightcade", help="Window name fuzzy match for xdotool / vision")
    return p


def main(argv=None):
    args = build_arg_parser().parse_args(argv)
    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s: %(message)s")

    character = args.character.lower()
    opponent = args.opponent.lower()

    # Choose extractor
    extractor = None
    if args.vision:
        try:
            extractor = VisionExtractor(window_name_fuzzy=args.window_name)
            logging.info("Vision extractor initialized.")
        except Exception as e:
            logging.exception("Vision extractor failed to initialize: %s", e)
            extractor = DummyExtractor()
    else:
        extractor = DummyExtractor()

    # choose AI
    if args.mode == "simple":
        ai = Fightcade3rdStrikeAI(character=character)
    else:
        ai = GhostAI(character=character, opponent=opponent)

    # executor
    executor = LinuxInputExecutor(window_name_fuzzy=args.window_name)
    recorder = InputRecorder() if HAS_PYNPUT else None
    bot = FightingGameBot(extractor, ai, executor=executor, recorder=recorder, debug=args.debug)

    # Modes
    if args.mode == "simple":
        bot.start()
    elif args.mode == "record":
        bot.start_recording()
    elif args.mode == "play":
        # auto load latest ghost if exists
        if isinstance(ai, GhostAI):
            files = ai.list_ghosts()
            if files:
                latest = max(files, key=lambda f: f.stat().st_mtime)
                ai.load_patterns(latest.name)
                logging.info(f"Auto-loaded {latest.name}")
            else:
                logging.warning("No ghosts found; switching to simple AI.")
                ai = Fightcade3rdStrikeAI(character=character)
                bot.ai = ai
        bot.start()
    elif args.mode == "load":
        if isinstance(ai, GhostAI):
            files = ai.list_ghosts()
            if not files:
                logging.error("No ghost files found to load.")
                return
            for i, f in enumerate(files, 1):
                logging.info(f"{i}. {f.name}")
            sel = input("Select number: ").strip()
            try:
                idx = int(sel) - 1
                ai.load_patterns(files[idx].name)
                bot.start()
            except Exception as e:
                logging.error("Invalid selection: %s", e)
        else:
            logging.error("Load mode requires GhostAI.")
    elif args.mode == "view":
        if isinstance(ai, GhostAI):
            ai.list_ghosts()
        else:
            logging.info("Simple AI mode has no saved ghosts.")

if __name__ == "__main__":
    main()
