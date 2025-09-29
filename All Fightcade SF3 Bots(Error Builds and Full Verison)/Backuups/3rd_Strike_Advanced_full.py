
#!/usr/bin/env python3
"""
3rd_Strike_Advanced_full.py
Enhanced version with:
 - Fixes from previous patch (throttle, logging, normalized diagonals)
 - Phase 3: Interactive menu system (options 1-8 from roadmap)
 - Phase 4: Mirror Mode (P1 -> P2 real-time mirroring) with options:
     * exact timing copy
     * delayed copy (fixed delay)
     * random delay copy
     * occasional mistakes
 - Training type submenu and help system
 - Basic Mirror Mode UI printed to terminal
Note: For full real-time key capture and replay, `pynput` is required. If pynput is missing,
mirror/recording will fail gracefully and instruct the user.
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
import sys
import zipfile
import os

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

# Optional keyboard listener (for recording/mirroring)
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
        # default watch keys for P2 (WASD + y,u,i,h,j,k)
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
            # handle special keys like arrows
            if isinstance(key, kb.Key):
                name = str(key).split('.')[-1]
                return name.lower()
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
        self.action_cooldown = 0.05
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
    def __init__(self, dynamic=False):
        # if dynamic True, we'll vary distance slowly so simple AI has different behaviors
        self.t0 = time.time()
        self.dynamic = dynamic

    def extract_state(self) -> Optional[GameState]:
        # Basic dummy data so the bot can run without vision: centered positions.
        dist = 320
        if self.dynamic:
            # oscillate between 80 and 320 over 6 seconds
            t = time.time() - self.t0
            dist = 80 + abs(240 * (0.5 + 0.5 * math.sin(t * 2 * math.pi / 6)))
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
            distance=dist,
            frame_advantage=0,
            round_timer=99.0
        )


class VisionExtractor(GameStateExtractor):
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
        try:
            r = subprocess.run(["xdotool", "getwindowgeometry", "--shell", self.window_id], capture_output=True, text=True, check=True)
            lines = r.stdout.splitlines()
            geom = {}
            for line in lines:
                if "=" in line:
                    k, v = line.split("=", 1)
                    geom[k.strip()] = int(v.strip())
            return {"left": geom.get("X", 0), "top": geom.get("Y", 0), "width": geom.get("WIDTH", 640), "height": geom.get("HEIGHT", 480)}
        except Exception:
            return None

    def _grab(self):
        r = {"top": self.geom["top"], "left": self.geom["left"], "width": self.geom["width"], "height": self.geom["height"]}
        img = self.sct.grab(r)
        arr = np.array(img)
        if arr.shape[2] == 4:
            arr = arr[:, :, :3]
        return arr

    def _read_health_percentage(self, img, left_region=True) -> float:
        h, w, _ = img.shape
        y = int(h * 0.05)
        hbar_h = int(h * 0.03)
        if left_region:
            x1, x2 = int(w * 0.05), int(w * 0.35)
        else:
            x1, x2 = int(w * 0.65), int(w * 0.95)
        crop = img[y:y + hbar_h, x1:x2]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(float)
        filled = np.mean(v) / 255.0
        return max(0.0, min(1.0, filled))

    def extract_state(self) -> Optional[GameState]:
        try:
            img = self._grab()
            p1 = self._read_health_percentage(img, left_region=True)
            p2 = self._read_health_percentage(img, left_region=False)
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
        self.action_cooldown = 0.05
        self.combos = {
            'hadoken': ['down', 'down-right', 'right', 'hp'],
            'dp': ['right', 'down', 'down-right', 'hp']
        }

    def decide_action(self, game_state: GameState, history: List[GameState]) -> Optional[ActionCommand]:
        current_time = game_state.timestamp
        if current_time - self.last_action_time < self.action_cooldown:
            return None
        distance = game_state.distance
        # Long range: walk forward but throttle
        if distance > 250:
            self.last_action_time = current_time
            return ActionCommand(inputs=['right'], timing=current_time, duration=0.1, priority=1)
        # Midrange: hadoken attempt
        if 150 < distance <= 250:
            self.last_action_time = current_time
            return ActionCommand(inputs=self.combos['hadoken'], timing=current_time, duration=0.2, priority=5)
        # Close: dp/anti-air
        if distance <= 150:
            self.last_action_time = current_time
            return ActionCommand(inputs=self.combos['dp'], timing=current_time, duration=0.2, priority=10)
        return None


# ---------------------- Mirror Mode ----------------------
class MirrorMode:
    """
    Mirrors P1 inputs to P2.
    - Requires pynput to capture P1 keys.
    - Options:
      - mode: 'exact' -> immediate replay with same timing
      - mode: 'delay' -> fixed delay (seconds)
      - mode: 'random' -> random delay in [0, max_delay]
      - mode: 'mistake' -> occasionally drop or change an input with mistake_prob
    """
    def __init__(self, executor: 'LinuxInputExecutor', p1_watch_keys: Optional[set] = None, p2_mapping: Optional[Dict[str,str]] = None):
        self.executor = executor
        self.running = False
        self.p1_watch_keys = p1_watch_keys or {'up','down','left','right','kp_1'}  # default minimal
        self.listener = None
        self.buffer = []  # list of (timestamp, keys, duration)
        # default P2 mapping: map arrow keys -> wasd, and atk buttons to y/u/i/h/j/k if present
        self.p2_mapping = p2_mapping or {
            'up': 'w', 'down': 's', 'left': 'a', 'right': 'd',
            'lp': 'y', 'mp': 'u', 'hp': 'i', 'lk': 'h', 'mk': 'j', 'hk': 'k'
        }

    def _normalize_key(self, key):
        try:
            if hasattr(key, 'char') and key.char:
                return key.char.lower()
            if isinstance(key, kb.Key):
                return str(key).split('.')[-1].lower()
            return None
        except Exception:
            return None

    def _on_press(self, key):
        k = self._normalize_key(key)
        if not k:
            return
        ts = time.time()
        # store press time; will wait for release to know duration
        self.buffer.append({'key': k, 'press': ts, 'release': None})

    def _on_release(self, key):
        k = self._normalize_key(key)
        if not k:
            return
        ts = time.time()
        # find last unmatched press for this key
        for b in reversed(self.buffer):
            if b['key'] == k and b['release'] is None:
                b['release'] = ts
                # now schedule replay immediately or as per mode
                break

    def start(self, mode='exact', delay=0.05, max_random=0.2, mistake_prob=0.0):
        if not HAS_PYNPUT:
            logging.error("pynput not available: cannot start Mirror Mode.")
            return
        self.mode = mode
        self.delay = delay
        self.max_random = max_random
        self.mistake_prob = mistake_prob
        self.running = True
        self.listener = kb.Listener(on_press=self._on_press, on_release=self._on_release)
        self.listener.start()
        self._consumer_thread = threading.Thread(target=self._consume_buffer, daemon=True)
        self._consumer_thread.start()
        logging.info(f"Mirror mode started (mode={mode}, delay={delay}, max_random={max_random}, mistake_prob={mistake_prob})")

    def stop(self):
        self.running = False
        if self.listener:
            self.listener.stop()
        logging.info("Mirror mode stopped.")

    def _consume_buffer(self):
        # Continuously check buffer for completed press/release pairs and replay them
        while self.running:
            try:
                for b in list(self.buffer):
                    if b.get('release') is not None and not b.get('replayed', False):
                        press = b['press']
                        release = b['release']
                        duration = max(0.01, release - press)
                        key = b['key']
                        # decide replay timing
                        if self.mode == 'exact':
                            play_at = time.time()
                        elif self.mode == 'delay':
                            play_at = time.time() + self.delay
                        elif self.mode == 'random':
                            play_at = time.time() + random.random() * self.max_random
                        elif self.mode == 'mistake':
                            play_at = time.time() + self.delay
                        else:
                            play_at = time.time()
                        # schedule execution
                        threading.Thread(target=self._replay_key, args=(key, duration, play_at), daemon=True).start()
                        b['replayed'] = True
                time.sleep(0.005)
            except Exception:
                time.sleep(0.01)

    def _replay_key(self, key, duration, play_at):
        to_wait = play_at - time.time()
        if to_wait > 0:
            time.sleep(to_wait)
        # possibly introduce mistake
        if self.mode == 'mistake' and random.random() < self.mistake_prob:
            # either drop or map to a different key
            if random.random() < 0.5:
                logging.debug(f"Mirror mistake: dropping key {key}")
                return
            else:
                # pick random mapped key
                mapped_keys = list(self.p2_mapping.values())
                mapped = random.choice(mapped_keys)
                logging.debug(f"Mirror mistake: changing {key} -> {mapped}")
                self.executor._send_single_key(mapped, duration)
                return
        mapped = self.p2_mapping.get(key, None)
        if not mapped:
            # try simple normalization (arrow names like 'left' map to a/d etc.)
            mapped = self.p2_mapping.get(key.lower(), key.lower())
        self.executor._send_single_key(mapped, duration)


# ---------------------- Input executor (xdotool) ----------------------
class InputExecutor:
    def __init__(self):
        self.action_queue = queue.Queue()
        self.running = True

    def queue_action(self, action):
        try:
            if self.action_queue.qsize() > 200:
                logging.debug("Action queue full; dropping action")
                return
        except Exception:
            pass
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
            "coin": "6", "start": "2",
            # aliases
            "hp": "i", "mp": "u", "lp": "y",
            # arrow names fallback
            "left_arrow": "a", "right_arrow": "d", "up_arrow": "w", "down_arrow": "s"
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

    def _normalize_inputs(self, inputs: List[str]) -> List[str]:
        normalized = []
        for key in inputs:
            if key in ("down-right", "downright", "dr"):
                normalized.extend(["down", "right"])
            elif key in ("down-left", "downleft", "dl"):
                normalized.extend(["down", "left"])
            elif key in ("up-right", "upright", "ur"):
                normalized.extend(["up", "right"])
            elif key in ("up-left", "upleft", "ul"):
                normalized.extend(["up", "left"])
            else:
                normalized.append(key)
        return normalized

    def _execute_action(self, action: ActionCommand):
        if not self.window_id:
            logging.debug("No window id; skipping action execution")
            return
        norm_inputs = self._normalize_inputs(action.inputs)
        per_key_delay = action.duration / max(1, len(norm_inputs))
        for key in norm_inputs:
            mapped = self.key_mappings.get(key, key)
            try:
                logging.debug(f"Sending key for action: raw='{key}' mapped='{mapped}' to window {self.window_id}")
                subprocess.run(["xdotool", "key", "--window", self.window_id, mapped], check=True, capture_output=True)
                time.sleep(per_key_delay)
            except Exception as e:
                logging.exception("xdotool failed for key '%s' (mapped '%s'): %s", key, mapped, e)

    def _send_single_key(self, mapped_key: str, duration: float):
        # helper for MirrorMode to send a single mapped key by name (already mapped)
        if not self.window_id:
            logging.debug("No window id; _send_single_key skipped")
            return
        try:
            logging.debug(f"_send_single_key: sending '{mapped_key}' for {duration}s")
            subprocess.run(["xdotool", "key", "--window", self.window_id, mapped_key], check=True, capture_output=True)
            time.sleep(duration)
        except Exception as e:
            logging.exception("_send_single_key failed for '%s': %s", mapped_key, e)


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
                    try:
                        if self.executor.action_queue.qsize() < 150:
                            self.executor.queue_action(action)
                        else:
                            logging.debug("Executor queue busy; skipping action")
                    except Exception:
                        self.executor.queue_action(action)
            time.sleep(0.01)

    def stop(self):
        logging.info("Stopping bot...")
        self.running = False
        self.executor.stop()


# ---------------------- Interactive menu (Phase 3) ----------------------
def print_main_menu():
    print("\n === 3RD STRIKE BOT ===")
    print("1. Mirror Mode (copy P1 inputs to P2)")
    print("2. Simple AI (hardcoded behavior)")
    print("3. Record Training Session")
    print("4. Play with Ghost AI")
    print("5. Load Saved Ghost")
    print("6. View Saved Ghosts")
    print("7. Help")
    print("8. Exit")
    print()

def print_training_submenu():
    print("\n Training types:")
    print("1. Defensive training")
    print("2. Offensive training")
    print("3. Neutral/footsies training")
    print("4. Combo practice")
    print("5. Anti-air practice")
    print("6. Full match recording")
    print("7. Back")

def print_help():
    print("""
Help - Quick Reference
- Mirror Mode: copies your P1 inputs to P2. Requires pynput and xdotool.
- Simple AI: built-in heuristic AI (distance-based).
- Record: records inputs played on P2 controls (WASD + YUIHJK).
- Play with Ghost AI: uses saved ghosts to play. Save ghosts after recording.
Controls mapping:
- P1 (arrows + buttons) -> P2 (WASD + YUIHJK)
Troubleshooting:
- If keys aren't sending, ensure xdotool is installed and the window name matches (default 'Fightcade').
- Use --window-name to change the fuzzy match.
""")

def interactive_menu(argv=None):
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--window-name", type=str, default="Fightcade")
    parser.add_argument("--debug", action="store_true")
    known, _ = parser.parse_known_args(argv)
    log_level = logging.DEBUG if known.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s: %(message)s")

    window_name = known.window_name
    # default extractor
    extractor = DummyExtractor()
    executor = LinuxInputExecutor(window_name_fuzzy=window_name)
    recorder = InputRecorder() if HAS_PYNPUT else None

    while True:
        print_main_menu()
        sel = input("Select option (1-8): ").strip()
        if sel == '1':
            # Mirror Mode
            if not HAS_PYNPUT:
                print("Mirror Mode requires pynput (keyboard capture). Install pynput and retry.")
                continue
            print("\nMirror Mode options:")
            print("1. Exact timing copy")
            print("2. Fixed delay copy")
            print("3. Random delay copy")
            print("4. Mirror with occasional mistakes")
            print("5. Back")
            msel = input("Select: ").strip()
            mirror = MirrorMode(executor)
            if msel == '1':
                mirror.start(mode='exact')
            elif msel == '2':
                d = input("Delay in seconds (e.g. 0.05): ").strip()
                try:
                    dd = float(d)
                except:
                    dd = 0.05
                mirror.start(mode='delay', delay=dd)
            elif msel == '3':
                m = input("Max random delay in seconds (e.g. 0.15): ").strip()
                try:
                    mm = float(m)
                except:
                    mm = 0.15
                mirror.start(mode='random', max_random=mm)
            elif msel == '4':
                p = input("Mistake probability (0.0-1.0, e.g. 0.05): ").strip()
                try:
                    pp = float(p)
                except:
                    pp = 0.05
                mirror.start(mode='mistake', delay=0.05, mistake_prob=pp)
            elif msel == '5':
                continue
            print("Mirror running. Press Ctrl+C to stop.")
            try:
                while True:
                    time.sleep(0.5)
            except KeyboardInterrupt:
                mirror.stop()
                continue

        elif sel == '2':
            # Simple AI play
            ai = Fightcade3rdStrikeAI()
            bot = FightingGameBot(extractor, ai, executor=executor, recorder=recorder, debug=known.debug)
            print("Starting Simple AI. Ctrl+C to stop.")
            try:
                bot.start()
            except KeyboardInterrupt:
                bot.stop()
                continue

        elif sel == '3':
            # Record training session
            if not HAS_PYNPUT:
                print("Recording requires pynput. Install it to record local inputs.")
                continue
            ai = GhostAI()
            bot = FightingGameBot(extractor, ai, executor=executor, recorder=recorder, debug=known.debug)
            bot.start_recording()
            continue

        elif sel == '4':
            # Play with Ghost AI
            ai = GhostAI()
            # auto-load latest if exists
            files = ai.list_ghosts()
            if files:
                latest = max(files, key=lambda f: f.stat().st_mtime)
                ai.load_patterns(latest.name)
                print(f"Auto-loaded {latest.name}")
            else:
                print("No ghosts found, switching to Simple AI.")
                ai = Fightcade3rdStrikeAI()
            bot = FightingGameBot(extractor, ai, executor=executor, recorder=recorder, debug=known.debug)
            try:
                bot.start()
            except KeyboardInterrupt:
                bot.stop()
                continue

        elif sel == '5':
            # Load saved ghost
            ai = GhostAI()
            files = ai.list_ghosts()
            if not files:
                continue
            for i, f in enumerate(files, 1):
                print(f"{i}. {f.name}")
            sel2 = input("Select number: ").strip()
            try:
                idx = int(sel2) - 1
                ai.load_patterns(files[idx].name)
                bot = FightingGameBot(extractor, ai, executor=executor, recorder=recorder, debug=known.debug)
                bot.start()
            except Exception as e:
                logging.error("Invalid selection: %s", e)
                continue

        elif sel == '6':
            ai = GhostAI()
            ai.list_ghosts()
            input("Press Enter to continue...")

        elif sel == '7':
            print_help()
            input("Press Enter to continue...")

        elif sel == '8':
            print("Exiting.")
            return

        else:
            print("Invalid selection.")

# ---------------------- CLI Entrypoint & ZIP helper ----------------------
def create_zip_of_project(output_zip="3rd_strike_bot_package.zip"):
    files_to_include = ["3rd_Strike_Advanced_full.py"]
    with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files_to_include:
            if os.path.exists(f):
                z.write(f)
    return output_zip

def main(argv=None):
    # write the file (self) to disk so we can zip it for the user
    this_file = Path("3rd_Strike_Advanced_full.py")
    if not this_file.exists():
        # attempt to write source from __file__ not reliable; instead assume content already saved
        pass

    # if user passed --menu run interactive
    if len(sys.argv) > 1 and sys.argv[1] == "--menu":
        interactive_menu(sys.argv[2:])
        return

    parser = argparse.ArgumentParser(description="3rd Strike Advanced Bot - Full (menu + mirror)")
    parser.add_argument("--menu", action="store_true", help="Run interactive menu")
    parser.add_argument("--export-zip", action="store_true", help="Export project as zip and exit")
    parser.add_argument("--debug", action="store_true", help="Enable debug logging")
    args = parser.parse_args(argv)

    log_level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s: %(message)s")

    if args.export_zip:
        z = create_zip_of_project()
        print(f"Created {z}")
        return

    # default behavior: show menu
    interactive_menu()

if __name__ == "__main__":
    main()
