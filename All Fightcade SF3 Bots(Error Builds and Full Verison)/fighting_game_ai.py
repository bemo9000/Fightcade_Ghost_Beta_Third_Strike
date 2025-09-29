import cv2
import numpy as np
import time
import threading
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
import queue
import json
from abc import ABC, abstractmethod

# For input injection (you'll need to install these)
try:
    import keyboard
    import mouse
    HAS_INPUT_LIBS = True
except ImportError:
    HAS_INPUT_LIBS = False
    print("Install keyboard and mouse libs: pip install keyboard mouse")

@dataclass
class GameState:
    """Represents the current state of the fighting game"""
    timestamp: float
    player1_health: float
    player2_health: float
    player1_super: float
    player2_super: float
    player1_position: Tuple[int, int]
    player2_position: Tuple[int, int]
    player1_state: str  # "idle", "attacking", "blocking", "hitstun", etc.
    player2_state: str
    distance: float
    frame_advantage: int  # positive = player advantage
    round_timer: float
    
    def to_dict(self) -> Dict:
        return {
            'timestamp': self.timestamp,
            'p1_health': self.player1_health,
            'p2_health': self.player2_health,
            'p1_super': self.player1_super,
            'p2_super': self.player2_super,
            'p1_pos': self.player1_position,
            'p2_pos': self.player2_position,
            'p1_state': self.player1_state,
            'p2_state': self.player2_state,
            'distance': self.distance,
            'frame_advantage': self.frame_advantage,
            'timer': self.round_timer
        }

@dataclass
class ActionCommand:
    """Represents an input command to execute"""
    inputs: List[str]  # e.g., ["down", "down-forward", "forward", "punch"]
    timing: float  # when to execute (timestamp)
    duration: float  # how long to hold inputs
    priority: int  # higher = more important

class GameStateExtractor(ABC):
    """Abstract base class for extracting game state from different sources"""
    
    @abstractmethod
    def extract_state(self) -> Optional[GameState]:
        pass

class ScreenCaptureExtractor(GameStateExtractor):
    """Extracts game state from screen capture using computer vision"""
    
    def __init__(self, capture_region: Tuple[int, int, int, int] = None):
        self.capture_region = capture_region  # (x, y, width, height)
        self.health_bars_template = None
        self.character_templates = {}
        
    def extract_state(self) -> Optional[GameState]:
        # Capture screen
        screenshot = self._capture_screen()
        if screenshot is None:
            return None
            
        # Extract health bars
        p1_health, p2_health = self._extract_health_bars(screenshot)
        
        # Extract character positions
        p1_pos, p2_pos = self._extract_character_positions(screenshot)
        
        # Extract super meter
        p1_super, p2_super = self._extract_super_meters(screenshot)
        
        # Calculate derived values
        distance = abs(p1_pos[0] - p2_pos[0]) if p1_pos and p2_pos else 0
        
        return GameState(
            timestamp=time.time(),
            player1_health=p1_health,
            player2_health=p2_health,
            player1_super=p1_super,
            player2_super=p2_super,
            player1_position=p1_pos or (0, 0),
            player2_position=p2_pos or (0, 0),
            player1_state="unknown",  # Would need more sophisticated detection
            player2_state="unknown",
            distance=distance,
            frame_advantage=0,  # Would need frame data analysis
            round_timer=99.0  # Would need OCR
        )
    
    def _capture_screen(self) -> Optional[np.ndarray]:
        """Capture screen or game window"""
        try:
            # This is a placeholder - you'd implement actual screen capture
            # Could use PIL, mss, or game-specific capture methods
            import mss
            with mss.mss() as sct:
                if self.capture_region:
                    monitor = {
                        "top": self.capture_region[1],
                        "left": self.capture_region[0],
                        "width": self.capture_region[2],
                        "height": self.capture_region[3]
                    }
                else:
                    monitor = sct.monitors[1]  # Primary monitor
                
                screenshot = sct.grab(monitor)
                return np.array(screenshot)
        except ImportError:
            print("Install mss for screen capture: pip install mss")
            return None
    
    def _extract_health_bars(self, screenshot: np.ndarray) -> Tuple[float, float]:
        """Extract health bar percentages using template matching"""
        # Placeholder implementation
        # In reality, you'd use template matching or color detection
        return 1.0, 1.0  # Full health for both players
    
    def _extract_character_positions(self, screenshot: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Extract character positions from screenshot"""
        # Placeholder - would use template matching or contour detection
        return (200, 300), (600, 300)
    
    def _extract_super_meters(self, screenshot: np.ndarray) -> Tuple[float, float]:
        """Extract super meter levels"""
        return 0.0, 0.0

class MemoryReadExtractor(GameStateExtractor):
    """Extracts game state directly from game memory (requires memory hacking)"""
    
    def __init__(self, process_name: str = "sf3.exe"):
        self.process_name = process_name
        self.memory_addresses = {
            'p1_health': 0x1234567,  # These would be actual memory addresses
            'p2_health': 0x1234568,
            'p1_x': 0x1234569,
            'p1_y': 0x123456A,
            # ... etc
        }
    
    def extract_state(self) -> Optional[GameState]:
        # This would require a memory reading library like pymem
        # Placeholder implementation
        return None

class FightingGameAI(ABC):
    """Abstract base class for fighting game AI decision making"""
    
    @abstractmethod
    def decide_action(self, game_state: GameState, history: List[GameState]) -> Optional[ActionCommand]:
        pass

class RuleBased3rdStrikeAI(FightingGameAI):
    """Rule-based AI for Street Fighter 3rd Strike"""
    
    def __init__(self):
        self.combos = {
            'basic_punish': ["down", "down-forward", "forward", "punch"],
            'anti_air': ["forward", "down", "down-forward", "punch"],
            'super_art': ["down", "down-forward", "forward", "down", "down-forward", "forward", "punch"]
        }
        
        self.last_action_time = 0
        self.action_cooldown = 0.1  # Minimum time between actions
    
    def decide_action(self, game_state: GameState, history: List[GameState]) -> Optional[ActionCommand]:
        current_time = game_state.timestamp
        
        # Cooldown check
        if current_time - self.last_action_time < self.action_cooldown:
            return None
        
        # Simple decision tree
        distance = game_state.distance
        
        # If opponent is close and we have super meter
        if distance < 100 and game_state.player1_super > 0.8:
            return ActionCommand(
                inputs=self.combos['super_art'],
                timing=current_time + 0.05,
                duration=0.1,
                priority=10
            )
        
        # Anti-air if opponent might be jumping (simplified)
        if game_state.player2_position[1] < 250:  # Assuming lower Y = higher position
            return ActionCommand(
                inputs=self.combos['anti_air'],
                timing=current_time + 0.02,
                duration=0.08,
                priority=8
            )
        
        # Basic attack if in range
        if 50 < distance < 150:
            return ActionCommand(
                inputs=["forward", "punch"],
                timing=current_time + 0.01,
                duration=0.05,
                priority=5
            )
        
        # Move forward if too far
        if distance > 200:
            return ActionCommand(
                inputs=["forward"],
                timing=current_time,
                duration=0.1,
                priority=2
            )
        
        return None

class LinuxInputExecutor(InputExecutor):
    """Linux-optimized input executor using multiple methods"""
    
    def __init__(self):
        super().__init__()
        self.input_method = self._detect_input_method()
        self.device_path = self._find_input_device()
        
        # MX Linux / Debian key mappings for Fightcade
        self.key_mappings = {
            'left': 'a',
            'right': 'd', 
            'up': 'w',
            'down': 's',
            'lp': 'u',     # Light Punch
            'mp': 'i',     # Medium Punch  
            'hp': 'o',     # Heavy Punch
            'lk': 'j',     # Light Kick
            'mk': 'k',     # Medium Kick
            'hk': 'l',     # Heavy Kick
            'down-right': ['s', 'd'],
            'down-left': ['s', 'a'],
            'up-right': ['w', 'd'],
            'up-left': ['w', 'a']
        }
    
    def _detect_input_method(self) -> str:
        """Detect best input method for current Linux setup"""
        try:
            import subprocess
            
            # Check if we're in X11 or Wayland
            session_type = subprocess.run(['echo', '$XDG_SESSION_TYPE'], 
                                        capture_output=True, text=True, shell=True)
            
            # Check available tools
            tools = []
            for tool in ['xdotool', 'ydotool', 'evdev']:
                try:
                    subprocess.run(['which', tool], capture_output=True, check=True)
                    tools.append(tool)
                except subprocess.CalledProcessError:
                    pass
            
            # Prefer xdotool for X11, ydotool for Wayland
            if 'xdotool' in tools:
                return 'xdotool'
            elif 'ydotool' in tools:
                return 'ydotool'
            elif 'evdev' in tools:
                return 'evdev'
            else:
                return 'keyboard'  # Fallback to Python keyboard library
                
        except Exception:
            return 'keyboard'
    
    def _find_input_device(self) -> Optional[str]:
        """Find keyboard input device for direct access"""
        try:
            import glob
            
            # Look for keyboard devices
            devices = glob.glob('/dev/input/event*')
            for device in devices:
                try:
                    with open(device, 'rb') as f:
                        # This is a simplified check - you'd want more robust detection
                        return device
                except PermissionError:
                    continue
                    
        except Exception:
            pass
        
        return None
    
    def _execute_action(self, action: ActionCommand):
        """Execute action using the best available method"""
        if self.input_method == 'xdotool':
            self._execute_with_xdotool(action)
        elif self.input_method == 'ydotool':
            self._execute_with_ydotool(action)
        elif self.input_method == 'evdev':
            self._execute_with_evdev(action)
        else:
            self._execute_with_keyboard(action)
    
    def _execute_with_xdotool(self, action: ActionCommand):
        """Execute using xdotool (X11)"""
        import subprocess
        
        # Get Fightcade window ID
        try:
            result = subprocess.run(['xdotool', 'search', '--name', 'FightCade'], 
                                  capture_output=True, text=True)
            if result.returncode != 0:
                return
            
            window_id = result.stdout.strip().split('\n')[0]
            
            for input_name in action.inputs:
                keys = self.key_mappings.get(input_name, input_name)
                
                if isinstance(keys, list):
                    # Simultaneous inputs
                    for key in keys:
                        subprocess.run(['xdotool', 'windowfocus', window_id, 
                                      'key', '--delay', '16', key])
                else:
                    subprocess.run(['xdotool', 'windowfocus', window_id,
                                  'key', '--delay', '16', keys])
                    
        except Exception as e:
            print(f"xdotool execution failed: {e}")
    
    def _execute_with_ydotool(self, action: ActionCommand):
        """Execute using ydotool (Wayland)"""
        import subprocess
        
        for input_name in action.inputs:
            keys = self.key_mappings.get(input_name, input_name)
            
            if isinstance(keys, list):
                for key in keys:
                    subprocess.run(['ydotool', 'key', f'{key}:1'])
                    time.sleep(0.016)  # 1 frame
                    subprocess.run(['ydotool', 'key', f'{key}:0'])
            else:
                subprocess.run(['ydotool', 'key', f'{keys}:1'])
                time.sleep(action.duration / len(action.inputs))
                subprocess.run(['ydotool', 'key', f'{keys}:0'])
    
    def _execute_with_evdev(self, action: ActionCommand):
        """Direct input device access using evdev"""
        try:
            from evdev import UInput, ecodes as e
            
            # This would need proper setup and permissions
            # Placeholder implementation
            pass
            
        except ImportError:
            print("evdev not available, falling back to keyboard library")
            self._execute_with_keyboard(action)
    
    def _execute_with_keyboard(self, action: ActionCommand):
        """Fallback to keyboard library"""
        if not HAS_INPUT_LIBS:
            return
            
        for input_name in action.inputs:
            keys = self.key_mappings.get(input_name, input_name)
            
            if isinstance(keys, list):
                for key in keys:
                    keyboard.press(key)
                time.sleep(action.duration / len(action.inputs))
                for key in reversed(keys):
                    keyboard.release(key)
            else:
                keyboard.press(keys)
                time.sleep(action.duration / len(action.inputs))
                keyboard.release(keys)

# Updated main class to use Linux-optimized components
class LinuxFightingGameBot(FightingGameBot):
    """MX Linux optimized fighting game bot"""
    
    def __init__(self, extractor: GameStateExtractor, ai: FightingGameAI):
        super().__init__(extractor, ai)
        self.executor = LinuxInputExecutor()  # Use Linux-optimized executor
    
    def setup_permissions(self):
        """Check and setup required permissions for Linux"""
        print("Checking Linux permissions and dependencies...")
        
        # Check if user is in input group (for /dev/input access)
        import subprocess
        import os
        
        try:
            groups = subprocess.run(['groups'], capture_output=True, text=True)
            if 'input' not in groups.stdout:
                print("⚠️  Consider adding your user to 'input' group for better input access:")
                print("   sudo usermod -a -G input $USER")
                print("   (requires logout/login)")
        except:
            pass
        
        # Check for required tools
        tools_status = {}
        for tool in ['xdotool', 'import', 'xwininfo']:
            try:
                subprocess.run(['which', tool], capture_output=True, check=True)
                tools_status[tool] = "✓ Found"
            except subprocess.CalledProcessError:
                tools_status[tool] = "✗ Missing"
        
        print("\nTool availability:")
        for tool, status in tools_status.items():
            print(f"  {tool}: {status}")
        
        if tools_status.get('xdotool') == "✗ Missing":
            print("\n📦 Install xdotool for better input handling:")
            print("   sudo apt install xdotool")
        
        if tools_status.get('import') == "✗ Missing":
            print("\n📦 Install ImageMagick for fast window capture:")
            print("   sudo apt install imagemagick")

# Updated example usage for MX Linux
if __name__ == "__main__":
    print("🐧 Fightcade SF3S AI Bot for MX Linux")
    print("=" * 50)
    
    # Setup and check permissions
    bot_instance = LinuxFightingGameBot(None, None)  # Temporary for setup
    bot_instance.setup_permissions()
    
    print("\nMake sure:")
    print("1. Fightcade is running with 3rd Strike loaded")
    print("2. Fightcade window is visible and focused")
    print("3. You're not running as root (for security)")
    
    # Use Fightcade-specific extractor
    extractor = FightcadeExtractor()
    
    # Use 3rd Strike AI
    character = input("\nChoose character (ken/ryu/chun): ").lower() or "ken"
    ai = Fightcade3rdStrikeAI(character=character)
    
    # Create Linux-optimized bot
    bot = LinuxFightingGameBot(extractor, ai)
    
    print(f"\n🥋 Starting {character.upper()} AI in 3 seconds...")
    print("Press Ctrl+C to stop")
    time.sleep(3)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()
        print("\n🛑 Bot stopped by user")

class FightingGameBot:
    """Main bot class that orchestrates everything"""
    
    def __init__(self, extractor: GameStateExtractor, ai: FightingGameAI):
        self.extractor = extractor
        self.ai = ai
        self.executor = InputExecutor()
        
        self.running = False
        self.game_state_history = []
        self.max_history = 100
        
        # Performance metrics
        self.fps = 0
        self.last_fps_time = time.time()
        self.frame_count = 0
    
    def start(self):
        """Start the bot"""
        print("Starting Fighting Game AI Bot...")
        self.running = True
        
        # Start input executor in separate thread
        executor_thread = threading.Thread(target=self.executor.execute_inputs)
        executor_thread.daemon = True
        executor_thread.start()
        
        # Main game loop
        self.run_game_loop()
    
    def run_game_loop(self):
        """Main game loop"""
        while self.running:
            try:
                # Extract current game state
                game_state = self.extractor.extract_state()
                if game_state is None:
                    time.sleep(0.001)  # Small delay if no state extracted
                    continue
                
                # Update history
                self.game_state_history.append(game_state)
                if len(self.game_state_history) > self.max_history:
                    self.game_state_history.pop(0)
                
                # AI decision making
                action = self.ai.decide_action(game_state, self.game_state_history)
                if action:
                    self.executor.queue_action(action)
                
                # Update performance metrics
                self._update_fps()
                
                # Small delay to prevent overwhelming the system
                time.sleep(0.001)
                
            except KeyboardInterrupt:
                break
            except Exception as e:
                print(f"Error in game loop: {e}")
                time.sleep(0.1)
        
        self.stop()
    
    def _update_fps(self):
        """Update FPS counter"""
        self.frame_count += 1
        current_time = time.time()
        
        if current_time - self.last_fps_time >= 1.0:
            self.fps = self.frame_count / (current_time - self.last_fps_time)
            print(f"Bot FPS: {self.fps:.1f}")
            self.frame_count = 0
            self.last_fps_time = current_time
    
    def stop(self):
        """Stop the bot"""
        print("Stopping Fighting Game AI Bot...")
        self.running = False
        self.executor.stop()

class FightcadeExtractor(GameStateExtractor):
    """Specialized extractor for Fightcade Street Fighter 3rd Strike"""
    
    def __init__(self):
        # Fightcade window dimensions (typically 640x480 or 1280x960)
        self.window_title = "FightCade FBNeo v0.2.97.44.55 Street Fighter III 3rd Strike: Fight for the Future(Japan 990512, NO CD) "
        
        # Pixel coordinates for UI elements in Fightcade SF3S
        # These would need to be calibrated for your specific setup
        self.ui_regions = {
            'p1_health_bar': (47, 20, 200, 10),   # (x, y, width, height)
            'p2_health_bar': (393, 20, 200, 10),
            'p1_super_bar': (47, 40, 150, 8),
            'p2_super_bar': (443, 40, 150, 8),
            'timer': (310, 20, 30, 20),
            'game_area': (0, 56, 640, 424)  # Main gameplay area
        }
        
        # Color thresholds for health/super detection
        self.health_color_ranges = {
            'full_health': ([0, 200, 0], [50, 255, 50]),      # Green range
            'mid_health': ([0, 200, 200], [50, 255, 255]),    # Yellow range  
            'low_health': ([0, 0, 200], [50, 50, 255])        # Red range
        }
        
        # Character detection templates (you'd create these)
        self.character_sprites = {}
        
    def extract_state(self) -> Optional[GameState]:
        # Find Fightcade window
        screenshot = self._capture_fightcade_window()
        if screenshot is None:
            return None
        
        # Extract UI elements
        p1_health = self._extract_health_bar(screenshot, 'p1')
        p2_health = self._extract_health_bar(screenshot, 'p2')
        p1_super = self._extract_super_bar(screenshot, 'p1')
        p2_super = self._extract_super_bar(screenshot, 'p2')
        timer = self._extract_timer(screenshot)
        
        # Extract character positions from gameplay area
        p1_pos, p2_pos = self._extract_character_positions_fightcade(screenshot)
        
        # Calculate distance
        distance = abs(p1_pos[0] - p2_pos[0]) if p1_pos and p2_pos else 320
        
        return GameState(
            timestamp=time.time(),
            player1_health=p1_health,
            player2_health=p2_health,
            player1_super=p1_super,
            player2_super=p2_super,
            player1_position=p1_pos or (160, 400),
            player2_position=p2_pos or (480, 400),
            player1_state="unknown",
            player2_state="unknown", 
            distance=distance,
            frame_advantage=0,
            round_timer=timer
        )
    
    def _capture_fightcade_window(self) -> Optional[np.ndarray]:
        """Capture specifically the Fightcade window - Linux optimized"""
        try:
            # Try multiple methods for Linux window capture
            return self._capture_with_xwininfo() or self._capture_with_pyautogui() or self._capture_with_mss()
            
        except Exception as e:
            print(f"Window capture failed: {e}")
            return None
    
    def _capture_with_xwininfo(self) -> Optional[np.ndarray]:
        """Use xwininfo and import for fast Linux window capture"""
        try:
            import subprocess
            
            # Find Fightcade window using xwininfo
            result = subprocess.run(['xwininfo', '-name', 'FightCade'], 
                                  capture_output=True, text=True, timeout=1)
            
            if result.returncode != 0:
                # Try alternative window names
                for name in ['fightcade', 'FightCade2', 'GGPO']:
                    result = subprocess.run(['xwininfo', '-name', name], 
                                          capture_output=True, text=True, timeout=1)
                    if result.returncode == 0:
                        break
                else:
                    return None
            
            # Parse window info
            lines = result.stdout.split('\n')
            window_id = None
            x, y, width, height = 0, 0, 640, 480
            
            for line in lines:
                if 'Window id:' in line:
                    window_id = line.split()[3]
                elif 'Absolute upper-left X:' in line:
                    x = int(line.split()[-1])
                elif 'Absolute upper-left Y:' in line:
                    y = int(line.split()[-1])
                elif 'Width:' in line:
                    width = int(line.split()[-1])
                elif 'Height:' in line:
                    height = int(line.split()[-1])
            
            if not window_id:
                return None
            
            # Use import command for fast screenshot
            import_result = subprocess.run([
                'import', '-window', window_id, '-crop', f'{width}x{height}+0+0', 
                'png:-'
            ], capture_output=True, timeout=2)
            
            if import_result.returncode == 0:
                # Convert to numpy array
                import io
                from PIL import Image
                img = Image.open(io.BytesIO(import_result.stdout))
                return np.array(img)
                
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, ImportError):
            pass
        
        return None
    
    def _capture_with_pyautogui(self) -> Optional[np.ndarray]:
        """Fallback to pyautogui (cross-platform but slower)"""
        try:
            import pyautogui
            
            # On Linux, pyautogui might need display setup
            import os
            if 'DISPLAY' not in os.environ:
                os.environ['DISPLAY'] = ':0'
            
            # Simple fullscreen capture first, then we'll crop
            screenshot = pyautogui.screenshot()
            return np.array(screenshot)
            
        except ImportError:
            return None
    
    def _capture_with_mss(self) -> Optional[np.ndarray]:
        """Use python-mss for fast Linux screen capture"""
        try:
            import mss
            
            with mss.mss() as sct:
                # Get primary monitor
                monitor = sct.monitors[1]
                screenshot = sct.grab(monitor)
                return np.array(screenshot)
                
        except ImportError:
            return None
    
    def _extract_health_bar(self, screenshot: np.ndarray, player: str) -> float:
        """Extract health bar percentage for Fightcade"""
        region = self.ui_regions[f'{player}_health_bar']
        health_bar = screenshot[region[1]:region[1]+region[3], 
                              region[0]:region[0]+region[2]]
        
        # Convert to HSV for better color detection
        hsv = cv2.cvtColor(health_bar, cv2.COLOR_RGB2HSV)
        
        # Count pixels in health color ranges
        total_pixels = health_bar.shape[0] * health_bar.shape[1]
        health_pixels = 0
        
        for color_name, (lower, upper) in self.health_color_ranges.items():
            mask = cv2.inRange(hsv, np.array(lower), np.array(upper))
            health_pixels += np.sum(mask > 0)
        
        return min(1.0, health_pixels / total_pixels)
    
    def _extract_super_bar(self, screenshot: np.ndarray, player: str) -> float:
        """Extract super bar percentage"""
        region = self.ui_regions[f'{player}_super_bar']
        super_bar = screenshot[region[1]:region[1]+region[3], 
                             region[0]:region[0]+region[2]]
        
        # Super bars are typically blue/yellow
        hsv = cv2.cvtColor(super_bar, cv2.COLOR_RGB2HSV)
        
        # Define super meter colors (blue/yellow typically)
        super_lower = np.array([100, 100, 100])
        super_upper = np.array([130, 255, 255])
        mask = cv2.inRange(hsv, super_lower, super_upper)
        
        total_pixels = super_bar.shape[0] * super_bar.shape[1]
        super_pixels = np.sum(mask > 0)
        
        return min(1.0, super_pixels / total_pixels)
    
    def _extract_timer(self, screenshot: np.ndarray) -> float:
        """Extract round timer using OCR"""
        region = self.ui_regions['timer']
        timer_area = screenshot[region[1]:region[1]+region[3],
                              region[0]:region[0]+region[2]]
        
        # This would need OCR implementation
        # For now, return a placeholder
        return 99.0
    
    def _extract_character_positions_fightcade(self, screenshot: np.ndarray) -> Tuple[Tuple[int, int], Tuple[int, int]]:
        """Extract character positions from Fightcade gameplay area"""
        game_region = self.ui_regions['game_area']
        game_area = screenshot[game_region[1]:game_region[1]+game_region[3],
                             game_region[0]:game_region[0]+game_region[2]]
        
        # Simple approach: find largest contours (characters)
        gray = cv2.cvtColor(game_area, cv2.COLOR_RGB2GRAY)
        
        # Apply threshold to isolate characters from background
        _, thresh = cv2.threshold(gray, 50, 255, cv2.THRESH_BINARY)
        
        # Find contours
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Sort by area and take two largest (should be the characters)
        contours = sorted(contours, key=cv2.contourArea, reverse=True)[:2]
        
        positions = []
        for contour in contours:
            if cv2.contourArea(contour) > 500:  # Filter small noise
                x, y, w, h = cv2.boundingRect(contour)
                center_x = x + w // 2
                center_y = y + h // 2
                positions.append((center_x, center_y))
        
        # Ensure we have two positions, sort by x-coordinate (left player first)
        if len(positions) >= 2:
            positions = sorted(positions, key=lambda p: p[0])
            return positions[0], positions[1]
        elif len(positions) == 1:
            # Assume other player is on opposite side
            if positions[0][0] < 320:  # Left side
                return positions[0], (480, 400)
            else:  # Right side
                return (160, 400), positions[0]
        else:
            # Default positions
            return (160, 400), (480, 400)

class Fightcade3rdStrikeAI(FightingGameAI):
    """Specialized AI for 3rd Strike on Fightcade with actual frame data"""
    
    def __init__(self, character: str = "ken"):
        self.character = character.lower()
        self.last_action_time = 0
        self.action_cooldown = 0.016  # ~1 frame at 60fps
        
        # 3rd Strike specific data
        self.frame_data = {
            'ken': {
                'cr_mk': {'startup': 5, 'active': 3, 'recovery': 9, 'advantage': -1},
                'dp': {'startup': 3, 'active': 2, 'recovery': 29, 'advantage': -25},
                'sa3': {'startup': 1, 'active': 45, 'recovery': 4, 'advantage': 10}
            },
            'ryu': {
                'cr_mk': {'startup': 5, 'active': 3, 'recovery': 9, 'advantage': -1},
                'dp': {'startup': 3, 'active': 2, 'recovery': 29, 'advantage': -25},
                'sa1': {'startup': 3, 'active': 23, 'recovery': 4, 'advantage': 10}
            }
        }
        
        # Fightcade input mappings (standard MAME controls)
        self.inputs = {
            'left': 'a',
            'right': 'd', 
            'up': 'w',
            'down': 's',
            'lp': 'u',    # Light Punch
            'mp': 'i',    # Medium Punch  
            'hp': 'o',    # Heavy Punch
            'lk': 'j',    # Light Kick
            'mk': 'k',    # Medium Kick
            'hk': 'l'     # Heavy Kick
        }
        
        # Common combos and inputs
        self.combos = {
            'ken_sa3': ['down', 'down-right', 'right', 'down', 'down-right', 'right', 'hp'],
            'hadoken': ['down', 'down-right', 'right', 'hp'],
            'dp': ['right', 'down', 'down-right', 'hp'],
            'super_jump': ['down', 'up']
        }
    
    def decide_action(self, game_state: GameState, history: List[GameState]) -> Optional[ActionCommand]:
        current_time = game_state.timestamp
        
        # Frame-based cooldown
        if current_time - self.last_action_time < self.action_cooldown:
            return None
        
        distance = game_state.distance
        p1_health = game_state.player1_health
        p2_health = game_state.player2_health
        super_meter = game_state.player1_super
        
        # Critical health - play defensively
        if p1_health < 0.2:
            return self._defensive_action(game_state)
        
        # Super Art available
        if super_meter >= 1.0 and distance < 150:
            return ActionCommand(
                inputs=self.combos[f'{self.character}_sa3'],
                timing=current_time + 0.016,  # 1 frame delay
                duration=0.5,
                priority=15
            )
        
        # Anti-air (opponent jumping)
        if len(history) >= 2:
            prev_state = history[-2]
            if (game_state.player2_position[1] < prev_state.player2_position[1] and 
                distance < 200):
                return ActionCommand(
                    inputs=self.combos['dp'],
                    timing=current_time + 0.05,
                    duration=0.2,
                    priority=12
                )
        
        # Footsies range - cr.MK
        if 120 < distance < 180:
            return ActionCommand(
                inputs=['down', 'mk'],
                timing=current_time + 0.016,
                duration=0.1,
                priority=8
            )
        
        # Close range - throw or attack
        if distance < 50:
            # Simple throw (LP+LK)
            return ActionCommand(
                inputs=['lp+lk'],  # Would need to implement simultaneous inputs
                timing=current_time + 0.016,
                duration=0.05,
                priority=10
            )
        
        # Too far - walk forward
        if distance > 250:
            return ActionCommand(
                inputs=['right'],
                timing=current_time,
                duration=0.1,
                priority=3
            )
        
        # Mid-range hadoken
        if 150 < distance < 300 and super_meter < 0.5:
            return ActionCommand(
                inputs=self.combos['hadoken'],
                timing=current_time + 0.1,
                duration=0.3,
                priority=6
            )
        
        self.last_action_time = current_time
        return None
    
    def _defensive_action(self, game_state: GameState) -> ActionCommand:
        """Defensive actions when health is low"""
        distance = game_state.distance
        
        if distance < 100:
            # Block or backdash
            return ActionCommand(
                inputs=['left'],  # Block
                timing=game_state.timestamp,
                duration=0.2,
                priority=7
            )
        else:
            # Keep distance
            return ActionCommand(
                inputs=['left'],  # Walk back
                timing=game_state.timestamp,
                duration=0.15,
                priority=5
            )

# Updated example usage for Fightcade
if __name__ == "__main__":
    print("Fightcade SF3S AI Bot")
    print("Make sure Fightcade is running with 3rd Strike loaded!")
    
    # Use Fightcade-specific extractor
    extractor = FightcadeExtractor()
    
    # Use 3rd Strike AI (specify your character)
    ai = Fightcade3rdStrikeAI(character="ken")  # or "ryu", "chun", etc.
    
    # Create and start bot
    bot = FightingGameBot(extractor, ai)
    
    print("Starting bot in 3 seconds... Position Fightcade window!")
    time.sleep(3)
    
    try:
        bot.start()
    except KeyboardInterrupt:
        bot.stop()
        print("Bot stopped by user")