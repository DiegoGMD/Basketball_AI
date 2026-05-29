"""
Basketball AI Tracker — FastAPI backend.

Pipeline:
    1. Client uploads a video → /upload  (stores it under uploads/)
    2. Client starts processing → /process/{file_id}  (spawns a background thread)
    3. Client polls → /status/{file_id}  (progress + live stats)
    4. Client downloads → /download-zip/{file_id}  (processed video + CSV in a zip)

Key components:
    Config          - all tunable constants in one place
    AutoCleanup     - daemon thread that deletes files older than RETENTION_SECONDS
    GameStats       - shot/basket accounting with per-player breakdown
    Visualizer      - OpenCV drawing helpers (HUD, effects, summary screen)
    MinimapRenderer - projects detections onto a 2-D court image via homography
    VideoProcessor  - frame-by-frame YOLO inference + writing the output video
"""

import cv2
import numpy as np
from ultralytics import YOLO
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pathlib import Path
import shutil
import uuid
import uvicorn
import threading
from collections import deque, defaultdict
from enum import Enum
import time
import json     # Added to parse custom thresholds
import csv      # Added to make the resulting csv for the training session
import zipfile  # Added to give the video and the report csv in one file
import io

# ==================== CONFIGURATION ====================
class Config:
    """Centralized configuration for the application."""
    # Paths
    UPLOAD_DIR = Path(__file__).parent / "uploads"
    PROCESSED_DIR = Path(__file__).parent / "processed"
    MODEL_PATH = Path(__file__).parent / "basketball_training" / "yolo26m_5classes_2" / "weights" / "best.pt"

    # TRACKER_PATH = Path(__file__).parent / "tracker" / "bytetrack.yaml" # Faster but in theory better with objects going in and out camera
    TRACKER_PATH = Path(__file__).parent / "tracker" / "botsort.yaml" # Keeps better the classes if always visible

    MINIMAP_PATH = Path(__file__).parent / "tracker" / "minimap.png"
    HOMOGRAPHY_PATH = Path(__file__).parent / "tracker" / "homography.npy"

    # Video Constraints
    MAX_DURATION_SECONDS = 180  # Max processing limit (3 mins)
    TEST_MODE_DURATION = 15     # Duration for test mode
    MIN_VIDEO_WIDTH = 640
    MAX_VIDEO_WIDTH = 1920
    MIN_VIDEO_HEIGHT = 360
    MAX_VIDEO_HEIGHT = 1080

    # Retention Policy
    RETENTION_SECONDS = 600    # 10 Minutes: Files older than this are auto-deleted
    CLEANUP_INTERVAL = 60       # Run cleanup check every 60 seconds

    # Physics & Rules (Time in seconds)
    SHOT_COOLDOWN = 1     # if the model recognizes a shot,  it waits 0.5 seconds before counting another.
                            # Prevente a single shot from being counted 10 times in 10 consecutive frames 
                            # If is too low, might activate the basket with no shot detector

    BASKET_COOLDOWN = 1.0   # the same for the basket recognition
    ANIMATION_DURATION = 1

    # Confidence Thresholds
    THRESHOLDS = {
        0: 0.6,     # Ball
        1: 0.25,    # Ball in Basket
        2: 0.7,     # Player
        3: 0.7,     # Basket
        4: 0.7      # Player Shooting
    }

    # Colors (BGR Format for OpenCV)
    COLORS = {
        0: (0, 165, 255),    # Ball (Orange)
        1: (0, 215, 255),    # Ball in Basket (Gold)
        2: (0, 255, 0),      # Player (Green)
        3: (0, 0, 255),      # Basket (Red)
        4: (255, 100, 0),    # Player Shooting (Blue)
    }
    
    CLASSES = {
        0: "Ball",
        1: "Ball in Basket",
        2: "Player",
        3: "Basket",
        4: "Player Shooting"
    }

    # Must match the court coordinates used by calibrate_new.py.
    # Defaults below are FIBA half-court: 1500 cm wide, 1400 cm deep.
    COURT_WIDTH_CM  = 1500
    COURT_HEIGHT_CM = 1400

# Ensure directories exist
Config.UPLOAD_DIR.mkdir(exist_ok=True)
Config.PROCESSED_DIR.mkdir(exist_ok=True)

# ==================== ENUMS ====================
class ProcessingMode(str, Enum):
    STATS_ONLY = "stats_only"
    STATS_EFFECTS = "stats_effects"
    FULL_TRACKING = "full_tracking"

# ==================== GLOBAL STATE ====================
processing_status = {}
stop_flags = {}


def validate_video_dimensions(width: int, height: int) -> None:
    """Reject videos outside the supported resolution window."""
    if width < Config.MIN_VIDEO_WIDTH or width > Config.MAX_VIDEO_WIDTH:
        raise RuntimeError(
            f"Unsupported video width {width}px. "
            f"Supported range: {Config.MIN_VIDEO_WIDTH}-{Config.MAX_VIDEO_WIDTH}px."
        )
    if height < Config.MIN_VIDEO_HEIGHT or height > Config.MAX_VIDEO_HEIGHT:
        raise RuntimeError(
            f"Unsupported video height {height}px. "
            f"Supported range: {Config.MIN_VIDEO_HEIGHT}-{Config.MAX_VIDEO_HEIGHT}px."
        )


def compute_panel_layout(frame_w: int, frame_h: int, minimap_shape: tuple[int, int, int]) -> dict:
    """Build a right-panel layout that fits the full supported input range.

    The minimap panel height is derived from the actual scaled minimap image so
    that any minimap (portrait, landscape, square) is handled correctly.
    All remaining vertical space is given to the stats area.
    """
    minimap_src_h, minimap_src_w = minimap_shape[:2]
    right_w = max(320, min(480, int(frame_w * 0.33)))
    final_w = frame_w + right_w
    final_h = frame_h
    pad = 10

    # Scale the minimap to fit the full panel width (minus padding), keeping
    # its aspect ratio.  The height is determined by the image itself — we
    # never reserve a fixed block for it.
    avail_w = max(1, right_w - pad * 2)
    mm_scale = avail_w / minimap_src_w           # fit width exactly
    mm_w = avail_w                               # fills available width
    mm_h = max(1, int(minimap_src_h * mm_scale))

    # The minimap panel is exactly as tall as the scaled image (plus padding).
    minimap_panel_h = mm_h + pad * 2

    # Stats panel gets everything above the minimap panel.
    stats_h = max(1, final_h - minimap_panel_h)

    # Center the minimap image inside its panel (vertically and horizontally).
    x_offset = (right_w - mm_w) // 2
    y_offset = stats_h + pad   # top of minimap panel + one pad unit

    return {
        "right_w": right_w,
        "final_w": final_w,
        "final_h": final_h,
        "stats_h": stats_h,
        "minimap_panel_h": minimap_panel_h,
        "mm_w": mm_w,
        "mm_h": mm_h,
        "mm_x": x_offset,
        "mm_y": y_offset,
    }

# ==================== HOMOGRAPHY ====================
def load_homography():
    """Loads the homography matrix saved by calibrate_new.py."""
    path = Config.HOMOGRAPHY_PATH
    if not path.exists():
        print("⚠️  homography.npy not found — minimap dots will be disabled.")
        print("   Run:  python calibrate_new.py --video uploads\\your_video.mp4")
        return None, 1.0
    H = np.load(str(path))
    scale_path = path.parent / "homography_scale.npy"
    scale = float(np.load(str(scale_path))[0]) if scale_path.exists() else 1.0
    print(f"✅ Homography loaded. Calibration scale: {scale:.4f}")
    return H, scale

homography_matrix, homography_scale = load_homography() # None if not calibrated yet

# ==================== AI MODEL ====================
def load_model():
    """Loads the YOLO model with error handling."""
    print("🔄 Loading AI Model...")
    if not Config.MODEL_PATH.exists():
        raise FileNotFoundError(f"❌ Model not found at {Config.MODEL_PATH}")
    model = YOLO(str(Config.MODEL_PATH))
    print("✅ Model loaded successfully!")
    return model

yolo_model = load_model()

# ==================== CLEANUP SERVICE ====================
class AutoCleanup:
    """Background service to delete old files and free up space."""
    
    @staticmethod
    def start():
        """Starts the cleanup thread."""
        thread = threading.Thread(target=AutoCleanup._cleanup_loop, daemon=True)
        thread.start()
        print(f"🧹 Auto-Cleanup started. Retention: {Config.RETENTION_SECONDS}s")

    @staticmethod
    def _cleanup_loop():
        """Runs periodically to remove old files."""
        while True:
            time.sleep(Config.CLEANUP_INTERVAL)
            try:
                now = time.time()
                deleted_count = 0
                
                # 1. Clean Uploads
                for f in Config.UPLOAD_DIR.iterdir():
                    if f.is_file() and (now - f.stat().st_mtime) > Config.RETENTION_SECONDS:
                        try:
                            f.unlink()
                            deleted_count += 1
                        except Exception: pass

                # 2. Clean Processed Videos
                for f in Config.PROCESSED_DIR.iterdir():
                    if f.is_file() and (now - f.stat().st_mtime) > Config.RETENTION_SECONDS:
                        try:
                            f.unlink()
                            deleted_count += 1
                        except Exception: pass

                # 3. Clean Memory (Status Dictionary) (To avoid Memory Leak)
                # Remove keys that haven't been updated in a while (using a simplified heuristic here)
                # Since we don't timestamp status updates, we'll just check if the file exists on disk.
                # If file is gone (deleted above), remove status.
                keys_to_remove = []
                for file_id in processing_status:
                    # Check if any file related to this ID still exists
                    has_files = any(Config.UPLOAD_DIR.glob(f"{file_id}.*")) or \
                                any(Config.PROCESSED_DIR.glob(f"{file_id}*"))
                    
                    if not has_files and processing_status[file_id]['status'] != 'processing':
                        keys_to_remove.append(file_id)
                
                for k in keys_to_remove:
                    del processing_status[k]

                if deleted_count > 0:
                    print(f"🧹 Auto-Cleanup: Removed {deleted_count} old files.")

            except Exception as e:
                print(f"⚠️ Cleanup Error: {e}")

# ==================== LOGIC CLASSES ====================

class PlayerStats:
    """Tracks shots and baskets for a single player (identified by track ID)."""
    def __init__(self):
        self.shots_attempted = 0
        self.baskets_made = 0
        self.shot_positions: list[list] = []  # [x_cm, y_cm, scored]

    @property
    def accuracy(self):
        if self.shots_attempted == 0: return 0.0
        return (self.baskets_made / self.shots_attempted) * 100

    def __repr__(self):
        return f"PlayerStats(shots={self.shots_attempted}, baskets={self.baskets_made}, acc={self.accuracy:.2f}%)"


class GameStats:
    """Handles the logic for tracking shots, baskets, and percentages — globally and per-player."""
    def __init__(self, fps):
        self.fps = fps
        self.shots_attempted = 0
        self.baskets_made = 0

        # Per-player stats: { track_id (int) -> PlayerStats }
        self.player_stats: dict[int, PlayerStats] = {}
        # The track ID of the last player who shot (so we can credit the basket to them)
        self._last_shooter_id: int | None = None

        #calculate how many frames the cooldown lasts based on the FPS of the video
        self.shot_cooldown_frames = int(fps * Config.SHOT_COOLDOWN)
        self.basket_cooldown_frames = int(fps * Config.BASKET_COOLDOWN)
        self.anim_duration_frames = int(fps * Config.ANIMATION_DURATION)

        self.last_shot_frame = -self.shot_cooldown_frames
        self.last_basket_frame = -self.basket_cooldown_frames

        self.basket_position = None
        self.last_known_basket_pos = None
        self.animation_frames = deque(maxlen=self.anim_duration_frames)

    def _get_player(self, track_id: int) -> PlayerStats:
        """Lazily create a PlayerStats entry on first sight of a track ID."""
        if track_id not in self.player_stats:
            self.player_stats[track_id] = PlayerStats()
        return self.player_stats[track_id]

    def register_shot(self, frame_idx, track_id: int | None = None, court_pos: tuple | None = None):
        """
        Record a shot attempt if the cooldown has elapsed.

        Args:
            frame_idx:  Current frame number (used to enforce the cooldown).
            track_id:   Tracker ID of the shooting player; Stored so the next
                        basket can be credited to the right person.
            court_pos:  (x_cm, y_cm) on the real court, appended to the player's
                        shot-position list.
        Returns:
            True if the shot was registered, False if still in cooldown.
        """

        if frame_idx - self.last_shot_frame >= self.shot_cooldown_frames:
            self.shots_attempted += 1
            self.last_shot_frame = frame_idx
            if track_id is not None:
                self._last_shooter_id = track_id
            if self._last_shooter_id is not None:
                ps = self._get_player(self._last_shooter_id)
                ps.shots_attempted += 1
                if court_pos is not None:
                    ps.shot_positions.append([court_pos[0], court_pos[1], False])
                print(f"🏀  Shot registered → Player #{self._last_shooter_id}")
            return True
        return False

    def register_basket(self, frame_idx, position=None):
        """
        Record a made basket if the cooldown has elapsed.

        If a basket is detected without a recent shot (AI missed the shooting
        pose), a shot is auto-added so that baskets never exceed attempts.
        The basket is credited to the last known shooter (_last_shooter_id).

        Args:
            frame_idx: Current frame number.
            position:  Pixel (cx, cy) of the basket, used for the score animation.
        Returns:
            True if the basket was registered, False if still in cooldown.
        """

        if frame_idx - self.last_basket_frame >= self.basket_cooldown_frames:
            # If there was no recent shot, auto-add one (AI missed the shooting pose)
            if (frame_idx - self.last_shot_frame) > (self.shot_cooldown_frames * 2):
                self.shots_attempted += 1
                self.last_shot_frame = frame_idx
                if self._last_shooter_id is not None:
                    self._get_player(self._last_shooter_id).shots_attempted += 1
                    print(f"⚠️   Basket detected without shot. Auto-added shot.   ⚠️")

            self.baskets_made += 1
            self.last_basket_frame = frame_idx
            self.basket_position = position

            if self.shots_attempted < self.baskets_made:
                self.shots_attempted = self.baskets_made
                print(f"⚠️   Shots < baskets correction applied.   ⚠️")

            # Credit the basket to the last known shooter
            if self._last_shooter_id is not None:
                ps = self._get_player(self._last_shooter_id)
                # Guard: baskets can't exceed shots for this player either
                if ps.baskets_made < ps.shots_attempted:
                    ps.baskets_made += 1
                    if ps.shot_positions: ps.shot_positions[-1][2] = True  # mark last shot as scored
                    print(f"✅  Basket credited → Player #{self._last_shooter_id}")

            self.animation_frames.clear()
            for i in range(self.anim_duration_frames):
                self.animation_frames.append(frame_idx + i)
            return True
        return False

    #calculate the global shoot percentage (%)
    @property
    def accuracy(self):
        if self.shots_attempted == 0: return 0.0
        return (self.baskets_made / self.shots_attempted) * 100

    def get_animation_progress(self, current_frame):
        """Return 0.0–1.0 progress through the basket animation, or 0.0 if inactive."""
        
        if current_frame not in self.animation_frames: return 0.0
        delta = current_frame - self.last_basket_frame
        return min(1.0, delta / self.anim_duration_frames)


class Visualizer:
    """Handles all drawing operations on the video frames."""
    
    # Draw animation (pulsing concentric circles) when you score a basket. (Use math,sin and alpha, to make a fade effect)
    @staticmethod
    def draw_basket_effect(frame, center_pos, progress):
        if not center_pos: return
        cx, cy = center_pos
        alpha = 1.0
        if progress < 0.15: alpha = progress / 0.15
        elif progress > 0.85: alpha = (1.0 - progress) / 0.15
        
        overlay = frame.copy()
        for i in range(4):
            delay = i * 0.1
            local_prog = max(0, min(1, (progress - delay) / (1 - delay)))
            if local_prog > 0:
                radius = int(20 + local_prog * 100)
                thickness = max(2, int(8 * (1 - local_prog)))
                cv2.circle(overlay, (cx, cy), radius, (0, 215, 255), thickness)
        
        pulse = 1.0 + np.sin(progress * np.pi * 4) * 0.3
        cv2.circle(overlay, (cx, cy), int(15 * pulse), (0, 255, 255), -1)
        cv2.addWeighted(overlay, alpha * 0.7, frame, 1 - alpha * 0.3, 0, frame)

    # Draw the scoreboard at the bottom (Shots, Baskets, Accuracy). Make it semi-transparent to make it readable.
    @staticmethod
    def draw_hud(frame, stats, w, h):
        panel_h, panel_w = 100, min(700, w - 30)
        x, y = 15, h - panel_h - 15
        
        sub_img = frame[y:y+panel_h, x:x+panel_w]
        white_rect = np.full(sub_img.shape, 30, dtype=np.uint8)
        res = cv2.addWeighted(sub_img, 0.2, white_rect, 0.8, 0)
        frame[y:y+panel_h, x:x+panel_w] = res
        cv2.rectangle(frame, (x, y), (x+panel_w, y+panel_h), (0, 200, 255), 2)
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        def draw_stat_col(offset_x, label, value, color=(255,255,255)):
            cv2.putText(frame, label, (x + offset_x, y + 30), font, 0.5, (180,180,180), 1)
            cv2.putText(frame, str(value), (x + offset_x, y + 70), font, 1.3, color, 3)

        col_w = panel_w // 3
        draw_stat_col(20, "SHOTS", stats.shots_attempted)
        draw_stat_col(20 + col_w, "BASKETS", stats.baskets_made, (0, 255, 100))
        
        acc_x = x + 2 * col_w + 20
        cv2.putText(frame, "ACCURACY", (acc_x, y + 30), font, 0.5, (180,180,180), 1)
        if stats.accuracy == 100:
            cv2.putText(frame, f"{stats.accuracy:.0f}%", (acc_x, y + 70), font, 1.0, (0, 255, 255), 2)
        else:
            cv2.putText(frame, f"{stats.accuracy:.2f}%", (acc_x, y + 70), font, 1.0, (0, 255, 255), 2)
        
        bar_x, bar_y = acc_x, y + 80
        bar_w = col_w - 40
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 8), (60,60,60), -1)
        fill_w = int((stats.accuracy / 100) * bar_w)
        if fill_w > 0:
            cv2.rectangle(frame, (bar_x, bar_y), (bar_x + fill_w, bar_y + 8), (0, 200, 255), -1)
    
# ==================== MINIMAP RENDERER ====================
class MinimapRenderer:
    """
    Projects detected players and the ball onto the 2D court minimap
    using the pre-computed homography matrix.

    MINIMAP ORIENTATION (minimap.png — half-court view):
      TOP    of image = BASELINE  (basket end, FAR from camera)
      BOTTOM of image = HALF-COURT LINE (NEAR the camera [may be behind the camera])

    This matches what the camera sees in reverse:
      In the video:   basket is at the FAR end (bottom of perspective view)
      In the minimap: basket is at the TOP of the image

    calibrate_new.py uses an image-oriented x-axis:
      x_cm = 0       → right side of the video image
      x_cm = COURT_W → left side of the video image

    To draw correctly on the minimap, we mirror x back to court-left/court-right.

    The minimap has a thin white border on all sides before the court area
    starts. We account for this so dots land on the correct court features
    rather than being offset by the border width.
    """

    # Real-world court size (cm) — must match calibrate_new.py
    COURT_W = Config.COURT_WIDTH_CM    # total width (sideline to sideline)
    COURT_H = Config.COURT_HEIGHT_CM   # visible half-court depth (baseline → half-court line)

    # Fractional offsets of the actual court area within the minimap image.
    # Tuned for the new minimap.png (1042x974 px, very thin outer border).
    # These are resolution-independent — multiply by mm_w / mm_h at draw time.
    _F_LEFT  = 0.005    # left sideline
    _F_RIGHT = 0.995    # right sideline
    _F_TOP   = 0.005    # baseline (basket end) — TOP of minimap
    _F_BOT   = 0.995    # half-court line       — BOTTOM of minimap

    @staticmethod
    def project_point(pixel_xy: tuple, H: np.ndarray, scale: float = 1.0) -> tuple | None:
        """
        Transforms a single image pixel (x, y) → real-world court (x_cm, y_cm)
        using the homography matrix H.
        Returns None if the point projects outside the court bounds.
        """
        # Scale pixel from video resolution → calibration resolution
        px = float(pixel_xy[0]) * scale
        py = float(pixel_xy[1]) * scale
        pt = np.array([[[px, py]]], dtype=np.float32)
        projected = cv2.perspectiveTransform(pt, H)
        x_cm, y_cm = projected[0][0]

        # Reject points that land far outside the court
        if x_cm < -200 or x_cm > MinimapRenderer.COURT_W + 200:
            # print("Position outside the court [hori]")
            return None
        if y_cm < -200 or y_cm > MinimapRenderer.COURT_H + 200:
            # print("Position outside the court [vert]")
            return None
        
        return (x_cm, y_cm)

    @staticmethod
    def court_to_minimap(x_cm: float, y_cm: float, mm_w: int, mm_h: int) -> tuple:
        """
        Converts real-world court coordinates (cm) → minimap pixel coordinates,
        accounting for the border padding in the minimap image.

        calibrate_new.py coordinate system:
          x_cm = 0          → right side of the image / right-labelled points
          x_cm = COURT_W    → left side of the image / left-labelled points
          y_cm = 0          → baseline (basket end)  → TOP    of minimap
          y_cm = COURT_H    → half-court line        → BOTTOM of minimap

        minimap.png coordinate system:
          left side of minimap  = left sideline on the court drawing
          right side of minimap = right sideline on the court drawing

        So x must be mirrored before drawing.
        """
        r = MinimapRenderer

        # Pixel extents of the court drawing within the (possibly resized) minimap
        court_left  = r._F_LEFT  * mm_w
        court_right = r._F_RIGHT * mm_w
        court_top   = r._F_TOP   * mm_h
        court_bot   = r._F_BOT   * mm_h
        court_w_px  = court_right - court_left
        court_h_px  = court_bot   - court_top

        # Map court cm → minimap pixel, clamp to court area
        t = x_cm / r.COURT_W                # 0.0 (left) … 1.0 (right)
        px = court_left + t * court_w_px

        # y_cm=0 (baseline) → top of minimap; y_cm=COURT_H → bottom
        s = y_cm / r.COURT_H                        # 0.0 (baseline) … 1.0 (half-court)
        py = court_top + s * court_h_px

        px = int(np.clip(px, court_left,  court_right))
        py = int(np.clip(py, court_top,   court_bot))
        return (px, py)

    @staticmethod
    def draw(minimap_base: np.ndarray, results, H: np.ndarray,
             thresholds: dict, mm_w: int, mm_h: int) -> np.ndarray:
        """
        Takes the static minimap image as base, draws live player/ball dots on top,
        and returns the annotated minimap copy (does NOT modify the original).

        - Players (cls 2) and Player Shooting (cls 4) → colored dots with track ID
        - Ball (cls 0) → smaller orange dot
        - Basket and Ball-in-Basket are not projected (not on the floor plane)
        """
        mm = minimap_base.copy()

        if not results or results[0] is None:
            return mm
        boxes_data = results[0].boxes
        if boxes_data is None or len(boxes_data) == 0:
            return mm  # No tracking data yet, return blank minimap

        if H is None:
            # No homography: draw a warning on the minimap
            cv2.putText(mm, "No homography.npy", (10, mm_h // 2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
            cv2.putText(mm, "Run calibrate_new.py", (10, mm_h // 2 + 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 80, 255), 1)
            return mm

        boxes = boxes_data.xyxy.cpu().numpy()   # x1,y1,x2,y2
        track_ids = [None] * len(boxes_data)
        if boxes_data.id is not None:
            ids_list = boxes_data.id.int().cpu().tolist()
            for j, tid in enumerate(ids_list):
                track_ids[j] = tid

        classes = boxes_data.cls.int().cpu().tolist()
        confs = boxes_data.conf.cpu().tolist()

        # Collect player centers for ID assignment
        player_centers = {}  # track_id -> center
        player_boxes = {}
        for i, (box, cls, conf, tid) in enumerate(zip(boxes, classes, confs, track_ids)):
            if cls == 2 and conf >= thresholds.get(2, 0.3) and tid is not None:
                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                player_centers[tid] = center
                player_boxes[tid] = (x1, y1, x2, y2)

        # Assign IDs to shooting players without ID by finding closest player
        for i in range(len(boxes)):
            cls = classes[i]
            conf = confs[i]
            tid = track_ids[i]
            if cls == 4 and conf >= thresholds.get(4, 0.3):
                # Find closest cls 2 player by proximity
                x1, y1, x2, y2 = boxes[i]
                shoot_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                closest_tid = None
                min_dist = float('inf')
                for pid, pcenter in player_centers.items():
                    dist = ((shoot_center[0] - pcenter[0]) ** 2 + (shoot_center[1] - pcenter[1]) ** 2) ** 0.5
                    if dist < min_dist:
                        min_dist = dist
                        closest_tid = pid
                if closest_tid is not None and min_dist < 200:  # increased from 100
                    track_ids[i] = closest_tid

        for i, (box, cls, conf) in enumerate(zip(boxes, classes, confs)):
            if conf < thresholds.get(cls, 0.3):
                continue

            # Only project objects that are on the floor (players and ball)
            # Skip cls 0 (ball), cls 1 (ball in basket), 3 (basket) — they are elevated or difficult to draw precisely
            if cls not in (2, 4):
                continue

            x1, y1, x2, y2 = box

            # Use foot position for players (bottom-center of bounding box)
            # Use center for ball (it is often in the air, but this is an approximation)
            if cls in (2, 4):
                foot_x = (x1 + x2) / 2
                foot_y = y2          # bottom of bounding box = feet on floor
            else:
                foot_x = (x1 + x2) / 2
                foot_y = (y1 + y2) / 2

            court_pos = MinimapRenderer.project_point((foot_x, foot_y), H, scale=homography_scale)
            if court_pos is None:
                if cls in (2, 4):
                    print(f"[minimap] cls={cls} foot=({foot_x:.0f},{foot_y:.0f}) → project_point returned None")
                continue

            mm_px = MinimapRenderer.court_to_minimap(court_pos[0], court_pos[1], mm_w, mm_h)

            # Draw the dot
            color     = Config.COLORS.get(cls, (200, 200, 200))
            dot_size  = 6  # players dot size

            cv2.circle(mm, mm_px, dot_size + 2, (0, 0, 0), -1)    # black outline
            cv2.circle(mm, mm_px, dot_size,     color,     -1)

            # Label players with their track ID
            if cls in (2, 4) and track_ids[i] is not None:
                tid = track_ids[i]
                cv2.putText(mm, str(tid), (mm_px[0] + 7, mm_px[1] + 4),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

        return mm

class VideoProcessor:
    """Manages the video processing loop."""
    def __init__(self, file_id, input_path, output_path, test_mode, mode: ProcessingMode, thresholds: dict = None):
        self.file_id = file_id
        self.input_path = input_path
        self.output_path = output_path
        self.test_mode = test_mode
        self.mode = mode
        # Use custom thresholds if provided, otherwise use defaults
        self.thresholds = thresholds if thresholds else Config.THRESHOLDS
        self.track_history = defaultdict(lambda: [])
        self.last_player_ids: dict = {}  # bbox_region -> stable_id
        
    def run(self):
        try:
            # open video 
            cap = cv2.VideoCapture(str(self.input_path))
            if not cap.isOpened(): raise RuntimeError("Could not open video file.")

            # reading video settings (FPS, width, height, total frames)    
            fps = cap.get(cv2.CAP_PROP_FPS)
            w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            validate_video_dimensions(w, h)

            if fps <= 0:
                fps = 30.0
            
            if self.test_mode: max_frames = int(fps * Config.TEST_MODE_DURATION)
            else: max_frames = min(total_frames, int(fps * Config.MAX_DURATION_SECONDS))

            print("🔄 Loading Tracker...")
            if not Config.TRACKER_PATH.exists():
                raise FileNotFoundError(f"❌ Tracker not found at {Config.MODEL_PATH}")
            else:
                print("✅ Tracker loaded successfully!")

            # prepare the minimap
            minimap = cv2.imread(str(Config.MINIMAP_PATH))
            if minimap is None:
                raise FileNotFoundError("❌ minimap.png not found")

            layout = compute_panel_layout(w, h, minimap.shape)
            FINAL_W = layout["final_w"]
            FINAL_H = layout["final_h"]
            RIGHT_W = layout["right_w"]
            STATS_H = layout["stats_h"]
            MINIMAP_H = layout["minimap_panel_h"]
            mm_w = layout["mm_w"]
            mm_h = layout["mm_h"]
            mm_x = layout["mm_x"]
            mm_y = layout["mm_y"]

            writer = cv2.VideoWriter(
                str(self.output_path),
                cv2.VideoWriter_fourcc(*'mp4v'),
                fps,
                (FINAL_W, FINAL_H)
            )
                        
            # prepare the "scoreboard" by resetting it
            stats = GameStats(fps)
            frame_idx = 0
            self.track_history = defaultdict(lambda: [])  # Reset trails each run
            
            self._update_status("processing", 0, max_frames, stats)
            print(f"🎬 Processing {self.file_id} | Mode: {self.mode.value} | Frames: {max_frames}")

            minimap_resized = cv2.resize(minimap, (mm_w, mm_h))

            # read frame by frame using a while loop
            while cap.isOpened() and frame_idx < max_frames:
                # check if the user is pressed "Stop"
                if stop_flags.get(self.file_id, False):
                    print(f"🛑 Stopped by user.")
                    break

                # take the single current frame    
                success, frame = cap.read()
                if not success: break
                
                # create a copy of the original frame (it is a good practice)
                annotated = frame.copy()
                
                # --- TRACKING --- (the model analyzes the original frame, but modifies the copy )
                # detection using the model

                results = yolo_model.track(
                    frame, persist=True, verbose=False, 
                    conf=0.25, tracker=Config.TRACKER_PATH, imgsz=640, iou=0.4
                )

                # Guard: tracker can return [None] on the first frame
                if not results or results[0] is None:
                    writer.write(np.hstack((annotated, np.zeros((FINAL_H, RIGHT_W, 3), dtype=np.uint8))))
                    frame_idx += 1
                    continue

                # --- LOGIC --- 
                # Update scores if it finds shots or baskets. 
                self._process_detections(results, stats, frame_idx)
                
                # --- DRAWING LOGIC BASED ON MODE ---
                
                # 1. Boxes (Only in FULL_TRACKING)
                if self.mode == ProcessingMode.FULL_TRACKING:
                    self._draw_yolo_boxes(annotated, results)
                    self._draw_tracking_trails(annotated, results)  # Motion trails
                
                # 2. Effects (In FULL_TRACKING or STATS_EFFECTS)
                if self.mode in [ProcessingMode.FULL_TRACKING, ProcessingMode.STATS_EFFECTS]:
                    if stats.get_animation_progress(frame_idx) > 0:
                        Visualizer.draw_basket_effect(annotated, stats.basket_position, stats.get_animation_progress(frame_idx))
                
                # 3. HUD (Only in STATS EFFECTS)
                # [The stats panel is already in FULL_TRACKING mode]
                if self.mode == ProcessingMode.STATS_EFFECTS:
                    Visualizer.draw_hud(annotated, stats, w, h)
                
                # Writes the modified frame to the new video file.
                # --- RIGHT PANEL ---
                right_panel = np.zeros((FINAL_H, RIGHT_W, 3), dtype=np.uint8)
                right_panel[:] = (25, 25, 25)  # dark background


                # --- STATS PANEL (TOP) ---
                stats_panel = np.zeros((STATS_H, RIGHT_W, 3), dtype=np.uint8)
                font = cv2.FONT_HERSHEY_SIMPLEX

                # ── Header ────────────────────────────────────────────────
                cv2.putText(stats_panel, "STATS", (15, 35), font, 0.8, (255, 255, 255), 2)
                cv2.line(stats_panel, (10, 45), (RIGHT_W - 10, 45), (80, 80, 80), 1)

                # ── Global totals row ──────────────────────────────────────
                cv2.putText(stats_panel, f"Shots:    {stats.shots_attempted}", (15, 75), font, 0.6, (200, 200, 200), 1)
                cv2.putText(stats_panel, f"Baskets:  {stats.baskets_made}",    (15, 105), font, 0.6, (0, 255, 100), 1)
                if stats.accuracy == 100:
                    cv2.putText(stats_panel, f"Accuracy: {stats.accuracy:.0f}%",   (15, 135), font, 0.65, (0, 255, 255), 2)
                else:
                    cv2.putText(stats_panel, f"Accuracy: {stats.accuracy:.2f}%",   (15, 135), font, 0.65, (0, 255, 255), 2)

                # Accuracy bar
                bar_x, bar_y, bar_w, bar_h_px = 15, 148, RIGHT_W - 30, 8
                cv2.rectangle(stats_panel, (bar_x, bar_y), (bar_x + bar_w, bar_y + bar_h_px), (60, 60, 60), -1)
                fill = int((stats.accuracy / 100) * bar_w)
                if fill > 0:
                    cv2.rectangle(stats_panel, (bar_x, bar_y), (bar_x + fill, bar_y + bar_h_px), (0, 200, 255), -1)

                # ── Per-player table ───────────────────────────────────────
                if stats.player_stats:
                    cv2.line(stats_panel, (10, 168), (RIGHT_W - 10, 168), (80, 80, 80), 1)
                    cv2.putText(stats_panel, "PLAYER BREAKDOWN", (15, 188), font, 0.5, (180, 180, 50), 1)

                    # Column headers
                    cv2.putText(stats_panel, "ID", (15, 210), font, 0.42, (140, 140, 140), 1)
                    cv2.putText(stats_panel, "SHOTS", (70, 210), font, 0.42, (140, 140, 140), 1)
                    cv2.putText(stats_panel, "MADE", (170, 210), font, 0.42, (140, 140, 140), 1)
                    cv2.putText(stats_panel, "ACC%", (260, 210), font, 0.42, (140, 140, 140), 1)
                    cv2.line(stats_panel, (10, 215), (RIGHT_W - 10, 215), (60, 60, 60), 1)

                    # One row per player, sorted by shots descending
                    row_y = 233
                    row_gap = 28
                    sorted_players = sorted(stats.player_stats.items(), key=lambda kv: kv[1].shots_attempted, reverse=True)
                    for pid, ps in sorted_players:
                        if row_y + row_gap > STATS_H - 10:
                            break  # Panel is full

                        # Highlight the most recent shooter
                        is_last_shooter = (pid == stats._last_shooter_id)
                        id_color = (0, 200, 255) if is_last_shooter else (255, 255, 255)

                        cv2.putText(stats_panel, f"#{pid}", (15, row_y), font, 0.55, id_color, 1)
                        cv2.putText(stats_panel, str(ps.shots_attempted), (80, row_y), font, 0.55, (200, 200, 200), 1)
                        cv2.putText(stats_panel, str(ps.baskets_made), (180, row_y), font, 0.55, (0, 255, 100), 1)
                        acc_str = f"{ps.accuracy:.0f}%"
                        acc_color = (0, 255, 100) if ps.accuracy >= 50 else (0, 140, 255)
                        cv2.putText(stats_panel, acc_str, (260, row_y), font, 0.55, acc_color, 1)
                        row_y += row_gap
                else:
                    cv2.putText(stats_panel, "Waiting for players...", (15, 190), font, 0.45, (100, 100, 100), 1)

                right_panel[0:STATS_H, :] = stats_panel

                # --- MINIMAP (BOTTOM) — live dots projected via homography ---
                live_minimap = MinimapRenderer.draw(
                    minimap_resized,       # static base image (never modified)
                    results,               # current frame detections with track IDs
                    homography_matrix,     # loaded once at startup (None if not calibrated)
                    self.thresholds,
                    mm_w,
                    mm_h
                )

                right_panel[
                    mm_y:mm_y + mm_h,
                    mm_x:mm_x + mm_w
                ] = live_minimap


                # --- COMBINE (NO RESIZE OF VIDEO) ---
                final_frame = np.hstack((annotated, right_panel))


                # --- CLEAN SEPARATORS ---
                cv2.line(final_frame, (w, 0), (w, FINAL_H), (255,255,255), 2)
                cv2.line(final_frame, (w, STATS_H), (FINAL_W, STATS_H), (255,255,255), 2)

                # --- BETA DISCLAIMER (top-left corner) ---
                disclaimer_text = "TEST VIDEO - Program in development"
                disclaimer_font = cv2.FONT_HERSHEY_SIMPLEX
                disclaimer_scale = 0.55
                disclaimer_thickness = 1
                (dw, dh), dbaseline = cv2.getTextSize(disclaimer_text, disclaimer_font, disclaimer_scale, disclaimer_thickness)
                pad_x, pad_y = 8, 6
                # Semi-transparent dark background rectangle
                overlay = final_frame.copy()
                cv2.rectangle(overlay, (8, 8), (8 + dw + pad_x * 2, 8 + dh + dbaseline + pad_y * 2), (20, 20, 20), -1)
                cv2.addWeighted(overlay, 0.65, final_frame, 0.35, 0, final_frame)
                # Amber text
                cv2.putText(final_frame, disclaimer_text,
                            (8 + pad_x, 8 + pad_y + dh),
                            disclaimer_font, disclaimer_scale, (0, 180, 255), disclaimer_thickness, cv2.LINE_AA)

                # --- WRITE ---
                writer.write(final_frame)
                frame_idx += 1
                
                # every 30 frames the status is updated (and the progress bar)
                if frame_idx % 30 == 0:
                    self._update_status("processing", frame_idx, max_frames, stats)

            if stop_flags.get(self.file_id, False):
                self._update_status("stopped", frame_idx, max_frames, stats)

            self._update_status("completed", frame_idx, max_frames, stats)
            print(f"✅ Finished. Acc: {stats.accuracy:.2f}%")

                # --- CSV EXPORT ---
            csv_path = Config.PROCESSED_DIR / f"{self.file_id}_stats.csv"
            with open(csv_path, "w", newline="") as f:
                writer_csv = csv.writer(f)
                writer_csv.writerow(["player_id", "shots", "baskets", "accuracy_pct", "shot_positions"])
                for pid, ps in sorted(stats.player_stats.items()):
                    positions_str = "; ".join(f"({x:.2f},{y:.2f},{int(p)})" for x, y, p in ps.shot_positions)
                    writer_csv.writerow([pid, ps.shots_attempted, ps.baskets_made, f"{ps.accuracy:.2f}", positions_str])
            print(f"📄 CSV saved → {csv_path}")

            # close the files
            cap.release()
            writer.release()
            if self.file_id in stop_flags: del stop_flags[self.file_id]

        except Exception as e:
            print(f"❌ Error: {e}")
            processing_status[self.file_id] = {"status": "error", "message": str(e)}

    # Used to understand what is happening in the game and update the score
    def _process_detections(self, results, stats, frame_idx):
        """
        Parse YOLO detections for a single frame and update GameStats.

        Logic:
        - cls 3 (Basket):           updates last_known_basket_pos for animation anchoring.
        - cls 4 (Player Shooting):  calls stats.register_shot(); resolves the shooter's
                                    track ID via IoU overlap with cls-2 player boxes.
        - cls 1 (Ball in Basket):   calls stats.register_basket().
        
        Detections below their class confidence threshold are ignored.
        """

        if not results[0].boxes: return

        boxes_data = results[0].boxes
        track_ids = [None] * len(boxes_data)
        if boxes_data.id is not None:
            ids_list = boxes_data.id.int().cpu().tolist()
            for j, tid in enumerate(ids_list):
                track_ids[j] = tid

        classes = boxes_data.cls.int().cpu().tolist()
        confs = boxes_data.conf.cpu().tolist()
        boxes = boxes_data.xyxy.cpu().numpy()

        # Collect player centers for ID assignment
        player_centers = {}  # track_id -> center
        player_boxes = {}
        for i, (box, cls, conf, tid) in enumerate(zip(boxes, classes, confs, track_ids)):
            if cls == 2 and conf >= self.thresholds.get(2, 0.3) and tid is not None:
                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                player_centers[tid] = center
                player_boxes[tid] = (x1, y1, x2, y2)

        # Assign IDs to shooting players without ID by finding closest player
        for i in range(len(boxes)):
            cls = classes[i]
            conf = confs[i]
            tid = track_ids[i]
            if cls == 4 and conf >= self.thresholds.get(4, 0.3):
                x1, y1, x2, y2 = boxes[i]
                best_tid = None
                best_iou = 0.0
                for pid, (px1, py1, px2, py2) in player_boxes.items():  # need this dict too
                    ix1, iy1 = max(x1, px1), max(y1, py1)
                    ix2, iy2 = min(x2, px2), min(y2, py2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter == 0:
                        continue
                    union = (x2-x1)*(y2-y1) + (px2-px1)*(py2-py1) - inter
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = pid
                if best_tid is not None and best_iou > 0.3:  # at least 30% overlap
                    track_ids[i] = best_tid

        # --- DETECTION PRINT (every 30 frames to avoid flooding the console) ---
        if frame_idx % 30 == 0:
            detected_classes = set()
            for box in boxes_data:
                cls = int(box.cls[0])
                conf = float(box.conf[0])
                if conf >= self.thresholds.get(cls, 0.3):
                    detected_classes.add(Config.CLASSES.get(cls, f"Class {cls}"))
            # Terminal printer of detections
            # if detected_classes:
            #     print(f"[Frame {frame_idx}] 🔍 Detected: {', '.join(sorted(detected_classes))}")
            # else:
            #     print(f"[Frame {frame_idx}] 🔍 Detected: (nothing above threshold)")

        # Scroll through the list of all found objects
        for i, box in enumerate(boxes_data):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < self.thresholds.get(cls, 0.3): continue

            # The model returns a rectangle. Here we calculate the exact center point of that rectangle.
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            center = ((x1 + x2) // 2, (y1 + y2) // 2)

            # Resolve the track ID for this detection (if tracking is active)
            track_id = track_ids[i]

            if cls == 3:
                stats.last_known_basket_pos = center
            elif cls == 4:
                court_pos = None
                if homography_matrix is not None:
                    foot_x = (x1 + x2) / 2
                    foot_y = y2  # bottom of bounding box
                    court_pos = MinimapRenderer.project_point((foot_x, foot_y), H=homography_matrix, scale=homography_scale)
                # Pass track_id so the shot is credited to the right player
                stats.register_shot(frame_idx, track_id=track_id, court_pos=court_pos)
            elif cls == 1:
                target_pos = stats.last_known_basket_pos or center
                stats.register_basket(frame_idx, target_pos)

    def _draw_yolo_boxes(self, frame, results):
        """
        Draw coloured bounding boxes and labels on *frame* for all detections
        that exceed their class confidence threshold.

        Players (cls 2) and Player Shooting (cls 4) are labelled with their
        tracker ID (e.g. "#3 Player 0.82"). Shooting players without a tracker
        ID are assigned one by finding the cls-2 player box with the highest
        IoU overlap.
        """

        if not results[0].boxes: return

        boxes_data = results[0].boxes
        track_ids = [None] * len(boxes_data)
        if boxes_data.id is not None:
            ids_list = boxes_data.id.int().cpu().tolist()
            for j, tid in enumerate(ids_list):
                track_ids[j] = tid

        classes = boxes_data.cls.int().cpu().tolist()
        confs = boxes_data.conf.cpu().tolist()
        boxes = boxes_data.xyxy.cpu().numpy()

        # Collect player centers for ID assignment
        player_centers = {}  # track_id -> center
        player_boxes = {}
        for i, (box, cls, conf, tid) in enumerate(zip(boxes, classes, confs, track_ids)):
            if cls == 2 and conf >= self.thresholds.get(2, 0.3) and tid is not None:
                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                player_centers[tid] = center
                player_boxes[tid] = (x1, y1, x2, y2)

        # Assign IDs to shooting players without ID by finding closest player
        for i in range(len(boxes)):
            cls = classes[i]
            conf = confs[i]
            tid = track_ids[i]
            if cls == 4 and conf >= self.thresholds.get(4, 0.3):
                x1, y1, x2, y2 = boxes[i]
                best_tid = None
                best_iou = 0.0
                for pid, (px1, py1, px2, py2) in player_boxes.items():  # need this dict too
                    ix1, iy1 = max(x1, px1), max(y1, py1)
                    ix2, iy2 = min(x2, px2), min(y2, py2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter == 0:
                        continue
                    union = (x2-x1)*(y2-y1) + (px2-px1)*(py2-py1) - inter
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = pid
                if best_tid is not None and best_iou > 0.3:  # at least 30% overlap
                    track_ids[i] = best_tid

        for i, box in enumerate(boxes_data):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < self.thresholds.get(cls, 0.3): continue

            x1, y1, x2, y2 = map(int, box.xyxy[0])
            color = Config.COLORS.get(cls, (255, 255, 255))

            # Build label: include track ID for players so we can identify them
            class_name = Config.CLASSES.get(cls, f"cls{cls}")
            if cls in (0, 2, 4) and track_ids[i] is not None:
                tid = track_ids[i]
                label = f"#{tid} {class_name} {conf:.2f}"
            else:
                label = f"{class_name} {conf:.2f}"

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            # Draw a filled background behind the label so it is readable on any background
            (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1 - th - baseline - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(frame, label, (x1 + 2, y1 - baseline - 3),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)

    def _draw_tracking_trails(self, frame, results):
        """
        Draw motion-trail polylines for each tracked object.

        The last 30 centre-points of every tracked detection are stored in
        self.track_history[track_id] and connected with a coloured polyline.
        Shooting players (cls 4) without their own tracker ID borrow the ID
        of the overlapping cls-2 player (same IoU logic as _draw_yolo_boxes).
        """

        if not results or results[0] is None:
            return
        boxes_data = results[0].boxes
        if not boxes_data or not boxes_data.is_track:
            return

        boxes = boxes_data.xywh.cpu()
        track_ids = [None] * len(boxes_data)
        if boxes_data.id is not None:
            ids_list = boxes_data.id.int().cpu().tolist()
            for j, tid in enumerate(ids_list):
                track_ids[j] = tid

        classes = boxes_data.cls.int().cpu().tolist()
        confs = boxes_data.conf.cpu().tolist()
        boxes_xyxy = boxes_data.xyxy.cpu().numpy()

        # Collect player centers for ID assignment
        player_centers = {}  # track_id -> center
        player_boxes = {}
        for i, (box, cls, conf, tid) in enumerate(zip(boxes_xyxy, classes, confs, track_ids)):
            if cls == 2 and conf >= self.thresholds.get(2, 0.3) and tid is not None:
                x1, y1, x2, y2 = box
                center = ((x1 + x2) / 2, (y1 + y2) / 2)
                player_centers[tid] = center
                player_boxes[tid] = (x1, y1, x2, y2)

        # Assign IDs to shooting players without ID by finding closest player
        for i in range(len(boxes)):
            cls = classes[i]
            conf = confs[i]
            tid = track_ids[i]
            if cls == 4 and conf >= self.thresholds.get(4, 0.3):
                x1, y1, x2, y2 = boxes_xyxy[i]
                best_tid = None
                best_iou = 0.0
                for pid, (px1, py1, px2, py2) in player_boxes.items():  # need this dict too
                    ix1, iy1 = max(x1, px1), max(y1, py1)
                    ix2, iy2 = min(x2, px2), min(y2, py2)
                    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
                    if inter == 0:
                        continue
                    union = (x2-x1)*(y2-y1) + (px2-px1)*(py2-py1) - inter
                    iou = inter / union
                    if iou > best_iou:
                        best_iou = iou
                        best_tid = pid
                if best_tid is not None and best_iou > 0.3:  # at least 30% overlap
                    track_ids[i] = best_tid

        for box, track_id, cls, conf in zip(boxes, track_ids, classes, confs):
            if conf < self.thresholds.get(cls, 0.3) or track_id is None:
                continue

            x, y, w, h = box
            track = self.track_history[track_id]
            track.append((float(x), float(y)))  # x, y center point
            if len(track) > 30:  # retain 30 frames of history
                track.pop(0)

            if len(track) >= 2:
                color = Config.COLORS.get(cls, (230, 230, 230))
                points = np.hstack(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(frame, [points], isClosed=False, color=color, thickness=3)

    def _update_status(self, status, current, total, stats):
        """
        Write the current processing state into the shared processing_status dict.
        
        The frontend polls /status/{file_id} which reads this dict directly.
        Fields: status, progress (frame count), total, percentage (0-100), stats summary.
        """

        processing_status[self.file_id] = {
            "status": status,
            "progress": current,
            "total": total,
            "percentage": int((current/total)*100) if total > 0 else 0,
            "stats": {"shots": stats.shots_attempted, "baskets": stats.baskets_made, "accuracy": stats.accuracy}
        }

# ==================== FASTAPI APP ====================
app = FastAPI(title="Basketball AI Tracker")
from calibrate_api import router as calibration_router
app.include_router(calibration_router)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"], allow_credentials=True)

@app.on_event("startup")
def startup_event():
    """Starts background services on app startup."""
    AutoCleanup.start()

@app.get("/")
def home(): return {"message": "Basketball AI Tracker is Running", "docs": "/docs"}

@app.post("/upload")
async def upload_video(file: UploadFile = File(...)):
    """Accept a video file, save it with a UUID filename, return the file_id."""

    if not file.content_type.startswith("video/"): raise HTTPException(400, "File must be a video.")
    file_id = str(uuid.uuid4())
    ext = Path(file.filename).suffix
    save_path = Config.UPLOAD_DIR / f"{file_id}{ext}"
    with open(save_path, "wb") as f: shutil.copyfileobj(file.file, f)
    return {"file_id": file_id, "filename": file.filename}

@app.post("/process/{file_id}")
async def start_process(file_id: str, test_mode: bool = False, mode: ProcessingMode = ProcessingMode.FULL_TRACKING, thresholds: str = None):
    """Return the current processing status dict for a given file_id."""

    input_files = list(Config.UPLOAD_DIR.glob(f"{file_id}.*"))
    if not input_files: raise HTTPException(404, "Video not found.")
    
    # Parse custom thresholds if provided
    active_thresholds = Config.THRESHOLDS.copy()
    if thresholds:
        try:
            custom_vals = json.loads(thresholds)
            # Convert string keys to integers
            for k, v in custom_vals.items():
                active_thresholds[int(k)] = float(v)
            print(f"⚙️ Using custom thresholds for {file_id}: {active_thresholds}")
        except Exception as e:
            print(f"⚠️ Failed to parse thresholds: {e}. Using defaults.")

    processor = VideoProcessor(file_id, input_files[0], Config.PROCESSED_DIR / f"{file_id}_processed.mp4", test_mode, mode, active_thresholds)
    thread = threading.Thread(target=processor.run)
    thread.start()
    return {"status": "started", "file_id": file_id, "mode": mode}

@app.get("/status/{file_id}")
def get_status(file_id: str):
    """Return the current processing status dict for a given file_id."""
    return processing_status.get(file_id, {"status": "not_found"})

@app.post("/stop/{file_id}")
def stop_process(file_id: str):
    """Signal the processing thread to stop early by setting a stop flag."""
    
    if file_id in processing_status and processing_status[file_id]['status'] == 'processing':
        stop_flags[file_id] = True
        return {"message": "Stopping..."}
    return {"message": "Not processing or not found"}

@app.get("/download/{file_id}")
def download_result(file_id: str):
    """Stream the processed MP4 video file to the client."""

    path = Config.PROCESSED_DIR / f"{file_id}_processed.mp4"
    if not path.exists(): raise HTTPException(404, "File not ready.")
    
    # We don't delete immediately here to allow retries. 
    # The AutoCleanup service will handle it after RETENTION_SECONDS.
    return FileResponse(path, media_type="video/mp4", filename=f"basket_ai_{file_id}.mp4")

@app.get("/download-csv/{file_id}")
def download_csv(file_id: str):
    """Stream the per-player stats CSV file to the client."""

    path = Config.PROCESSED_DIR / f"{file_id}_stats.csv"
    if not path.exists():
        raise HTTPException(404, "CSV not ready or processing incomplete.")
    return FileResponse(path, media_type="text/csv", filename=f"basket_stats_{file_id}.csv")

@app.get("/download-zip/{file_id}")
def download_zip(file_id: str):
    """Stream the processed MP4 video file and the per-player stats CSV file to the client."""

    video_path = Config.PROCESSED_DIR / f"{file_id}_processed.mp4"
    csv_path = Config.PROCESSED_DIR / f"{file_id}_stats.csv"
    if not video_path.exists() or not csv_path.exists():
        raise HTTPException(404, "Files not ready")
    
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.write(video_path, "B_AI Demo - Video.mp4")
        zf.write(csv_path, "B_AI Demo - Report.csv")
    buf.seek(0)

    from fastapi.responses import StreamingResponse
    return StreamingResponse(
        buf, 
        media_type="application/zip"
        ,headers={
            "Content-Disposition": "attachment;"
            "filename=B_AI_Demo.zip"
        }
    )


if __name__ == "__main__":
    print("\n🏀 SERVER STARTING...")
    uvicorn.run(app, host="0.0.0.0", port=8000)