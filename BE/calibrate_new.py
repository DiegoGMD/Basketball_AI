# =============================================================================
#  calibrate.py  —  SwishAI Court Homography Calibration Tool
#  Place this file in: BE/calibrate.py
#
#  PURPOSE:
#      You click 6 points on a video frame that correspond to known real-world
#      positions on the basketball court. The script computes a homography
#      matrix H and saves it as  BE/homography.npy
#      app.py loads this file automatically at startup.
#
#  RUN ONCE (inside your venv):
#      cd BE
#      python calibrate.py --video uploads\your_sample.mp4
#      -- or --
#      python calibrate.py --image sample_frame.jpg
#
#  CONTROLS (during the click window):
#      Left-click  → place next reference point
#      U           → undo last point
#      R           → reset all points
#      S           → save homography and exit  (requires all 6 points placed)
#      Q / ESC     → quit without saving
#
#  REFERENCE POINTS — you must click these 6 court landmarks IN ORDER:
#      1. Left  sideline where it meets the baseline (bottom-left of court)
#      2. Right sideline where it meets the baseline (bottom-right)
#      3. Left  corner of the three-point line (where it meets the sideline)
#      4. Right corner of the three-point line
#      5. Left  edge of the free-throw lane at the free-throw line
#      6. Right edge of the free-throw lane at the free-throw line
#
#  Their corresponding REAL-WORLD positions on a half-court (in cm) are
#  pre-defined below in COURT_REFERENCE_PTS. Adjust these values if your
#  court uses different dimensions (e.g. FIBA vs NBA, or a shorter court).
#
#  TIPS:
#      • Use a frame with the best court-line visibility (good lighting, no blur)
#      • Zoom in with your OS image viewer to confirm line positions before clicking
#      • More points = more accurate projection.
#        After saving, you can re-run with --extra to add more points (advanced).
# =============================================================================

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Real-world court reference coordinates (x_cm, y_cm) on a top-down half-court
# Origin (0,0) is bottom-left corner of the half-court.
# Y increases toward the basket. X increases to the right.
#
# Default = NBA half-court dimensions (cm)
#   Full court width:  1524 cm  →  used as X range 0–1524
#   Half-court length: 1422 cm  →  used as Y range 0–1422 (baseline at y=0)
#
# These 6 points correspond exactly to the click order described above.
# ---------------------------------------------------------------------------
COURT_REFERENCE_PTS = np.array([
    [   0,    0],    # 1. Bottom-left corner  (left sideline × baseline)
    [1524,    0],    # 2. Bottom-right corner (right sideline × baseline)
    [   0,  861],    # 3. Left  3pt corner    (left sideline × 3pt line)
    [1524,  861],    # 4. Right 3pt corner    (right sideline × 3pt line)
    [ 457,  575],    # 5. Left  free-throw lane edge at free-throw line
    [1067,  575],    # 6. Right free-throw lane edge at free-throw line
], dtype=np.float32)

POINT_LABELS = [
    "1. Bottom-left corner (sideline x baseline)",
    "2. Bottom-right corner (sideline x baseline)",
    "3. Left 3pt corner (sideline x 3pt arc cutoff)",
    "4. Right 3pt corner (sideline x 3pt arc cutoff)",
    "5. Left free-throw lane, at free-throw line",
    "6. Right free-throw lane, at free-throw line",
]

NUM_POINTS  = len(COURT_REFERENCE_PTS)
OUTPUT_PATH = Path(__file__).parent / "homography.npy"

# ---------------------------------------------------------------------------
# UI colours
# ---------------------------------------------------------------------------
DOT_COLOR   = (0,   255,   0)
DONE_COLOR  = (0,   200, 255)
LINE_COLOR  = (100, 100, 100)
TEXT_COLOR  = (255, 255, 255)
HINT_COLOR  = (180, 180,  50)


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
clicked_pts: list[list[int]] = []
display_img: np.ndarray      = None


def _redraw(base_img: np.ndarray) -> np.ndarray:
    img = base_img.copy()
    h, w = img.shape[:2]

    # Draw placed points
    for i, pt in enumerate(clicked_pts):
        color = DONE_COLOR if i < NUM_POINTS - 1 else DOT_COLOR
        cv2.circle(img, tuple(pt), 7, color, -1)
        cv2.circle(img, tuple(pt), 7, (255, 255, 255), 1)
        cv2.putText(img, str(i + 1), (pt[0] + 9, pt[1] - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1)

    # Next-point instruction
    idx = len(clicked_pts)
    if idx < NUM_POINTS:
        hint = f"Click point {idx + 1}: {POINT_LABELS[idx]}"
        cv2.rectangle(img, (0, h - 36), (w, h), (30, 30, 30), -1)
        cv2.putText(img, hint, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, HINT_COLOR, 1)
    else:
        msg = "All points placed!  Press S to SAVE  |  U to undo  |  R to reset"
        cv2.rectangle(img, (0, h - 36), (w, h), (20, 60, 20), -1)
        cv2.putText(img, msg, (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (100, 255, 100), 1)

    # Top legend
    cv2.rectangle(img, (0, 0), (w, 24), (30, 30, 30), -1)
    controls = "U=undo  R=reset  S=save  Q/ESC=quit"
    cv2.putText(img, controls, (8, 17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)

    return img


def _mouse_cb(event, x, y, flags, param):
    global display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_pts) < NUM_POINTS:
            clicked_pts.append([x, y])
            display_img = _redraw(param)
            cv2.imshow("SwishAI Calibration", display_img)


def _compute_and_save():
    if len(clicked_pts) < NUM_POINTS:
        print(f"[calibrate] Need {NUM_POINTS} points, only {len(clicked_pts)} placed.")
        return False

    src = np.array(clicked_pts[:NUM_POINTS], dtype=np.float32)
    dst = COURT_REFERENCE_PTS[:NUM_POINTS]

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        print("[calibrate] ERROR: findHomography failed. Try re-clicking the points.")
        return False

    inliers = int(mask.sum()) if mask is not None else 0
    print(f"[calibrate] Homography computed. Inliers: {inliers}/{NUM_POINTS}")

    np.save(str(OUTPUT_PATH), H)
    print(f"[calibrate] Saved → {OUTPUT_PATH}")
    return True


def _get_frame_from_video(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[calibrate] Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # Jump to 10% into the video (likely past any dark intro)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(int(total * 0.10), total - 1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("[calibrate] Failed to read frame from video.")
    return frame


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    global display_img, clicked_pts

    parser = argparse.ArgumentParser(
        description="SwishAI Court Homography Calibration")
    group  = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to a basketball video file")
    group.add_argument("--image", help="Path to a court image / screenshot")
    parser.add_argument(
        "--frame", type=int, default=None,
        help="Specific frame number to use (video only). Default: 10%% into video.")
    args = parser.parse_args()

    # ---- Load image ----
    if args.video:
        if args.frame is not None:
            cap = cv2.VideoCapture(args.video)
            cap.set(cv2.CAP_PROP_POS_FRAMES, args.frame)
            ret, base_img = cap.read()
            cap.release()
            if not ret:
                sys.exit(f"[calibrate] Cannot read frame {args.frame}.")
        else:
            base_img = _get_frame_from_video(args.video)
    else:
        base_img = cv2.imread(args.image)
        if base_img is None:
            sys.exit(f"[calibrate] Cannot read image: {args.image}")

    # Resize if very large (keep aspect ratio, max width 1280)
    h, w = base_img.shape[:2]
    if w > 1280:
        scale    = 1280 / w
        base_img = cv2.resize(base_img, (1280, int(h * scale)))

    print("=" * 60)
    print("  SwishAI Homography Calibration")
    print("=" * 60)
    print("Click the following 6 court landmarks IN ORDER:")
    for i, lbl in enumerate(POINT_LABELS):
        print(f"  {lbl}")
    print()
    print("Controls:  U=undo  R=reset  S=save & exit  Q/ESC=quit")
    print("=" * 60)

    display_img = _redraw(base_img)

    cv2.namedWindow("SwishAI Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SwishAI Calibration", _mouse_cb, base_img)
    cv2.imshow("SwishAI Calibration", display_img)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):   # Q or ESC
            print("[calibrate] Quit without saving.")
            break

        elif key == ord('u'):       # Undo
            if clicked_pts:
                clicked_pts.pop()
                display_img = _redraw(base_img)
                cv2.imshow("SwishAI Calibration", display_img)

        elif key == ord('r'):       # Reset
            clicked_pts.clear()
            display_img = _redraw(base_img)
            cv2.imshow("SwishAI Calibration", display_img)

        elif key == ord('s'):       # Save
            if _compute_and_save():
                print("[calibrate] Done! You can now run app.py.")
                break
            # else: keep window open so user can fix

        if cv2.getWindowProperty("SwishAI Calibration",
                                  cv2.WND_PROP_VISIBLE) < 1:
            break   # Window was closed by user

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
