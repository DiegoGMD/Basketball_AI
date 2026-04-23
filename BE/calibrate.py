# =============================================================================
#  calibrate_new.py  —  SwishAI Court Homography Calibration Tool
#
#  CAMERA POSITION:
#      Camera sits near the half-court line at 0.5–1 m height, pointing
#      toward the basket. Baseline is at the FAR end (bottom of frame).
#
#  ── SETUP BEFORE YOU RUN ────────────────────────────────────────────────────
#
#  STEP 1 — Choose your court standard.
#      Uncomment ONE of the three blocks in the "COURT STANDARD" section below.
#      If you don't know, measure the free-throw lane width on your court:
#        FIBA  → lane is ~4.9 m wide
#        NBA   → lane is ~4.88 m wide
#        NCAA  → lane is ~3.66 m wide (US college courts)
#      When in doubt, just pick FIBA — it will be very close.
#
#  STEP 2 — Measure the camera distance from the baseline (in cm).
#      Walk from the baseline to where the camera sits and measure.
#      Enter that value as CAMERA_DIST_FROM_BASELINE_CM below.
#      This is used for points 3 & 4 (sideline frame-cut).
#      If you truly cannot measure, leave it at 1400 — it will still work
#      but the near-camera area of the minimap will be less accurate.
#
#  RUN ONCE (inside your venv):
#      python calibrate_new.py --video uploads\your_sample.mp4
#      python calibrate_new.py --video uploads\your_sample.mp4 --frame 120
#      python calibrate_new.py --image sample_frame.jpg
#
#  CONTROLS:
#      Left-click  → place next point
#      U           → undo last point
#      R           → reset all
#      S           → save
#      Q / ESC     → quit without saving
#
# =============================================================================
#
#  REFERENCE POINTS — click EXACTLY in this order:
#
#     [C] camera location
#         (aproximate location where the camera who is fliming the input sits)
#
#     [B] Basket location
#         (aproximate location where the basket being shot at sits)
#     ────────────────────────[C]────────────────────────
#  R  |???????????????                   ???????????????| L
#     |????                                         ????| 
#  s  |?                                               ?| s
#  i [10]            ·········[9]·········           [11] i
#  d  |           ···                     ···           | d
#  e  |        ···          ·······          ···        | e
#  l  |      ··           ··       ··           ··      | l
#  i  |     |        ────[7]───────[8]────        |     | i
#  n  |     |        |    ··       ··    |        |     | n
#  e  |     |        |      ·······      |        |     | e
#     |     |        |                   |        |     |
#     |     |        |       ─[B]─       |        |     |
#    [1]───[3]──────[5]─────────────────[6]──────[4]───[2]
#                          baseline
#     ───────────────────────────────────────────────────
#
#  POINTS:
#   1. Right  baseline corner  (left sideline × baseline)          [FAR-RIGHT]
#   2. Left baseline corner  (right sideline × baseline)           [FAR-LEFT]
#   3. Right  3pt arc × baselin                                    [FAR, inner-right]
#   4. Left 3pt arc × baseline                                     [FAR, inner-left]
#   5. Right free throw box × baseline                             [FAR, inner-right]
#   6. Left free throw box × baseline                              [FAR, inner-left]
#   7. Right free-throw line end                                   [MID-RIGHT]
#   8. Left  free-throw line end                                   [MID-LEFT]
#   9. Top of 3pt arc (peak of the arc, furthest from baseline)    [CLOSE, centre]
#  10. Right  sideline frame-cut (where frame cuts left sideline)  [CLOSE, right]
#  11. Left  sideline frame-cut (where frame cuts left sideline)   [CLOSE, left]
# =============================================================================

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# ---------------------------------------------------------------------------
#  ▼▼▼  CONFIGURE THESE TWO THINGS BEFORE RUNNING  ▼▼▼
# ---------------------------------------------------------------------------

# How far is the camera from the baseline? Measure in cm.
# (Walk from the baseline to the camera tripod/mount and measure.)
CAMERA_DIST_FROM_BASELINE_CM = 1400   # ← change this if you can measure it

# ---------------------------------------------------------------------------
#  COURT STANDARD — uncomment the one that matches your court.
#  Only ONE block should be active at a time.
# ---------------------------------------------------------------------------

# ── FIBA (most gyms outside the US) ──────────────────────────────────────
# Lane width ≈ 4.9 m, 3pt radius = 6.75 m
COURT_W   = 1500    # cm, total court width
BASKET_X  =  750    # cm, basket centre x (middle of court)
BASKET_Y  =  157    # cm, basket centre y from baseline
R_3PT     =  675    # cm, 3pt arc radius
PAINT_L   =  505    # cm, left  edge of paint (free-throw lane)
PAINT_R   =  995    # cm, right edge of paint
FT_Y      =  575    # cm, free-throw line distance from baseline

# ── NBA (US professional) — uncomment to use ─────────────────────────────
# Lane width ≈ 4.88 m, 3pt radius = 7.23 m
# COURT_W  = 1524
# BASKET_X =  762
# BASKET_Y =  157
# R_3PT    =  723
# PAINT_L  =  518
# PAINT_R  = 1006
# FT_Y     =  575

# ── NCAA (US college) — uncomment to use ─────────────────────────────────
# Lane width ≈ 3.66 m, 3pt radius = 6.30 m
# COURT_W  = 1524
# BASKET_X =  762
# BASKET_Y =  157
# R_3PT    =  630
# PAINT_L  =  518
# PAINT_R  = 1006
# FT_Y     =  575

# ---------------------------------------------------------------------------
#  ▲▲▲  END OF CONFIGURATION  ▲▲▲
# ---------------------------------------------------------------------------

# Derived coordinates — do not edit below this line
_val   = R_3PT**2 - BASKET_Y**2
X_3PT_L = BASKET_X - np.sqrt(_val)   # pt 5: left  3pt × baseline
X_3PT_R = BASKET_X + np.sqrt(_val)   # pt 6: right 3pt × baseline
Y_3PT_TOP = BASKET_Y + R_3PT         # pt 7: top of 3pt arc

COURT_REFERENCE_PTS = np.array([
    [0.0,                             0.0                          ],  # 1
    [float(COURT_W),                  0.0                          ],  # 2
    [0.0,                             float(CAMERA_DIST_FROM_BASELINE_CM)],  # 3
    [float(COURT_W),                  float(CAMERA_DIST_FROM_BASELINE_CM)],  # 4
    [float(X_3PT_L),                  0.0                          ],  # 5
    [float(X_3PT_R),                  0.0                          ],  # 6
    [float(BASKET_X),                 float(Y_3PT_TOP)             ],  # 7
    [float(PAINT_L),                  float(FT_Y)                  ],  # 8
    [float(PAINT_R),                  float(FT_Y)                  ],  # 9
], dtype=np.float32)

POINT_LABELS = [
    " 1. LEFT  baseline corner        — left sideline meets baseline     [FAR-LEFT,   bottom of frame]",
    " 2. RIGHT baseline corner        — right sideline meets baseline    [FAR-RIGHT,  bottom of frame]",
    " 3. LEFT  sideline frame-cut     — where frame cuts left sideline   [NEAR-LEFT,  top of frame]",
    " 4. RIGHT sideline frame-cut     — where frame cuts right sideline  [NEAR-RIGHT, top of frame]",
    " 5. LEFT  3pt arc × baseline     — 3pt line meets baseline (left)   [FAR, inner-left]",
    " 6. RIGHT 3pt arc × baseline     — 3pt line meets baseline (right)  [FAR, inner-right]",
    " 7. TOP of 3pt arc               — peak of the arc, centre court    [MID, centre]",
    " 8. LEFT  free-throw line end    — left edge of free-throw line     [MID-LEFT]",
    " 9. RIGHT free-throw line end    — right edge of free-throw line    [MID-RIGHT]",
]

NUM_POINTS = len(COURT_REFERENCE_PTS)   # 9
OUTPUT_PATH = Path(__file__).parent / "tracker" / "homography.npy"

# ---------------------------------------------------------------------------
# UI colours
# ---------------------------------------------------------------------------
DOT_COLOR  = (0,   255,   0)
DONE_COLOR = (0,   200, 255)
TEXT_COLOR = (255, 255, 255)
HINT_COLOR = (180, 180,  50)

# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------
clicked_pts: list[list[int]] = []
display_img: np.ndarray      = None


def _redraw(base_img: np.ndarray) -> np.ndarray:
    img = base_img.copy()
    h, w = img.shape[:2]

    for i, pt in enumerate(clicked_pts):
        is_latest = (i == len(clicked_pts) - 1)
        color     = DOT_COLOR if is_latest else DONE_COLOR
        cv2.circle(img, tuple(pt), 8, color, -1)
        cv2.circle(img, tuple(pt), 9, (0, 0, 0), 1)
        cv2.putText(img, str(i + 1), (pt[0] + 10, pt[1] - 7),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    idx = len(clicked_pts)
    cv2.rectangle(img, (0, h - 52), (w, h), (30, 30, 30), -1)
    if idx < NUM_POINTS:
        cv2.putText(img, f"Point {idx + 1}/{NUM_POINTS}:",
                    (8, h - 32), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 180, 50), 1)
        cv2.putText(img, POINT_LABELS[idx],
                    (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.42, HINT_COLOR, 1)
    else:
        cv2.rectangle(img, (0, h - 52), (w, h), (20, 60, 20), -1)
        cv2.putText(img, f"All {NUM_POINTS} points placed!  S=SAVE  |  U=undo  |  R=reset",
                    (8, h - 22), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (100, 255, 100), 1)

    cv2.rectangle(img, (0, 0), (w, 26), (30, 30, 30), -1)
    cv2.putText(img, "U=undo  R=reset  S=save  Q/ESC=quit without saving",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.45, TEXT_COLOR, 1)

    return img


def _mouse_cb(event, x, y, flags, param):
    global display_img
    if event == cv2.EVENT_LBUTTONDOWN:
        if len(clicked_pts) < NUM_POINTS:
            clicked_pts.append([x, y])
            display_img = _redraw(param)
            cv2.imshow("SwishAI Calibration", display_img)


def _compute_and_save() -> bool:
    n = len(clicked_pts)
    if n < NUM_POINTS:
        print(f"[calibrate] Need all {NUM_POINTS} points. Only {n} placed.")
        return False

    src = np.array(clicked_pts, dtype=np.float32)
    dst = COURT_REFERENCE_PTS

    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        print("[calibrate] ERROR: findHomography failed.")
        print("            Try clicking the points more precisely and retry.")
        return False

    inliers = int(mask.sum()) if mask is not None else 0
    print(f"[calibrate] Homography computed. Inliers: {inliers}/{NUM_POINTS}")
    if inliers < 6:
        print(f"[calibrate] WARNING: only {inliers} inliers — accuracy may be poor.")
        print("            Re-run and click the intersections more precisely.")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_PATH), H)
    print(f"[calibrate] Saved → {OUTPUT_PATH}")
    return True


def _get_frame_from_video(video_path: str) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[calibrate] Cannot open video: {video_path}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        description="SwishAI Calibration")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--video", help="Path to a basketball video file")
    group.add_argument("--image", help="Path to a court image / screenshot")
    parser.add_argument("--frame", type=int, default=None,
                        help="Specific frame number to use (video only).")
    args = parser.parse_args()

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

    h, w = base_img.shape[:2]
    if w > 1280:
        scale    = 1280 / w
        base_img = cv2.resize(base_img, (1280, int(h * scale)))

    print("=" * 70)
    print("  SwishAI Calibration")
    print("=" * 70)
    print(f"  Court standard : {'FIBA' if COURT_W == 1500 else 'NBA/NCAA'} "
          f"(court width {COURT_W} cm, 3pt radius {R_3PT} cm)")
    print(f"  Camera distance: {CAMERA_DIST_FROM_BASELINE_CM} cm from baseline")
    print()
    print("  COMPUTED REFERENCE COORDINATES:")
    for i, (lbl, pt) in enumerate(zip(POINT_LABELS, COURT_REFERENCE_PTS)):
        print(f"    {lbl}")
        print(f"       → real-world ({pt[0]:.1f} cm, {pt[1]:.1f} cm)")
    print()
    print("  FRAME ORIENTATION:")
    print("    TOP    of image = near camera  (sideline cuts, top of 3pt arc)")
    print("    BOTTOM of image = far from camera (baseline, basket)")
    print()
    print("  TIP: Click exact line intersections. Use --frame N for clearest frame.")
    print("  TIP: If inliers < 7, re-run and click more carefully.")
    print("=" * 70)

    display_img = _redraw(base_img)
    cv2.namedWindow("SwishAI Calibration", cv2.WINDOW_NORMAL)
    cv2.setMouseCallback("SwishAI Calibration", _mouse_cb, base_img)
    cv2.imshow("SwishAI Calibration", display_img)

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (ord('q'), 27):
            print("[calibrate] Quit without saving.")
            break
        elif key == ord('u'):
            if clicked_pts:
                removed = clicked_pts.pop()
                print(f"[calibrate] Undo — removed point {len(clicked_pts) + 1} at {removed}")
                display_img = _redraw(base_img)
                cv2.imshow("SwishAI Calibration", display_img)
        elif key == ord('r'):
            clicked_pts.clear()
            print("[calibrate] Reset — all points cleared.")
            display_img = _redraw(base_img)
            cv2.imshow("SwishAI Calibration", display_img)
        elif key == ord('s'):
            if _compute_and_save():
                print("[calibrate] Done! Restart app.py to reload the homography.")
                break

        if cv2.getWindowProperty("SwishAI Calibration", cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()