# =============================================================================
#  calibrate.py  —  SwishAI Court Homography Calibration Tool
#
#  Works for ANY camera position — close to the basket, mid-court, or at the
#  half-court line.  Height 0.5-2 m, capturing ONE HALF of the court.
#
#      BOTTOM of frame = baseline (basket end)     — always the FAR end
#      TOP    of frame = closest visible court line — changes with cam position
#
#  ── QUICK START ──────────────────────────────────────────────────────────────
#
#  OPTION A — Interactive wizard (recommended — works for ANY court/position):
#      python calibrate.py --video uploads\sample.mp4 --setup
#
#  OPTION B — Manual preset (FIBA / NBA / NCAA configured below):
#      python calibrate.py --video uploads\sample.mp4
#      python calibrate.py --video uploads\sample.mp4 --frame 120
#      python calibrate.py --image sample_frame.jpg
#
#  CONTROLS:
#      Left-click        → place next REQUIRED point
#      Right-click       → place/skip next OPTIONAL point
#                          (right-click same spot a 2nd time to SKIP)
#      U                 → undo last point
#      R                 → reset all points
#      S                 → save + open reprojection preview
#      P                 → open reprojection preview without saving
#      Q / ESC           → quit without saving
#
# =============================================================================
#   old
#                          baseline
#    [1]───[3]──────[5]─────────────────[6]──────[4]───[2]
#  L  |     |        |    |  ─[B]─  |    |        |     | R
#     |     |        |    |         |    |        |     |
#  s  |     |        |    | ······· |    |        |     | s
#  i  |     |        |    ··       ··    |        |     | i
#  d  |     |        ────[7]───────[8]────        |     | d
#  e  |      ··           ··       ··           ··      | e
#  l  |        ···          ·······          ···        | l
#  i  |           ···                     ···           | i
#  n [10]            ·········[9]·········           [11] n
#  e  |?                                               ?| e
#     |????                                         ????|
#     |???????????????                   ???????????????| 
#     ────────────────────────[C]────────────────────────
#
#   new
#                          baseline
#    [1]───[3]───────────[5]───────[6]───────────[4]───[2]
#  L  |     |             |  ─[B]─  |             |     | R
#     |     |             |         |             |     |
#  s  |     |             | ······· |             |     | s
#  i  |     |             ··       ··             |     | i
#  d  |     |            [7]───────[8]            |     | d
#  e  |      ··           ··       ··           ··      | e
#  l  |        ···          ·······          ···        | l
#  i  |           ···                     ···           | i
#  n [10]            ·········[9]·········           [11] n
#  e  |?                                               ?| e
#     |????                                         ????|
#     |???????????????                   ???????????????| 
#     ────────────────────────[C]────────────────────────
#
#     [C] camera location
#         (approximate location where the input camera sits)
#
#     [B] basket location
#         (approximate location where the basket being shot at sits)
#
#     ───────────────────────────────────────────────────
#
#  REQUIRED (left-click all 9):
#   1. Right baseline corner        (left sideline x baseline)          [FAR-RIGHT]
#   2. Left baseline corner         (right sideline x baseline)         [FAR-LEFT]
#   3. Right 3pt arc x baseline     (left 3pt-baseline intersection)    [FAR, inner-right]
#   4. Left 3pt arc x baseline      (right 3pt-baseline intersection)   [FAR, inner-left]
#   5. Right free-throw box x base  (left paint line x baseline)        [FAR, inner-right]
#   6. Left free-throw box x base   (right paint line x baseline)       [FAR, inner-left]
#   7. Right free-throw line end    (left edge of FT line)              [MID-RIGHT]
#   8. Left free-throw line end     (right edge of FT line)             [MID-LEFT]
#   9. Top of 3pt arc               peak of the arc, furthest from base [CLOSE, centre]
#
#  OPTIONAL (right-click to place, right-click same spot again to skip):
#  10. Right sideline frame-cut     where frame cuts the left sideline  [CLOSE, right]
#  11. Left sideline frame-cut      where frame cuts the right sideline [CLOSE, left]
#  12. Left centre-circle edge      left edge of centre circle
#  13. Right centre-circle edge     right edge of centre circle
#  14. Half-court centre            midpoint of the half-court line
#
#  MINIMUM to save:   9 required
#  Better mid-court:  1..11
#  Half-court cam:    1..14
#
# =============================================================================

import argparse
import sys
import math
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
#  CONFIGURE BEFORE RUNNING  (or use --setup for the interactive wizard)
# ---------------------------------------------------------------------------

# Distance from the FAR BASELINE (basket end) to your camera in cm.
# Measure physically: walk from the baseline to the camera and measure.
#   1-4 m from basket    → 100-400
#   5-8 m (mid-court)   → 500-800
#   At half-court line  → 1400 (FIBA)
# Used only for optional legacy points (3, 4). If you skip those, the
# exact value here does not affect accuracy.
CAMERA_DIST_FROM_BASELINE_CM = 1200

# ---------------------------------------------------------------------------
#  COURT STANDARD — uncomment ONE block.  Use --setup for non-standard courts.
#
#  COURT_H = full court length (both halves).  Half-court depth = COURT_H/2.
# ---------------------------------------------------------------------------

# FIBA
COURT_W  = 1500;  COURT_H  = 2800
BASKET_X =  750;  BASKET_Y =  157
R_3PT    =  675;  PAINT_L  =  505;  PAINT_R  =  995
FT_Y     =  575;  CIRCLE_R =  180;  HALF_Y   = COURT_H // 2

# ---------------------------------------------------------------------------


def _build_reference_pts(court_w, court_h, basket_x, basket_y, r_3pt,
                          paint_l, paint_r, ft_y, circle_r, half_y, cam_dist):
    """
    Returns (required, opt_near, opt_half) as np.float32 arrays.
      required  9pts: 1..9
      opt_near  2pts: 10 11
      opt_half  3pts: 12 13 14
    """
    val    = max(0.0, r_3pt**2 - basket_y**2)
    x3l    = basket_x - math.sqrt(val)
    x3r    = basket_x + math.sqrt(val)
    y3top  = basket_y + r_3pt
    y_mid3 = basket_y + r_3pt * 0.5
    dx_m3  = math.sqrt(max(0.0, r_3pt**2 - (y_mid3 - basket_y)**2))

    required = np.array([
        [0.0,            0.0       ],   # 1  right baseline corner in image
        [float(court_w), 0.0       ],   # 2  left baseline corner in image
        [float(x3l),     0.0       ],   # 3  right 3pt-baseline point in image
        [float(x3r),     0.0       ],   # 4  left 3pt-baseline point in image
        [float(paint_l), 0.0       ],   # 5  right paint-baseline point in image
        [float(paint_r), 0.0       ],   # 6  left paint-baseline point in image
        [float(paint_l), float(ft_y)],  # 7  right FT end in image
        [float(paint_r), float(ft_y)],  # 8  left FT end in image
        [float(basket_x),float(y3top)], # 9  top of 3pt arc
    ], dtype=np.float32)

    opt_near = np.array([
        [0.0,                                float(cam_dist)],  # 10 right frame-cut in image
        [float(court_w),                     float(cam_dist)],  # 11 left frame-cut in image
    ], dtype=np.float32)

    opt_half = np.array([
        [float(basket_x) - float(circle_r),  float(half_y)  ],  # 12
        [float(basket_x) + float(circle_r),  float(half_y)  ],  # 13
        [float(basket_x),                    float(half_y)  ],  # 14
    ], dtype=np.float32)

    return required, opt_near, opt_half


# ---------------------------------------------------------------------------
#  Labels
# ---------------------------------------------------------------------------
REQUIRED_LABELS = [
    " 1. RIGHT baseline corner      left sideline x baseline                 [far-right in image]",
    " 2. LEFT  baseline corner      right sideline x baseline                [far-left in image]",
    " 3. RIGHT 3pt arc x baseline   left 3pt line meets baseline             [inner-right in image]",
    " 4. LEFT  3pt arc x baseline   right 3pt line meets baseline            [inner-left in image]",
    " 5. RIGHT FT box x baseline    left paint line meets baseline           [inner-right in image]",
    " 6. LEFT  FT box x baseline    right paint line meets baseline          [inner-left in image]",
    " 7. RIGHT FT line end          left edge of free-throw line             [mid-right in image]",
    " 8. LEFT  FT line end          right edge of free-throw line            [mid-left in image]",
    " 9. TOP of 3pt arc             peak of the arc, furthest from baseline  [top-centre]",
]
OPT_NEAR_LABELS = [
    "10. RIGHT sideline frame-cut   where frame cuts the left sideline       [skip if off-camera]",
    "11. LEFT  sideline frame-cut   where frame cuts the right sideline      [skip if off-camera]",
]
OPT_HALF_LABELS = [
    "12. LEFT  centre-circle edge   left edge of centre circle                [skip if near basket]",
    "13. RIGHT centre-circle edge   right edge of centre circle               [skip if near basket]",
    "14. HALF-COURT centre          midpoint of the half-court line           [skip if near basket]",
]

NUM_REQUIRED = 9
NUM_OPT_NEAR = 2
NUM_OPT_HALF = 3
NUM_OPTIONAL = NUM_OPT_NEAR + NUM_OPT_HALF   # 5

OUTPUT_PATH = Path(__file__).parent / "tracker" / "homography.npy"

# ---------------------------------------------------------------------------
#  Colours
# ---------------------------------------------------------------------------
C_REQ_DONE = (0,   200, 255)
C_REQ_NEXT = (0,   255,   0)
C_OPT_NEAR = (255, 170,  70)   # orange
C_OPT_HALF = (200, 160,  50)   # amber
C_TEXT     = (255, 255, 255)
H_REQ      = (180, 180,  50)
H_OPT_NEAR = (230, 170,  60)
H_OPT_HALF = (190, 150,  40)

# ---------------------------------------------------------------------------
#  Shared state
# ---------------------------------------------------------------------------
required_pts: list              = []
optional_pts: list              = []   # [x,y] or None per optional slot
base_img:     np.ndarray | None = None
display_img:  np.ndarray | None = None
REF_REQUIRED: np.ndarray | None = None
REF_OPT_NEAR: np.ndarray | None = None
REF_OPT_HALF: np.ndarray | None = None
_pending_opt: list | None       = None
COURT_BASKET_Y_CM: float | None = None
COURT_R_3PT_CM:    float | None = None


def _opt_info(flat_idx: int) -> tuple[str, tuple, np.ndarray | None]:
    """Return (label, dot_color, ref_row) for a flat optional index."""
    if flat_idx < NUM_OPT_NEAR:
        return OPT_NEAR_LABELS[flat_idx], C_OPT_NEAR, (REF_OPT_NEAR[flat_idx] if REF_OPT_NEAR is not None else None)
    else:
        i = flat_idx - NUM_OPT_NEAR
        return OPT_HALF_LABELS[i], C_OPT_HALF, (REF_OPT_HALF[i] if REF_OPT_HALF is not None else None)


def _hint_color(flat_idx: int) -> tuple:
    if flat_idx < NUM_OPT_NEAR: return H_OPT_NEAR
    return H_OPT_HALF


def _group_name(flat_idx: int) -> str:
    if flat_idx < NUM_OPT_NEAR: return "10/11 (frame-cut)"
    return "12/13/14 (half-court)"


# ---------------------------------------------------------------------------
#  Drawing
# ---------------------------------------------------------------------------
def _redraw(img_base: np.ndarray) -> np.ndarray:
    img = img_base.copy()
    h, w = img.shape[:2]
    n_req = len(required_pts)
    n_opt = len(optional_pts)

    # Required points
    for i, pt in enumerate(required_pts):
        is_next = (i == n_req - 1) and n_req < NUM_REQUIRED
        c = C_REQ_NEXT if is_next else C_REQ_DONE
        cv2.circle(img, tuple(pt), 9, (0,0,0), -1)
        cv2.circle(img, tuple(pt), 7, c, -1)
        tag = REQUIRED_LABELS[i].split()[0].strip()
        cv2.putText(img, tag, (pt[0]+10, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 2)

    # Optional points
    for i, pt in enumerate(optional_pts):
        if pt is None: continue
        lbl, c, _ = _opt_info(i)
        cv2.circle(img, tuple(pt), 9, (0,0,0), -1)
        cv2.circle(img, tuple(pt), 7, c, -1)
        tag = lbl.split()[0].strip()
        cv2.putText(img, tag, (pt[0]+10, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 2)

    # Top status bar
    placed = sum(1 for p in optional_pts if p is not None)
    skipped = sum(1 for p in optional_pts if p is None)
    total = n_req + placed
    q = "EXCELLENT" if total >= 11 else ("GOOD" if total >= 8 else "MINIMUM (add more optional)")
    cv2.rectangle(img, (0, 0), (w, 26), (30,30,30), -1)
    status = (f"Req {n_req}/{NUM_REQUIRED}  Opt placed:{placed} skipped:{skipped}  "
              f"Total:{total} [{q}]   U=undo  R=reset  S=save  P=preview  Q=quit")
    cv2.putText(img, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.37, C_TEXT, 1)
    # colour legend
    for lx, txt, c in [(w-320, "REQ", C_REQ_DONE), (w-260, "10/11=cut", C_OPT_NEAR),
                        (w-150, "12+=half", C_OPT_HALF)]:
        cv2.putText(img, txt, (lx, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

    # Bottom instruction bar (3 lines)
    all_req = n_req >= NUM_REQUIRED
    all_opt = n_opt >= NUM_OPTIONAL
    cv2.rectangle(img, (0, h-74), (w, h), (30,30,30), -1)

    if not all_req:
        idx = n_req
        cv2.putText(img, f"REQUIRED {idx+1}/{NUM_REQUIRED}  LEFT-CLICK:", (8, h-54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, H_REQ, 1)
        cv2.putText(img, REQUIRED_LABELS[idx], (8, h-32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, H_REQ, 1)
        if idx in (2, 3):
            tip = "TIP: where the 3pt arc line INTERSECTS the BASELINE — inner-left / inner-right of the key"
        elif idx in (4, 5):
            tip = "TIP: where the PAINT / FREE-THROW BOX meets the BASELINE — sharp corners inside the 3pt intersections"
        elif idx in (6, 7):
            tip = "TIP: left/right ends of the FREE-THROW LINE — horizontal line at the top of the paint box"
        elif idx == 8:
            tip = "TIP: the highest visible point of the 3pt arc, centred above the basket"
        else:
            tip = "TIP: the far corners where the sideline meets the baseline — mirrored in the image vs court coordinates"
        cv2.putText(img, tip, (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (80,200,255), 1)

    elif not all_opt:
        idx = n_opt
        lbl, hint_c, _ = _opt_info(idx)
        grp = _group_name(idx)
        cv2.putText(img, f"OPTIONAL  Group {grp}  {idx+1}/{NUM_OPTIONAL}  RIGHT-CLICK (2nd right-click near same spot = SKIP):",
                    (8, h-54), cv2.FONT_HERSHEY_SIMPLEX, 0.37, hint_c, 1)
        cv2.putText(img, lbl, (8, h-32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, hint_c, 1)
        # Group-specific tips
        if idx < NUM_OPT_NEAR:
            tip = "Points 10/11: use the sideline frame-cuts if visible, or SKIP if the near sideline corners are outside the frame"
        else:
            tip = "Points 12/13/14: half-court / centre-circle anchors — SKIP if the half-court line is outside the frame"
        cv2.putText(img, tip, (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.37, hint_c, 1)

    else:
        cv2.rectangle(img, (0, h-74), (w, h), (20,60,20), -1)
        total2 = n_req + sum(1 for p in optional_pts if p is not None)
        qual = "excellent" if total2 >= 11 else ("good" if total2 >= 8 else "minimum — add more optionals")
        cv2.putText(img, f"All points done!  {total2} total ({qual})   S=SAVE & PREVIEW  U=undo  R=reset",
                    (8, h-42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100,255,100), 1)
    return img


# ---------------------------------------------------------------------------
#  Mouse callback
# ---------------------------------------------------------------------------
def _mouse_cb(event, x, y, flags, param):
    global display_img, _pending_opt
    n_req = len(required_pts)
    n_opt = len(optional_pts)
    all_req = n_req >= NUM_REQUIRED
    all_opt = n_opt >= NUM_OPTIONAL

    if event == cv2.EVENT_LBUTTONDOWN:
        if not all_req:
            required_pts.append([x, y])
            _pending_opt = None
            display_img = _redraw(param)
            cv2.imshow("SwishAI Calibration", display_img)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if all_req and not all_opt:
            if _pending_opt is None:
                _pending_opt = [x, y]
                ghost = _redraw(param)
                cv2.circle(ghost, (x, y), 8, (180,180,50), 2)
                cv2.putText(ghost, "right-click same spot again to SKIP",
                            (x+12, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,50), 1)
                cv2.imshow("SwishAI Calibration", ghost)
            else:
                dist = math.hypot(x - _pending_opt[0], y - _pending_opt[1])
                lbl, _, _ = _opt_info(n_opt)
                if dist < 25:
                    optional_pts.append(None)
                    print(f"[calibrate] {lbl.split()[0]} SKIPPED.")
                else:
                    optional_pts.append([x, y])
                    print(f"[calibrate] {lbl.split()[0]} placed at ({x},{y}).")
                _pending_opt = None
                display_img = _redraw(param)
                cv2.imshow("SwishAI Calibration", display_img)


# ---------------------------------------------------------------------------
#  Undo
# ---------------------------------------------------------------------------
def _undo():
    global _pending_opt
    _pending_opt = None
    if optional_pts:
        removed = optional_pts.pop()
        lbl, _, _ = _opt_info(len(optional_pts))
        status = "skipped" if removed is None else f"at {removed}"
        print(f"[calibrate] Undo — {lbl.split()[0]} ({status}) removed.")
    elif required_pts:
        removed = required_pts.pop()
        print(f"[calibrate] Undo — req pt {len(required_pts)+1} at {removed} removed.")


# ---------------------------------------------------------------------------
#  Homography
# ---------------------------------------------------------------------------
def _collect_src_dst():
    """Merge all clicked points (required + placed optionals) into parallel arrays."""
    src, dst = [], []
    for i, pt in enumerate(required_pts):
        src.append(pt); dst.append(REF_REQUIRED[i])
    for i, pt in enumerate(optional_pts):
        if pt is not None:
            _, _, ref_row = _opt_info(i)
            src.append(pt); dst.append(ref_row)
    return np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)


def _compute_homography():
    if len(required_pts) < NUM_REQUIRED:
        return None, 0, float('inf')
    src, dst = _collect_src_dst()
    total_placed = len(src)
    if total_placed < NUM_REQUIRED:
        print(f"[calibrate] Only {total_placed} points — all {NUM_REQUIRED} required points are needed before saving.")
        return None, 0, float('inf')
    H, mask = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)
    if H is None:
        return None, 0, float('inf')
    inliers = int(mask.sum()) if mask is not None else 0
    src_in = src[mask.ravel() == 1] if mask is not None else src
    dst_in = dst[mask.ravel() == 1] if mask is not None else dst
    proj   = cv2.perspectiveTransform(src_in.reshape(-1,1,2), H).reshape(-1,2)
    errs   = np.linalg.norm(proj - dst_in, axis=1)
    reproj = float(errs.mean()) if len(errs) > 0 else float('inf')
    return H, inliers, reproj


def _compute_and_save() -> bool:
    placed = sum(1 for p in optional_pts if p is not None)
    total  = len(required_pts) + placed
    if total < NUM_REQUIRED:
        print(f"[calibrate] Need all {NUM_REQUIRED} required points ({total} total so far).")
        return False
    H, inliers, reproj = _compute_homography()
    if H is None:
        return False
    print(f"[calibrate] Homography from {total} pts | inliers={inliers} | reproj={reproj:.1f} cm")
    if inliers < 6:
        print("[calibrate] WARNING: low inliers — re-click intersections more precisely.")
    if reproj > 50:
        print("[calibrate] WARNING: high error — check pts 1/2, 3/4, and 5/6 for exact baseline intersections.")
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_PATH), H)
    print(f"[calibrate] Saved → {OUTPUT_PATH}")
    return True


# ---------------------------------------------------------------------------
#  Reprojection preview
# ---------------------------------------------------------------------------
def _draw_reprojection_preview(img_base: np.ndarray, H: np.ndarray) -> np.ndarray:
    preview = img_base.copy()
    h_img, w_img = preview.shape[:2]
    H_inv = np.linalg.inv(H)

    # Pull key measurements from REF arrays
    court_w = float(REF_REQUIRED[1][0])           # pt 2 x
    paint_l = float(REF_REQUIRED[4][0])           # pt 5 x
    paint_r = float(REF_REQUIRED[5][0])           # pt 6 x
    ft_y    = float(REF_REQUIRED[6][1])           # pt 7 y
    x3l     = float(REF_REQUIRED[2][0])           # pt 3 x
    x3r     = float(REF_REQUIRED[3][0])           # pt 4 x
    bx      = (x3l + x3r) / 2                     # basket x approx
    basket_y = max(0.0, math.sqrt(max(0.0, COURT_R_3PT_CM**2 - (bx - x3l)**2))) if COURT_R_3PT_CM else 157.0
    basket_y = COURT_BASKET_Y_CM if COURT_BASKET_Y_CM is not None else basket_y
    r_3pt   = COURT_R_3PT_CM if COURT_R_3PT_CM is not None else math.hypot(bx - x3l, basket_y)

    # Preview depth: prefer the near frame-cut depth from optional pts 10/11.
    cam_y = float(REF_OPT_NEAR[0][1]) if REF_OPT_NEAR is not None else ft_y * 2.5

    def proj(cx, cy):
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        px = cv2.perspectiveTransform(pt, H_inv)
        return (int(round(px[0][0][0])), int(round(px[0][0][1])))

    # Grid every 200 cm up to visible depth
    for gy in range(0, int(cam_y)+200, 200):
        cv2.line(preview, proj(0, float(gy)), proj(court_w, float(gy)), (50,50,50), 1)
    for gx in range(0, int(court_w)+200, 200):
        cv2.line(preview, proj(float(gx), 0), proj(float(gx), cam_y), (50,50,50), 1)

    # Sidelines + baseline
    for cx in [0.0, court_w]:
        cv2.line(preview, proj(cx, 0), proj(cx, cam_y), (0,255,255), 2)
    cv2.line(preview, proj(0,0), proj(court_w, 0), (0,255,255), 2)

    # Paint
    cv2.line(preview, proj(paint_l, 0), proj(paint_l, ft_y), (255,120,0), 2)
    cv2.line(preview, proj(paint_r, 0), proj(paint_r, ft_y), (255,120,0), 2)
    cv2.line(preview, proj(paint_l, ft_y), proj(paint_r, ft_y), (255,120,0), 2)

    # 3pt arc
    arc_pts = []
    for deg in range(-90, 91, 2):
        rad = math.radians(deg)
        cx2 = bx + r_3pt * math.sin(rad)
        cy2 = basket_y + r_3pt * math.cos(rad)
        if 0 <= cy2 <= cam_y + 200:
            arc_pts.append(proj(cx2, cy2))
    for i in range(len(arc_pts)-1):
        cv2.line(preview, arc_pts[i], arc_pts[i+1], (0,180,255), 2)

    # Reprojection error dots
    src, dst = _collect_src_dst()
    proj_pts = cv2.perspectiveTransform(src.reshape(-1,1,2), H_inv).reshape(-1,2)
    for orig, pp in zip(src, proj_pts):
        ox,oy = int(orig[0]), int(orig[1])
        px,py = int(pp[0]),   int(pp[1])
        cv2.circle(preview, (ox,oy), 6, (0,255,0), -1)
        cv2.circle(preview, (px,py), 6, (0,0,255), 2)
        err = math.hypot(ox-px, oy-py)
        cv2.putText(preview, f"{err:.0f}px", (ox+8,oy-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1)

    cv2.rectangle(preview, (0,0), (w_img, 54), (20,20,20), -1)
    cv2.putText(preview, "REPROJECTION PREVIEW  green=click  blue=expected  yellow=error px",
                (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1)
    cv2.putText(preview, "Good: most errors < 10 px.  High error on 1/2 or 5/6? Re-click those baseline corners.",
                (8,40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
    return preview


def _show_preview(img_base: np.ndarray):
    H, inliers, reproj = _compute_homography()
    if H is None:
        print(f"[calibrate] Cannot preview — not enough points (need all {NUM_REQUIRED} required points).")
        return
    prev  = _draw_reprojection_preview(img_base, H)
    wname = "Reprojection Preview (Q / ESC to close)"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.imshow(wname, prev)
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('q'), 27): break
        if cv2.getWindowProperty(wname, cv2.WND_PROP_VISIBLE) < 1: break
    cv2.destroyWindow(wname)


# ---------------------------------------------------------------------------
#  Court setup wizard
# ---------------------------------------------------------------------------
def _run_setup_wizard():
    print()
    print("=" * 70)
    print("  COURT SETUP WIZARD — enter measurements in cm")
    print("  Press ENTER to accept the [FIBA default] shown.")
    print("=" * 70)

    def ask(prompt, default):
        try:
            raw = input(f"  {prompt} [{default}]: ").strip()
            return float(raw) if raw else float(default)
        except ValueError:
            print(f"  Invalid, using {default}.")
            return float(default)

    print("\n  ── COURT DIMENSIONS ─────────────────────────────────────────")
    court_w = ask("Court WIDTH  (sideline to sideline) cm", 1500)
    court_h = ask("Court LENGTH (full court, baseline to baseline) cm", 2800)
    half_y  = court_h / 2
    print(f"     Half-court depth = {half_y:.0f} cm")

    print("\n  ── BASKET ───────────────────────────────────────────────────")
    basket_x = ask("Basket X from LEFT sideline cm  (usually court_w/2)", court_w/2)
    basket_y = ask("Basket Y from FAR baseline cm   (centre of rim)", 157)

    print("\n  ── THREE-POINT ARC ──────────────────────────────────────────")
    r_3pt = ask("3pt arc RADIUS cm  (FIBA=675, NBA=723, NCAA=630)", 675)

    print("\n  ── PAINT / FREE-THROW LANE ──────────────────────────────────")
    lane_w  = ask("Lane WIDTH cm  (FIBA~490, NBA/NCAA~488)", 490)
    paint_l = basket_x - lane_w/2;  paint_r = basket_x + lane_w/2
    print(f"     Paint: left={paint_l:.1f} cm  right={paint_r:.1f} cm")
    ft_y    = ask("Free-throw line dist from FAR baseline cm  (FIBA=575)", 575)

    print("\n  ── CENTRE CIRCLE ────────────────────────────────────────────")
    circle_r = ask("Centre-circle RADIUS cm  (standard=180)", 180)

    print("\n  ── CAMERA DISTANCE ──────────────────────────────────────────")
    print("  Measure from the FAR BASELINE (basket end) to your camera.")
    print("  Examples:  1-4 m from basket → 100-400 cm")
    print("             Mid-court (5-8 m) → 500-800 cm")
    print("             At half-court     → 1400 cm (FIBA)")
    print("  This is only used for optional points 10/11 (sideline frame-cuts).")
    print("  If you plan to skip those, any value here is fine.")
    cam_dist = ask("Camera dist from baseline cm", half_y)

    print("\n  ── SUMMARY ──────────────────────────────────────────────────")
    print(f"  Court    : {court_w:.0f} x {court_h:.0f} cm")
    print(f"  Basket   : ({basket_x:.0f}, {basket_y:.0f}) cm")
    print(f"  3pt r    : {r_3pt:.0f} cm | Paint: {paint_l:.0f}-{paint_r:.0f} cm | FT: {ft_y:.0f} cm")
    print(f"  Cam dist : {cam_dist:.0f} cm from baseline")
    print("=" * 70)
    if input("  Proceed? (Y/n): ").strip().lower() == 'n':
        return _run_setup_wizard()
    return (court_w, court_h, basket_x, basket_y, r_3pt,
            paint_l, paint_r, ft_y, circle_r, half_y, cam_dist)


# ---------------------------------------------------------------------------
#  Video helper
# ---------------------------------------------------------------------------
def _get_frame(video_path: str, frame_num=None) -> np.ndarray:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[calibrate] Cannot open: {video_path}")
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = frame_num if frame_num is not None else min(int(total*0.10), total-1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(target, total-1))
    ret, frame = cap.read()
    cap.release()
    if not ret: sys.exit("[calibrate] Failed to read frame.")
    return frame


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------
def main():
    global base_img, display_img, REF_REQUIRED, REF_OPT_NEAR, REF_OPT_HALF
    global COURT_BASKET_Y_CM, COURT_R_3PT_CM

    parser = argparse.ArgumentParser(description="SwishAI Calibration")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video", help="Path to a basketball video file")
    grp.add_argument("--image", help="Path to a court image / screenshot")
    parser.add_argument("--frame", type=int, default=None)
    parser.add_argument("--setup", action="store_true",
                        help="Run the court-measurement wizard (any court size/position).")
    args = parser.parse_args()

    if args.video:
        base_img = _get_frame(args.video, args.frame)
    else:
        base_img = cv2.imread(args.image)
        if base_img is None: sys.exit(f"[calibrate] Cannot read: {args.image}")

    h0, w0 = base_img.shape[:2]
    if w0 > 1280:
        scale    = 1280/w0
        base_img = cv2.resize(base_img, (1280, int(h0*scale)))

    if args.setup:
        (court_w, court_h, basket_x, basket_y, r_3pt,
         paint_l, paint_r, ft_y, circle_r, half_y, cam_dist) = _run_setup_wizard()
    else:
        court_w, court_h   = COURT_W, COURT_H
        basket_x, basket_y = BASKET_X, BASKET_Y
        r_3pt              = R_3PT
        paint_l, paint_r   = PAINT_L, PAINT_R
        ft_y               = FT_Y
        circle_r           = CIRCLE_R
        half_y             = HALF_Y
        cam_dist           = CAMERA_DIST_FROM_BASELINE_CM

    REF_REQUIRED, REF_OPT_NEAR, REF_OPT_HALF = _build_reference_pts(
        court_w, court_h, basket_x, basket_y, r_3pt,
        paint_l, paint_r, ft_y, circle_r, half_y, cam_dist)
    COURT_BASKET_Y_CM = float(basket_y)
    COURT_R_3PT_CM = float(r_3pt)

    # Console summary
    print("=" * 70)
    print("  SwishAI Calibration")
    print("=" * 70)
    print(f"  Court  : {court_w:.0f} x {court_h:.0f} cm | Basket: ({basket_x:.0f}, {basket_y:.0f}) cm")
    print(f"  3pt r  : {r_3pt:.0f} cm | Cam dist: {cam_dist:.0f} cm from baseline")
    print()
    print("  REQUIRED (left-click all 9):")
    for lbl, pt in zip(REQUIRED_LABELS, REF_REQUIRED):
        print(f"    {lbl}  ({pt[0]:.0f}, {pt[1]:.0f}) cm")
    print()
    print("  OPTIONAL POINTS 10 & 11 — sideline frame-cuts (skip if off-camera):")
    for lbl, pt in zip(OPT_NEAR_LABELS, REF_OPT_NEAR):
        print(f"    {lbl}  ({pt[0]:.0f}, {pt[1]:.0f}) cm")
    print()
    print("  OPTIONAL GROUP C — half-court / centre-circle (skip if half-court is off-camera):")
    for lbl, pt in zip(OPT_HALF_LABELS, REF_OPT_HALF):
        print(f"    {lbl}  ({pt[0]:.0f}, {pt[1]:.0f}) cm")
    print()
    print("  TIPS:")
    print("   Base setup   → click all required points 1..9")
    print("   Mid-court    → add optional points 10 and 11 if the sideline frame-cuts are visible")
    print("   Close to rim → points 10/11 and 12/13/14 can be skipped if they are outside the frame")
    print("   Half-court   → add points 12/13/14 for maximum precision")
    print("   Save needs all 9 required points. Optional points improve stability.")
    print("   After saving, check the preview: errors < 10 px = good calibration.")
    print("=" * 70)

    display_img = _redraw(base_img)
    wname = "SwishAI Calibration"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(wname, _mouse_cb, base_img)
    cv2.imshow(wname, display_img)

    while True:
        key = cv2.waitKey(20) & 0xFF
        if key in (ord('q'), 27):
            print("[calibrate] Quit without saving.")
            break
        elif key == ord('u'):
            _undo()
            display_img = _redraw(base_img)
            cv2.imshow(wname, display_img)
        elif key == ord('r'):
            global _pending_opt
            required_pts.clear(); optional_pts.clear(); _pending_opt = None
            print("[calibrate] Reset.")
            display_img = _redraw(base_img)
            cv2.imshow(wname, display_img)
        elif key == ord('p'):
            _show_preview(base_img)
        elif key == ord('s'):
            if _compute_and_save():
                print("[calibrate] Opening reprojection preview...")
                _show_preview(base_img)
                print("[calibrate] Done. Restart app.py to use the new homography.")
                break
        if cv2.getWindowProperty(wname, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
