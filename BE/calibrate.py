# =============================================================================
#   calibrate.py - B_AI Court Homography Calibration Tool
#
#   Designed for a camera placed approximately at the CENTRE of the HALF-COURT
#   LINE, pointing toward the basket end. The tool works for any camera height
#   (1-4 m) that can see the free-throw box.
#
#   COORDINATE SYSTEM - ONE HALF-COURT ONLY:
#       y = 0       → FAR BASELINE (basket end, top of the video frame)
#       y = 1400    → HALF-COURT LINE (near end, bottom of the video frame) [FIBA]
#       x = 0       → LEFT sideline   (left in video frame)
#       x = 1500    → RIGHT sideline  (right in video frame)                [FIBA]
#
#  SCALE REFERENCE:
#       The X axis is anchored by the SIDELINES (pts 5 & 6, the two far
#       baseline sideline corners).  All depth (Y) is a fixed FIBA ratio
#       of that width: HALF_H / COURT_W = 1400 / 1500.
#
#       The FREE-THROW BOX (pts 1-4) establishes the homography kernel and
#       is clicked FIRST because it is always fully visible.  The sideline
#       corners (pts 5-6) then lock the absolute width, from which the
#       complete half-court scale is derived.  Everything else improves
#       accuracy in the mid-court and near-camera regions.
#
#   ASCII - CAMERA POV (approximately centred on the half-court line):
#
#   ── QUICK START ────────────────────────────────────────────────────────────
#
#   OPTION A - Interactive wizard (for non-FIBA courts):
#      python calibrate.py --video uploads\sample.mp4 --setup
#
#   OPTION B - FIBA preset (default, no wizard needed):
#       python calibrate.py --video uploads\sample.mp4
#       python calibrate.py --video uploads\sample.mp4 --frame 120
#       python calibrate.py --image sample_frame.jpg
#
#   CONTROLS:
#       Left-click      → place next REQUIRED point
#       Right-click     → place/skip next OPTIONAL point (right-click same spot a 2nd time to SKIP)
#       U               → undo last point
#       R               → reset all points
#       S               → save + open reprojection preview
#       P               → open reprojection preview without saving
#       Q / ESC         → quit without saving
#
#   ZOOM / PAN:
#       Scroll wheel    → zoom in / out (centred on cursor)
#       Z / X           → zoom in / out (keyboard alternative)
#       0               → reset zoom & pan
#       Middle-drag     → pan while zoomed in
#       Arrow keys      → pan while zoomed in
#
# =============================================================================
#
#   Expected court diagram
#
#                          baseline
#    [5]───[7]───────────[1]───────[2]───────────[8]───[6]
#  L  |     |             |  ─[B]─  |             |     | R
#     |     |             |         |             |     |
#  s  |     |             | ······· |             |     | s
#  i  [10]  |             ··       ··             |  [11] i
#  d  |?    |            [3]───────[4]            |    ?| d
#  e  |?     ··           ··       ··           ··     ?| e
#  l  |??      ···          ·······          ···      ??| l
#  i  |??         ···                     ···         ??| i
#  n  |???           ·········[9]·········           ???| n
#  e  |????                                         ????| e
#     |???????                                   ???????|
#     |???????????????                   ???????????????| 
#     ────────────────────────[C]────────────────────────
#                          half-court
#
#   [C] camera location
#       (approximate location where the input camera sits)
#
#   [B] basket location
#       (approximate location where the basket being shot at sits)
#
#   [?] Area not visible
#       (All covered in the "?" sign is not visible by the camera)
#
# =============================================================================
#
#   REQUIRED - left-click all 11 in order:
#   1. Left  FT box x baseline      left paint line meets baseline          [FT anchor - primary scale]
#   2. Right FT box x baseline      right paint line meets baseline         [FT anchor - primary scale]
#   3. Left  FT line end            left end of the free-throw line         [FT anchor - primary scale]
#   4. Right FT line end            right end of the free-throw line        [FT anchor - primary scale]
#   5. Left  baseline corner        left sideline meets baseline            [width anchor - far-left]
#   6. Right baseline corner        right sideline meets baseline           [width anchor - far-right]
#   7. Left  3pt area x baseline    left 3pt line meets baseline            [inner-left on baseline]
#   8. Right 3pt area x baseline    right 3pt line meets baseline           [inner-right on baseline]
#   9. Top of 3pt arc               peak of 3pt arc, centred above basket   [mid-frame, centre]
#  10. Left  near sideline          left sideline visible near the camera   [angle anchor - near-left]
#  11. Right near sideline          right sideline visible near the camera  [angle anchor - near-right]
#
#  NOTE:
#   ★ FT box (pts 1-4) - click FIRST and PRECISELY.  These four corners establish the homography kernel.
#   ◆ Sideline corners (pts 5-6) - lock the far end of the court WIDTH.
#   ▲ Near sideline pts (10-11) - REQUIRED to fix the sideline ANGLE.
#       Without both ends of each sideline the homography cannot recover the
#       correct perspective tilt. Click where the sideline stripe is clearly
#       visible closest to the camera (bottom of the frame).
#
#   OPTIONAL (right-click to place, right-click same spot again to skip):
#  12. Left centre-circle edge      left edge of centre circle
#  13. Right centre-circle edge     right edge of centre circle
#  14. Half-court centre            midpoint of the half-court line
#
#   MINIMUM to save:   11 required
#   Full half-court:   11 - 14
#
# =============================================================================

import argparse
import sys
import math
from pathlib import Path

import cv2
import numpy as np


# ---------------------------------------------------------------------------
#   COURT STANDARD - edit ONE block, or use --setup for the interactive wizard.
#
#   All coordinates are for ONE HALF-COURT only:
#       y = 0           → FAR BASELINE (basket end)
#       y = HALF_H      → HALF-COURT LINE (camera end)
# ---------------------------------------------------------------------------

# FIBA (default)
COURT_W  = 1500     # court width, sideline to sideline (cm)
HALF_H   = 1400     # half-court depth: baseline → half-court line (cm)
BASKET_X =  750     # basket X from left sideline (cm)
BASKET_Y =  157     # basket Y from far baseline (cm)
R_3PT    =  675     # 3pt arc radius (cm)
PAINT_L  =  505     # left  paint line X (cm)
PAINT_R  =  995     # right paint line X (cm)
FT_Y     =  575     # free-throw line Y from baseline (cm)
CIRCLE_R =  180     # centre-circle radius (cm)

# Camera distance from the FAR BASELINE (cm).
# Used only for optional points 10/11 (sideline frame-cuts at the camera end).
# For a centred camera at the half-court line: ~1400 cm (FIBA).
# Adjust if your camera is closer to the basket.
CAMERA_DIST_FROM_BASELINE_CM = 1400 # = HALF_H for centred camera

# ---------------------------------------------------------------------------


def _build_reference_pts(court_w, half_h, basket_x, basket_y, r_3pt,
                          paint_l, paint_r, ft_y, circle_r, cam_dist):
    """
    Build the court-space reference arrays for all clickable points.

    Coordinate system:
      x : left sideline = 0,  right sideline = court_w
      y : far baseline  = 0,  half-court line = half_h

    Click order:
      1. FT box corners (pts 1-4) — always visible, establish the kernel.
      2. Far baseline sideline corners (pts 5-6) — lock court WIDTH.
      3. 3pt intersections + arc top (pts 7-9) — mid-court geometry.
      4. Near sideline points (pts 10-11) — fix the sideline ANGLE.
         Both ends of each sideline are required; without them the
         homography cannot recover the correct perspective tilt.

    Returns (required, opt_half) as np.float32 arrays.
    """
    val   = max(0.0, r_3pt ** 2 - basket_y ** 2)
    x3l   = basket_x - math.sqrt(val)
    x3r   = basket_x + math.sqrt(val)
    y3top = basket_y + r_3pt

    required = np.array([
        [float(paint_l),    0.0           ],    # 1  left  FT box x baseline
        [float(paint_r),    0.0           ],    # 2  right FT box x baseline
        [float(paint_l),    float(ft_y)   ],    # 3  left  FT line end
        [float(paint_r),    float(ft_y)   ],    # 4  right FT line end
        [0.0,               0.0           ],    # 5  left  baseline corner
        [float(court_w),    0.0           ],    # 6  right baseline corner
        [float(x3l),        0.0           ],    # 7  left  3pt x baseline
        [float(x3r),        0.0           ],    # 8  right 3pt x baseline
        [float(basket_x),   float(y3top)  ],    # 9  top of 3pt arc
        [0.0,               float(cam_dist)],   # 10 left  near sideline
        [float(court_w),    float(cam_dist)],   # 11 right near sideline
    ], dtype=np.float32)

    opt_half = np.array([
        [float(basket_x) - float(circle_r), float(half_h)], # 12 left centre-circle
        [float(basket_x) + float(circle_r), float(half_h)], # 13 right centre-circle
        [float(basket_x),                   float(half_h)], # 14 half-court centre
    ], dtype=np.float32)

    return required, opt_half


# ---------------------------------------------------------------------------
#   Labels
# ---------------------------------------------------------------------------
REQUIRED_LABELS = [
    " 1. LEFT  FT box x baseline    left paint line meets baseline              [FT anchor — click first, precise!]",
    " 2. RIGHT FT box x baseline    right paint line meets baseline             [FT anchor — click first, precise!]",
    " 3. LEFT  FT line end          left end of the free-throw line             [FT anchor — precise!]",
    " 4. RIGHT FT line end          right end of the free-throw line            [FT anchor — precise!]",
    " 5. LEFT  baseline corner      left sideline meets far baseline            [width anchor — far-left in image]",
    " 6. RIGHT baseline corner      right sideline meets far baseline           [width anchor — far-right in image]",
    " 7. LEFT  3pt area x baseline  left 3pt line meets baseline                [inner-left on baseline]",
    " 8. RIGHT 3pt area x baseline  right 3pt line meets baseline               [inner-right on baseline]",
    " 9. TOP of 3pt area            peak of the 3pt arc, centred above basket   [mid-frame, centre]",
    "10. LEFT  near sideline        left sideline stripe near the camera        [angle anchor — near-left]",
    "11. RIGHT near sideline        right sideline stripe near the camera       [angle anchor — near-right]",
]
OPT_HALF_LABELS = [
    "12. LEFT  centre-circle edge   left edge of the centre circle              [skip if off-camera]",
    "13. RIGHT centre-circle edge   right edge of the centre circle             [skip if off-camera]",
    "14. HALF-COURT centre          midpoint of the half-court line             [skip if off-camera]",
]

NUM_REQUIRED = 11
NUM_OPT_HALF = 3
NUM_OPTIONAL = NUM_OPT_HALF

HOMOGRAPHY_OUTPUT_PATH = Path(__file__).parent / "tracker" / "homography.npy"
HC_HOMOGRAPHY_OUTPUT_PATH = Path(__file__).parent / "tracker" / "half_court_y.npy"
PROJECTION_OUTPUT_PATH = Path(__file__).parent / "tracker" / "reprojection_preview.png"

# ---------------------------------------------------------------------------
#   Colours
# ---------------------------------------------------------------------------
C_REQ_DONE  = (0,   200, 255)
C_REQ_NEXT  = (0,   255,   0)
C_OPT_NEAR  = (255, 170,  70)   # orange
C_OPT_HALF  = (200, 160,  50)   # amber
C_TEXT      = (255, 255, 255)
H_REQ       = (180, 180,  50)
H_OPT_NEAR  = (230, 170,  60)
H_OPT_HALF  = (190, 150,  40)

# ---------------------------------------------------------------------------
#   Shared state
# ---------------------------------------------------------------------------
required_pts:       list              = []
optional_pts:       list              = []      # [x,y] or None per optional slot
_calib_scale:       float             = 1.0
base_img:           np.ndarray | None = None
display_img:        np.ndarray | None = None
REF_REQUIRED:       np.ndarray | None = None
REF_OPT_HALF:       np.ndarray | None = None
_pending_opt:       list       | None = None
COURT_BASKET_Y_CM:  float      | None = None
COURT_R_3PT_CM:     float      | None = None
_HALF_H:            float      = float(HALF_H)  # active half-court depth

# ---------------------------------------------------------------------------
#   Zoom / pan state    (display-only — all stored points are in original coords)
# ---------------------------------------------------------------------------
_zoom:          float   = 1.0
_pan_x:         float   = 0.0
_pan_y:         float   = 0.0
_ZOOM_MIN:      float   = 1.0
_ZOOM_MAX:      float   = 8.0
_ZOOM_STEP:     float   = 1.25
_mid_drag:      bool    = False
_drag_start:    list    = [0, 0]
_pan_start:     list    = [0.0, 0.0]


def _screen_to_img(sx: int, sy: int) -> tuple[int, int]:
    ix = int(sx / _zoom + _pan_x)
    iy = int(sy / _zoom + _pan_y)
    return ix, iy


def _img_to_screen(ix: float, iy: float) -> tuple[int, int]:
    sx = int((ix - _pan_x) * _zoom)
    sy = int((iy - _pan_y) * _zoom)
    return sx, sy


def _clamp_pan(img_h: int, img_w: int, win_h: int, win_w: int):
    """Clamp _pan_x/_pan_y so the viewport never scrolls past the image edges."""
    global _pan_x, _pan_y
    max_px = max(0.0, img_w - win_w / _zoom)
    max_py = max(0.0, img_h - win_h / _zoom)
    _pan_x = max(0.0, min(_pan_x, max_px))
    _pan_y = max(0.0, min(_pan_y, max_py))


def _apply_zoom(img: np.ndarray) -> np.ndarray:
    """Crop and resize the image to simulate zoom+pan. Returns a display-sized copy."""
    if _zoom == 1.0 and _pan_x == 0.0 and _pan_y == 0.0:
        return img
    h, w = img.shape[:2]
    vx1 = int(_pan_x);  vy1 = int(_pan_y)
    vx2 = min(w, int(_pan_x + w / _zoom))
    vy2 = min(h, int(_pan_y + h / _zoom))
    crop = img[vy1:vy2, vx1:vx2]
    if crop.size == 0:
        return img
    return cv2.resize(crop, (w, h), interpolation=cv2.INTER_LINEAR)


def _zoom_at(screen_x: int, screen_y: int, factor: float, win_w: int, win_h: int,
             img_w: int, img_h: int):
    """Zoom keeping the point at (screen_x, screen_y) fixed."""
    global _zoom, _pan_x, _pan_y
    ix = screen_x / _zoom + _pan_x
    iy = screen_y / _zoom + _pan_y
    new_zoom = max(_ZOOM_MIN, min(_ZOOM_MAX, _zoom * factor))
    _zoom  = new_zoom
    _pan_x = ix - screen_x / _zoom
    _pan_y = iy - screen_y / _zoom
    _clamp_pan(img_h, img_w, win_h, win_w)


def _reset_zoom():
    """Reset zoom and pan to origin."""
    global _zoom, _pan_x, _pan_y
    _zoom = 1.0;  _pan_x = 0.0;  _pan_y = 0.0


def _opt_info(flat_idx: int) -> tuple[str, tuple, np.ndarray | None]:
    """Return (label, colour, reference_court_pt) for optional point `flat_idx`."""
    return (OPT_HALF_LABELS[flat_idx], C_OPT_HALF,
            REF_OPT_HALF[flat_idx] if REF_OPT_HALF is not None else None)


def _hint_color(flat_idx: int) -> tuple:
    """Return the HUD hint colour for optional group containing `flat_idx`."""
    return C_OPT_HALF


def _group_name(flat_idx: int) -> str:
    """Return a human-readable group label for optional point `flat_idx`."""
    return "12/13/14 (half-court / centre-circle)"


# ---------------------------------------------------------------------------
#   Drawing helpers
# ---------------------------------------------------------------------------
def _draw_box_marker(img: np.ndarray, pt: tuple, color: tuple, size: int = 7):
    """Filled square with black border — used for sideline corner anchors."""
    x, y = pt
    cv2.rectangle(img, (x-size-2, y-size-2), (x+size+2, y+size+2), (0,0,0), -1)
    cv2.rectangle(img, (x-size,   y-size),   (x+size,   y+size),   color,   -1)


_BOX_REQ_IDX = {4, 5, 9, 10}   # far & near sideline corners → square marker


# ---------------------------------------------------------------------------
#   Redraw
# ---------------------------------------------------------------------------
def _redraw(img_base: np.ndarray) -> np.ndarray:
    """Composite all overlays (points, guide lines, HUD) onto a copy of the base image and return it."""
    img = img_base.copy()
    h, w = img.shape[:2]
    n_req = len(required_pts)
    n_opt = len(optional_pts)

    # Required points
    for i, pt in enumerate(required_pts):
        is_next = (i == n_req - 1) and n_req < NUM_REQUIRED
        c = C_REQ_NEXT if is_next else C_REQ_DONE
        if i in _BOX_REQ_IDX:
            _draw_box_marker(img, tuple(pt), c, size=7)
        else:
            cv2.circle(img, tuple(pt), 9, (0,0,0), -1)
            cv2.circle(img, tuple(pt), 7, c, -1)
        tag = REQUIRED_LABELS[i].split()[0].strip()
        cv2.putText(img, tag, (pt[0]+10, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 2)

    # Optional points
    for i, pt in enumerate(optional_pts):
        if pt is None:
            continue
        lbl, c, _ = _opt_info(i)
        cv2.circle(img, tuple(pt), 9, (0,0,0), -1)
        cv2.circle(img, tuple(pt), 7, c, -1)
        tag = lbl.split()[0].strip()
        cv2.putText(img, tag, (pt[0]+10, pt[1]-8), cv2.FONT_HERSHEY_SIMPLEX, 0.52, c, 2)

    # Live sideline angle preview: connect far corner ↔ near corner once both are placed
    if n_req > 9 and n_req >= 5:   # have both pt5 (far-left) and pt10 (near-left)
        cv2.line(img, tuple(required_pts[4]), tuple(required_pts[9]),  (255, 170, 70), 2)
    if n_req > 10 and n_req >= 6:  # have both pt6 (far-right) and pt11 (near-right)
        cv2.line(img, tuple(required_pts[5]), tuple(required_pts[10]), (255, 170, 70), 2)

    # Projected court guide lines once homography is available
    C_EXPECTED = (160, 160, 40)
    if len(required_pts) >= NUM_REQUIRED:
        _H, _inliers, _ = _compute_homography()
        if _H is not None:
            try:
                _Hi = np.linalg.inv(_H)
                # idx 5 = pt 6 (right sideline corner) = COURT_W
                court_w  = float(REF_REQUIRED[5][0])
                half_y   = _HALF_H
                # FT box coords: indices 0-3
                ft_y_ref = float(REF_REQUIRED[2][1])    # pt 3 y (FT line depth)
                pl       = float(REF_REQUIRED[0][0])    # pt 1 x (left  FT box)
                pr       = float(REF_REQUIRED[1][0])    # pt 2 x (right FT box)

                def _proj_line(cx1, cy1, cx2, cy2, colour, thickness=1):
                    pts_c = np.array([[[cx1, cy1]], [[cx2, cy2]]], dtype=np.float32)
                    pts_i = cv2.perspectiveTransform(pts_c, _Hi)
                    p1i = (int(round(pts_i[0][0][0])), int(round(pts_i[0][0][1])))
                    p2i = (int(round(pts_i[1][0][0])), int(round(pts_i[1][0][1])))
                    cv2.line(img, p1i, p2i, colour, thickness)

                # Derive sideline x from clicked pts projected through H
                def _ct(px_idx):
                    if px_idx >= len(required_pts): return None
                    p = np.array([[required_pts[px_idx]]], dtype=np.float32)
                    c = cv2.perspectiveTransform(p, _H)[0][0]
                    return float(c[0])
                xl_f = _ct(4); xl_n = _ct(9)
                xr_f = _ct(5); xr_n = _ct(10)
                xl = ((xl_f or 0.0) + (xl_n or 0.0)) / (2 if xl_f and xl_n else 1)
                xr = ((xr_f or court_w) + (xr_n or court_w)) / (2 if xr_f and xr_n else 1)

                # Baseline
                _proj_line(xl, 0.0, xr, 0.0, C_EXPECTED, 2)
                # Sidelines
                _proj_line(xl, 0.0, xl, half_y, C_EXPECTED, 2)
                _proj_line(xr, 0.0, xr, half_y, C_EXPECTED, 2)
                # Half-court line
                _proj_line(xl, half_y, xr, half_y, (80, 200, 80), 1)
                # FT box outline (orange)
                _proj_line(pl, 0.0,      pr, 0.0,      (255, 160, 40), 1)
                _proj_line(pl, ft_y_ref, pr, ft_y_ref, (255, 160, 40), 1)
                _proj_line(pl, 0.0,      pl, ft_y_ref, (255, 160, 40), 1)
                _proj_line(pr, 0.0,      pr, ft_y_ref, (255, 160, 40), 1)
            except Exception:
                pass

    img = _apply_zoom(img)

    # ── Overlays drawn AFTER zoom ─────────────────────────────────────────
    placed  = sum(1 for p in optional_pts if p is not None)
    skipped = sum(1 for p in optional_pts if p is None)
    total   = n_req + placed
    q = "EXCELLENT" if total >= 11 else ("GOOD" if total >= 8 else "MINIMUM (add more optional)")

    cv2.rectangle(img, (0, 0), (w, 26), (30,30,30), -1)
    status = (f"Req {n_req}/{NUM_REQUIRED}  Opt placed:{placed} skipped:{skipped}  "
              f"Total:{total} [{q}]   U=undo  R=reset  S=save  P=preview  Q=quit")
    cv2.putText(img, status, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.37, C_TEXT, 1)
    for lx, txt, c in [(w-200, "REQ", C_REQ_DONE), (w-130, "12-14=opt", C_OPT_HALF)]:
        cv2.putText(img, txt, (lx, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.35, c, 1)

    if _zoom > 1.0:
        zoom_txt = f"ZOOM  {_zoom:.1f}x   0=reset  scroll/Z/X=zoom  mid-drag/arrows=pan"
        (tw, th), _ = cv2.getTextSize(zoom_txt, cv2.FONT_HERSHEY_SIMPLEX, 0.38, 1)
        tx = w - tw - 10
        cv2.rectangle(img, (tx-4, 28), (w-2, 28+th+6), (0,0,0), -1)
        cv2.putText(img, zoom_txt, (tx, 28+th), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 220, 255), 1)

    all_req = n_req >= NUM_REQUIRED
    all_opt = n_opt >= NUM_OPTIONAL
    cv2.rectangle(img, (0, h-74), (w, h), (30,30,30), -1)

    if not all_req:
        idx = n_req
        cv2.putText(img, f"REQUIRED {idx+1}/{NUM_REQUIRED}  LEFT-CLICK:", (8, h-54),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, H_REQ, 1)
        cv2.putText(img, REQUIRED_LABELS[idx], (8, h-32),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.40, H_REQ, 1)
        if idx in (0, 1):
            tip = "TIP (★ FT ANCHOR): where the PAINT LINE meets the BASELINE — corners of the free-throw lane on the baseline"
        elif idx in (2, 3):
            tip = "TIP (★ FT ANCHOR): left then right ends of the FREE-THROW LINE (horizontal line at top of the paint)"
        elif idx in (4, 5):
            tip = "TIP (◆ WIDTH): far corners where the SIDELINE meets the BASELINE — these lock the court width scale"
        elif idx in (6, 7):
            tip = "TIP: where the 3pt arc MEETS the BASELINE — inner-left then inner-right, outside the FT lane corners"
        elif idx == 8:
            tip = "TIP: highest point of the 3pt arc, directly above the basket — visible as the arc crown"
        elif idx in (9, 10):
            tip = "TIP (▲ ANGLE): click the sideline stripe CLOSEST TO THE CAMERA — fixes the sideline angle (near-left then near-right)"
        else:
            tip = ""
        cv2.putText(img, tip, (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.37, (80,200,255), 1)

    elif not all_opt:
        idx = n_opt
        lbl, hint_c, _ = _opt_info(idx)
        grp = _group_name(idx)
        cv2.putText(img,
                    f"OPTIONAL  Group {grp}  {idx+1}/{NUM_OPTIONAL}  "
                    "RIGHT-CLICK (2nd right-click near same spot = SKIP):",
                    (8, h-54), cv2.FONT_HERSHEY_SIMPLEX, 0.37, hint_c, 1)
        cv2.putText(img, lbl, (8, h-32), cv2.FONT_HERSHEY_SIMPLEX, 0.40, hint_c, 1)
        tip = ("Points 12/13/14: half-court / centre-circle anchors — "
               "SKIP if half-court line is not visible")
        cv2.putText(img, tip, (8, h-10), cv2.FONT_HERSHEY_SIMPLEX, 0.37, hint_c, 1)

    else:
        cv2.rectangle(img, (0, h-74), (w, h), (20,60,20), -1)
        total2 = n_req + sum(1 for p in optional_pts if p is not None)
        qual = ("excellent" if total2 >= 11 else
                ("good"     if total2 >= 8  else "minimum — add more optionals"))
        cv2.putText(img,
                    f"All points done!  {total2} total ({qual})   S=SAVE & PREVIEW  U=undo  R=reset",
                    (8, h-42), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (100,255,100), 1)
    return img


# ---------------------------------------------------------------------------
#   Mouse callback
# ---------------------------------------------------------------------------
def _mouse_cb(event, x, y, flags, param):
    """
    OpenCV mouse callback: handles left-click (required pts), right-click (optional pts/skip),
    middle-drag (pan), and scroll-wheel (zoom).
    """

    global display_img, _pending_opt
    global _zoom, _pan_x, _pan_y, _mid_drag, _drag_start, _pan_start

    img_base = param
    h_img, w_img = img_base.shape[:2]
    h_win, w_win = (display_img.shape[:2] if display_img is not None
                    else (h_img, w_img))

    if event == cv2.EVENT_MBUTTONDOWN:
        _mid_drag   = True
        _drag_start = [x, y]
        _pan_start  = [_pan_x, _pan_y]
        return

    if event == cv2.EVENT_MOUSEMOVE and _mid_drag:
        dx = (x - _drag_start[0]) / _zoom
        dy = (y - _drag_start[1]) / _zoom
        _pan_x = _pan_start[0] - dx
        _pan_y = _pan_start[1] - dy
        _clamp_pan(h_img, w_img, h_win, w_win)
        display_img = _redraw(img_base)
        cv2.imshow("B_AI Calibration", display_img)
        return

    if event == cv2.EVENT_MBUTTONUP:
        _mid_drag = False
        return

    if event == cv2.EVENT_MOUSEWHEEL:
        factor = _ZOOM_STEP if flags > 0 else 1.0 / _ZOOM_STEP
        _zoom_at(x, y, factor, w_win, h_win, w_img, h_img)
        display_img = _redraw(img_base)
        cv2.imshow("B_AI Calibration", display_img)
        return

    ix, iy = _screen_to_img(x, y)
    ix = max(0, min(w_img-1, ix))
    iy = max(0, min(h_img-1, iy))

    n_req  = len(required_pts)
    n_opt  = len(optional_pts)
    all_req = n_req >= NUM_REQUIRED
    all_opt = n_opt >= NUM_OPTIONAL

    if event == cv2.EVENT_LBUTTONDOWN:
        if not all_req:
            required_pts.append([ix, iy])
            _pending_opt = None
            display_img = _redraw(img_base)
            cv2.imshow("B_AI Calibration", display_img)

    elif event == cv2.EVENT_RBUTTONDOWN:
        if all_req and not all_opt:
            if _pending_opt is None:
                _pending_opt = [ix, iy]
                ghost = _redraw(img_base)
                cv2.circle(ghost, (x, y), 8, (180,180,50), 2)
                cv2.putText(ghost, "right-click same spot again to SKIP",
                            (x+12, y-6), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (180,180,50), 1)
                cv2.imshow("B_AI Calibration", ghost)
            else:
                dist = math.hypot(ix - _pending_opt[0], iy - _pending_opt[1])
                threshold = 25 / _zoom
                lbl, _, _ = _opt_info(n_opt)
                if dist < threshold:
                    optional_pts.append(None)
                    print(f"[calibrate] {lbl.split()[0]} SKIPPED.")
                else:
                    optional_pts.append([ix, iy])
                    print(f"[calibrate] {lbl.split()[0]} placed at ({ix},{iy}).")
                _pending_opt = None
                display_img = _redraw(img_base)
                cv2.imshow("B_AI Calibration", display_img)


# ---------------------------------------------------------------------------
#   Undo
# ---------------------------------------------------------------------------
def _undo():
    """Remove the most recently placed point."""
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
#   Homography computation
# ---------------------------------------------------------------------------
def _collect_src_dst():
    """Collect all placed pixel points (src) and their court-space targets (dst) as float32 arrays."""
    src, dst = [], []
    for i, pt in enumerate(required_pts):
        src.append(pt);  dst.append(REF_REQUIRED[i])
    for i, pt in enumerate(optional_pts):
        if pt is not None:
            _, _, ref_row = _opt_info(i)
            src.append(pt);  dst.append(ref_row)
    return np.array(src, dtype=np.float32), np.array(dst, dtype=np.float32)


def _compute_homography():
    """
    Compute the pixel to court homography from all placed points.

    Two-pass approach:
      Pass 1 — FT box only (4 pts, 6× weighted) for a reliable initial scale.
      Pass 2 — Project sideline clicks through Pass-1 H to derive symmetrised
               court coordinates; final RANSAC fit over all 11+ points.

    Returns (H, inliers, mean_reproj_error_cm), or (None, 0, inf) on failure.
    """
    if len(required_pts) < NUM_REQUIRED:
        return None, 0, float('inf')

    court_w = float(REF_REQUIRED[5][0]) # COURT_W (right sideline x = 1500 cm)
    ft_y_cm = float(REF_REQUIRED[2][1]) # FT line y (575 cm)
    half_y  = _HALF_H                   # half-court depth (1400 cm)

    # ── PASS 1: FT box only → reliable scale / aspect / depth ────────────
    # The FT box 4 corners are the most reliably clickable points on the court
    # (no wide-angle distortion concern — they're near the centre of the frame).
    # 4 corners over-determine a homography; we use them weighted heavily.
    src1 = np.array(required_pts[:4], dtype=np.float32)
    dst1 = REF_REQUIRED[:4].copy()
    src1w = np.vstack([src1] * 6)   # 6× weight
    dst1w = np.vstack([dst1] * 6)
    H1, _ = cv2.findHomography(src1w, dst1w, cv2.RANSAC, 5.0)
    if H1 is None:
        return None, 0, float('inf')

    # ── PASS 2: project sideline clicks through H1 → actual court coords ──
    # H1 (from FT box) gives us the best available mapping from pixel → court.
    # Project the sideline clicks to find where they actually land in court space.
    # We do NOT force x=0/1500 — we trust the user's pixel clicks and let
    # the projection determine the court coordinates naturally.
    #
    # Symmetry constraints (keep lines parallel to baseline):
    #   Far left  (idx 4) and far  right (idx 5): share the same y (on baseline)
    #   Near left (idx 9) and near right (idx 10): share the same y
    #   Left sideline  (idx 4 & 9):  share the same x (x_left)
    #   Right sideline (idx 5 & 10): share the same x (x_right)

    def _proj_court(px_idx):
        px = np.array([[required_pts[px_idx]]], dtype=np.float32)
        ct = cv2.perspectiveTransform(px, H1)[0][0]
        return float(ct[0]), float(ct[1])

    x_fl, y_fl = _proj_court(4)     # far left
    x_fr, y_fr = _proj_court(5)     # far right
    x_nl, y_nl = _proj_court(9)     # near left
    x_nr, y_nr = _proj_court(10)    # near right

    # Symmetrise: each sideline gets one x; baseline and near line each get one y
    x_left  = (x_fl + x_nl) / 2.0   # average of far-left and near-left x
    x_right = (x_fr + x_nr) / 2.0   # average of far-right and near-right x
    far_y   = float(np.clip((y_fl + y_fr) / 2.0, -ft_y_cm / 2, ft_y_cm / 2))
    near_y  = float(np.clip((y_nl + y_nr) / 2.0,  ft_y_cm + 50, half_y))

    derived_sideline_dst = {
        4:  [x_left,   far_y ],     # far  left  corner
        5:  [x_right,  far_y ],     # far  right corner
        9:  [x_left,   near_y],     # near left
        10: [x_right,  near_y],     # near right
    }

    # ── Final fit: all 11 required + optional pts ─────────────────────────
    src_all = []
    dst_all = []
    for i in range(11):
        src_all.append(required_pts[i])
        if i in derived_sideline_dst:
            dst_all.append(derived_sideline_dst[i])
        else:
            dst_all.append(REF_REQUIRED[i].tolist())

    for i, pt in enumerate(optional_pts):
        if pt is not None:
            _, _, ref_row = _opt_info(i)
            src_all.append(pt)
            dst_all.append(ref_row.tolist())

    src_all = np.array(src_all, dtype=np.float32)
    dst_all = np.array(dst_all, dtype=np.float32)

    # Weights:
    #   FT box (idx 0-3)            - 8× : primary scale - most reliable clicks
    #   Far sideline (idx 4-5)      - 4× : far end of sideline
    #   3pt-baseline (idx 6-7)      - 2× : baseline geometry
    #   Arc top (idx 8)             - 2× : depth anchor
    #   Near sideline (idx 9-10)    - 4× : near end of sideline - fixes angle
    weight_map = {0:8, 1:8, 2:8, 3:8, 4:4, 5:4, 6:2, 7:2, 8:2, 9:4, 10:4}
    src_w_list = []
    dst_w_list = []
    for i, (s, d) in enumerate(zip(src_all, dst_all)):
        reps = weight_map.get(i, 1)
        for _ in range(reps):
            src_w_list.append(s)
            dst_w_list.append(d)

    src_w = np.array(src_w_list, dtype=np.float32)
    dst_w = np.array(dst_w_list, dtype=np.float32)

    H, mask = cv2.findHomography(src_w, dst_w, cv2.RANSAC, 5.0)
    if H is None:
        return None, 0, float('inf')

    n = len(src_all)
    mask_orig = mask[:n].ravel() if mask is not None else np.ones(n, dtype=np.uint8)
    inliers     = int(mask_orig.sum())
    src_in      = src_all[mask_orig == 1]
    dst_in      = dst_all[mask_orig == 1]
    proj        = cv2.perspectiveTransform(src_in.reshape(-1, 1, 2), H).reshape(-1, 2)
    errs        = np.linalg.norm(proj - dst_in, axis=1)
    reproj      = float(errs.mean()) if len(errs) > 0 else float('inf')
    return H, inliers, reproj

def _compute_and_save() -> bool:
    """Compute the homography and save all output files
    homography.npy, homography_scale.npy, half_court_y.npy). Returns True on success."""
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
        print("[calibrate] WARNING: high error — check FT anchor pts 1/2/3/4 first.")

    HOMOGRAPHY_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(HOMOGRAPHY_OUTPUT_PATH), H)
    print(f"[calibrate] Saved → {HOMOGRAPHY_OUTPUT_PATH}")

    scale_path = HOMOGRAPHY_OUTPUT_PATH.parent / "homography_scale.npy"
    np.save(str(scale_path), np.array([_calib_scale], dtype=np.float32))
    print(f"[calibrate] Scale saved → {scale_path}  ({_calib_scale:.4f})")

    # Save the half-court boundary so app.py can discard detections beyond it.
    # In the single-half-court system this is simply _HALF_H.
    half_y_path = HOMOGRAPHY_OUTPUT_PATH.parent / "half_court_y.npy"
    np.save(str(half_y_path), np.array([_HALF_H], dtype=np.float32))
    print(f"[calibrate] Half-court boundary saved → {half_y_path}  (y ≤ {_HALF_H:.0f} cm is ACTIVE)")
    return True


# ---------------------------------------------------------------------------
#   Half-court boundary helpers  (import in app.py / tracker)
# ---------------------------------------------------------------------------
_HALF_Y_PATH = Path(__file__).parent / "tracker" / "half_court_y.npy"


def is_in_active_half(court_x: float, court_y: float) -> bool:
    """Return True if (court_x, court_y) is within the calibrated half-court.

    The active zone is  0 ≤ y ≤ HALF_H  and  0 ≤ x ≤ COURT_W.
    Points outside this rectangle are in the numb zone (homography is
    unreliable there) and should be discarded by the tracker.
    """
    try:
        half_y = float(np.load(str(_HALF_Y_PATH))[0])
    except Exception:
        half_y = float(HALF_H)
    court_w = float(COURT_W)
    return (0.0 <= court_x <= court_w) and (0.0 <= court_y <= half_y)


def filter_to_active_half(court_points: np.ndarray) -> np.ndarray:
    """Filter an (N, 2) array of court-space points to those inside the active
    half-court.  Returns the filtered subset (may be empty).
    """
    try:
        half_y = float(np.load(str(_HALF_Y_PATH))[0])
    except Exception:
        half_y = float(HALF_H)
    court_w = float(COURT_W)
    if len(court_points) == 0:
        return court_points
    mask = (
        (court_points[:, 0] >= 0.0) &
        (court_points[:, 0] <= court_w) &
        (court_points[:, 1] >= 0.0) &
        (court_points[:, 1] <= half_y)
    )
    return court_points[mask]


# ---------------------------------------------------------------------------
#   Reprojection preview
# ---------------------------------------------------------------------------
def _draw_reprojection_preview(img_base: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Overlay court geometry and reprojection error dots onto `img_base`.

    Green = clicked pixel,
    Blue = expected (reprojected) pixel,
    Yellow = error in px.

    Near-sideline pts (10/11) show green only — their y was derived, so no ground-truth exists.
    """
    preview = img_base.copy()
    h_img, w_img = preview.shape[:2]
    H_inv = np.linalg.inv(H)

    # New point order: 0=FT-L-base, 1=FT-R-base, 2=FT-L-line, 3=FT-R-line,
    #                  4=sideline-L, 5=sideline-R, 6=3pt-L, 7=3pt-R, 8=arc-top
    paint_l  = float(REF_REQUIRED[0][0])    # pt 1 x (left  FT box on baseline)
    paint_r  = float(REF_REQUIRED[1][0])    # pt 2 x (right FT box on baseline)
    ft_y_ref = float(REF_REQUIRED[2][1])    # pt 3 y (FT line depth)
    court_w  = float(REF_REQUIRED[5][0])    # pt 6 x (right sideline) = COURT_W
    x3l      = float(REF_REQUIRED[6][0])    # pt 7 x (left  3pt-baseline)
    x3r      = float(REF_REQUIRED[7][0])    # pt 8 x (right 3pt-baseline)
    bx       = (x3l + x3r) / 2
    basket_y = COURT_BASKET_Y_CM if COURT_BASKET_Y_CM is not None else 157.0
    r_3pt    = COURT_R_3PT_CM    if COURT_R_3PT_CM    is not None else 675.0
    half_y   = _HALF_H

    def proj(cx, cy):
        pt = np.array([[[cx, cy]]], dtype=np.float32)
        px = cv2.perspectiveTransform(pt, H_inv)
        return (int(round(px[0][0][0])), int(round(px[0][0][1])))

    # Derive sideline x by projecting clicked sideline pixels through H.
    # This ensures every drawn line passes through the clicked points.
    def _px_to_court(px_idx):
        if px_idx >= len(required_pts):
            return None
        px = np.array([[required_pts[px_idx]]], dtype=np.float32)
        ct = cv2.perspectiveTransform(px, H)[0][0]
        return float(ct[0]), float(ct[1])

    sl = _px_to_court(4);  sr = _px_to_court(5)
    nl = _px_to_court(9);  nr = _px_to_court(10)
    x_left  = ((sl[0] if sl else 0.0)     + (nl[0] if nl else 0.0))     / 2.0
    x_right = ((sr[0] if sr else court_w) + (nr[0] if nr else court_w)) / 2.0

    # Grid every 200 cm (between the sidelines)
    for gy in range(0, int(half_y) + 200, 200):
        cv2.line(preview, proj(x_left, float(gy)), proj(x_right, float(gy)), (50,50,50), 1)
    for gx_frac in range(0, 9):   # 8 vertical divisions
        gx = x_left + (x_right - x_left) * gx_frac / 8.0
        cv2.line(preview, proj(gx, 0), proj(gx, half_y), (50,50,50), 1)

    # Sidelines + baseline
    cv2.line(preview, proj(x_left,  0),      proj(x_left,  half_y), (0,255,255), 2)
    cv2.line(preview, proj(x_right, 0),      proj(x_right, half_y), (0,255,255), 2)
    cv2.line(preview, proj(x_left,  0),      proj(x_right, 0),      (0,255,255), 2)

    # FT box
    cv2.line(preview, proj(paint_l, 0),        proj(paint_r, 0),        (255,120,0), 2)
    cv2.line(preview, proj(paint_l, ft_y_ref), proj(paint_r, ft_y_ref), (255,120,0), 2)
    cv2.line(preview, proj(paint_l, 0),        proj(paint_l, ft_y_ref), (255,120,0), 2)
    cv2.line(preview, proj(paint_r, 0),        proj(paint_r, ft_y_ref), (255,120,0), 2)

    # Half-court line — marks the boundary of the active zone
    hc_l = proj(x_left, half_y);  hc_r = proj(x_right, half_y)
    cv2.line(preview, hc_l, hc_r, (0, 220, 60), 3)
    mid_hc = ((hc_l[0]+hc_r[0])//2, (hc_l[1]+hc_r[1])//2)
    cv2.putText(preview, "HALF-COURT LINE (active zone ends here)",
                (mid_hc[0]-160, mid_hc[1]-8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0,220,60), 1)

    # 3pt arc
    arc_pts = []
    for deg in range(-90, 91, 2):
        rad = math.radians(deg)
        cx2 = bx + r_3pt * math.sin(rad)
        cy2 = basket_y + r_3pt * math.cos(rad)
        if 0 <= cy2 <= half_y + 200:
            arc_pts.append(proj(cx2, cy2))
    for i in range(len(arc_pts)-1):
        cv2.line(preview, arc_pts[i], arc_pts[i+1], (0,180,255), 2)

    # Reprojection error dots
    # For pts 1-9 and optionals: show green=click, blue=expected, yellow=error
    # For pts 10-11 (near sideline): only show green click — y was derived,
    # so there is no independent ground-truth to compare against.
    src_fixed  = np.array(required_pts[:9], dtype=np.float32)
    dst_fixed  = REF_REQUIRED[:9].copy()
    for i, pt in enumerate(optional_pts):
        if pt is not None:
            _, _, ref_row = _opt_info(i)
            src_fixed = np.vstack([src_fixed, np.array(pt, dtype=np.float32)])
            dst_fixed = np.vstack([dst_fixed, ref_row])

    proj_pts = cv2.perspectiveTransform(src_fixed.reshape(-1,1,2), H_inv).reshape(-1,2)
    _sideline_idx = {4, 5}

    for idx2, (orig, pp) in enumerate(zip(src_fixed, proj_pts)):
        ox, oy = int(orig[0]), int(orig[1])
        px2, py2 = int(pp[0]), int(pp[1])
        if idx2 in _sideline_idx:
            s = 6
            cv2.rectangle(preview, (ox-s,oy-s), (ox+s,oy+s), (0,255,0),  -1)
            cv2.rectangle(preview, (px2-s,py2-s),(px2+s,py2+s),(0,0,255), 2)
        else:
            cv2.circle(preview, (ox,oy),   6, (0,255,0), -1)
            cv2.circle(preview, (px2,py2), 6, (0,0,255), 2)
        lbl = str(idx2 + 1) if idx2 < 9 else str(idx2 + 3)   # 1-9, then 12-14
        cv2.putText(preview, lbl, (ox+8, oy-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1)

    # Near sideline pts 10 & 11 — draw only the clicked pixel (green square, no reprojection error)
    for idx2 in range(9, min(11, len(required_pts))):
        ox, oy = int(required_pts[idx2][0]), int(required_pts[idx2][1])
        s = 6
        cv2.rectangle(preview, (ox-s,oy-s), (ox+s,oy+s), (0,255,0), -1)
        lbl = "10" if idx2 == 9 else "11"
        cv2.putText(preview, lbl, (ox+8, oy-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255,255,0), 1)

    cv2.rectangle(preview, (0,0), (w_img,54), (20,20,20), -1)
    cv2.putText(preview, "REPROJECTION PREVIEW  green=click  blue=expected  yellow=error px",
                (8,18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200,200,200), 1)
    cv2.putText(preview,
                "Good: most errors < 10 px.  High error on pts 1-4 (FT anchors)? Re-click those first.",
                (8,40), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (200,200,200), 1)
    return preview


def _show_preview(img_base: np.ndarray):
    """Compute homography, draw the reprojection preview, save it, and show it in a window."""
    H, inliers, reproj = _compute_homography()
    if H is None:
        print(f"[calibrate] Cannot preview — need all {NUM_REQUIRED} required points.")
        return
    prev = _draw_reprojection_preview(img_base, H)
    PROJECTION_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(PROJECTION_OUTPUT_PATH), prev)
    print(f"[calibrate] Preview saved → {PROJECTION_OUTPUT_PATH}")
    wname = "Reprojection Preview (Q / ESC to close)"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.imshow(wname, prev)
    while True:
        k = cv2.waitKey(20) & 0xFF
        if k in (ord('q'), 27):
            break
        if cv2.getWindowProperty(wname, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyWindow(wname)


# ---------------------------------------------------------------------------
#   Court setup wizard
# ---------------------------------------------------------------------------
def _run_setup_wizard():
    """Interactive CLI wizard to enter court measurements (any standard or custom court).
    Returns (court_w, half_h, basket_x, basket_y, r_3pt, paint_l, paint_r, ft_y, circle_r, cam_dist).
    """
    print()
    print("=" * 70)
    print("  COURT SETUP WIZARD — enter measurements in cm")
    print("  Press ENTER to accept the [FIBA default] shown.")
    print("  All coordinates are for ONE HALF-COURT only.")
    print("=" * 70)

    def ask(prompt, default):
        try:
            raw = input(f"  {prompt} [{default}]: ").strip()
            return float(raw) if raw else float(default)
        except ValueError:
            print(f"  Invalid — using {default}.")
            return float(default)

    print("\n  ── COURT DIMENSIONS ─────────────────────────────────────────────")
    court_w = ask("Court WIDTH  (sideline to sideline) cm", 1500)
    half_h  = ask("HALF-court depth (baseline → half-court line) cm", 1400)

    print("\n  ── BASKET ───────────────────────────────────────────────────────")
    basket_x = ask("Basket X from LEFT sideline cm  (usually court_w/2)", court_w/2)
    basket_y = ask("Basket Y from FAR baseline cm   (centre of rim, FIBA=157)", 157)

    print("\n  ── THREE-POINT ARC ──────────────────────────────────────────────")
    r_3pt = ask("3pt arc RADIUS cm  (FIBA=675, NBA=723, NCAA=630)", 675)

    print("\n  ── PAINT / FREE-THROW LANE ──────────────────────────────────────")
    lane_w  = ask("Lane WIDTH cm  (FIBA=490, NBA/NCAA~488)", 490)
    paint_l = basket_x - lane_w/2;  paint_r = basket_x + lane_w/2
    print(f"     Paint: left={paint_l:.1f} cm  right={paint_r:.1f} cm")
    ft_y = ask("Free-throw line dist from FAR baseline cm  (FIBA=575)", 575)

    print("\n  ── CENTRE CIRCLE ────────────────────────────────────────────────")
    circle_r = ask("Centre-circle RADIUS cm  (standard=180)", 180)

    print("\n  ── CAMERA DISTANCE ──────────────────────────────────────────────")
    print("  Distance from the FAR BASELINE (basket end) to your camera.")
    print("  For a centred camera at the half-court line: ~= half-court depth.")
    print("  Used only for optional points 10/11 (sideline frame-cuts).")
    cam_dist = ask("Camera dist from baseline cm", half_h)

    print("\n  ── SUMMARY ──────────────────────────────────────────────────────")
    print(f"  Court   : {court_w:.0f} x {half_h:.0f} cm (half-court)")
    print(f"  Basket  : ({basket_x:.0f}, {basket_y:.0f}) cm")
    print(f"  3pt r   : {r_3pt:.0f} cm | Paint: {paint_l:.0f}-{paint_r:.0f} cm | FT: {ft_y:.0f} cm")
    print(f"  Cam dist: {cam_dist:.0f} cm from baseline")
    print("=" * 70)
    if input("  Proceed? (Y/n): ").strip().lower() == 'n':
        return _run_setup_wizard()
    return (court_w, half_h, basket_x, basket_y, r_3pt,
            paint_l, paint_r, ft_y, circle_r, cam_dist)


# ---------------------------------------------------------------------------
#   Video helper
# ---------------------------------------------------------------------------
def _get_frame(video_path: str, frame_num=None) -> np.ndarray:
    """Extract a single frame from `video_path`. Defaults to 10% through the video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"[calibrate] Cannot open: {video_path}")
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = frame_num if frame_num is not None else min(int(total * 0.10), total-1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, min(target, total-1))
    ret, frame = cap.read()
    cap.release()
    if not ret:
        sys.exit("[calibrate] Failed to read frame.")
    return frame


# ---------------------------------------------------------------------------
#   Main
# ---------------------------------------------------------------------------
def main():
    global base_img, display_img, REF_REQUIRED, REF_OPT_HALF
    global COURT_BASKET_Y_CM, COURT_R_3PT_CM, _HALF_H

    parser = argparse.ArgumentParser(description="B_AI Calibration — single half-court")
    grp = parser.add_mutually_exclusive_group(required=True)
    grp.add_argument("--video", help="Path to a basketball video file")
    grp.add_argument("--image", help="Path to a court image / screenshot")
    parser.add_argument("--frame", type=int, default=None,
                        help="Frame number to extract (default: 10%% through video)")
    parser.add_argument("--setup", action="store_true",
                        help="Run the court-measurement wizard (any court size/position).")
    args = parser.parse_args()

    if args.video:
        base_img = _get_frame(args.video, args.frame)
    else:
        base_img = cv2.imread(args.image)
        if base_img is None:
            sys.exit(f"[calibrate] Cannot read: {args.image}")

    h0, w0 = base_img.shape[:2]
    global _calib_scale
    if w0 > 1280:
        _calib_scale = 1280 / w0
        base_img = cv2.resize(base_img, (1280, int(h0 * _calib_scale)))

    if args.setup:
        (court_w, half_h, basket_x, basket_y, r_3pt,
         paint_l, paint_r, ft_y, circle_r, cam_dist) = _run_setup_wizard()
    else:
        court_w  = COURT_W
        half_h   = HALF_H
        basket_x = BASKET_X;  basket_y = BASKET_Y
        r_3pt    = R_3PT
        paint_l  = PAINT_L;   paint_r  = PAINT_R
        ft_y     = FT_Y
        circle_r = CIRCLE_R
        cam_dist = CAMERA_DIST_FROM_BASELINE_CM

    _HALF_H = float(half_h)

    REF_REQUIRED, REF_OPT_HALF = _build_reference_pts(
        court_w, half_h, basket_x, basket_y, r_3pt,
        paint_l, paint_r, ft_y, circle_r, cam_dist)
    COURT_BASKET_Y_CM = float(basket_y)
    COURT_R_3PT_CM    = float(r_3pt)

    # ── Console summary ───────────────────────────────────────────────────
    print("=" * 70)
    print("  B_AI Calibration - SINGLE HALF-COURT")
    print("=" * 70)
    print(f"  Court (half) : {court_w:.0f} wide x {half_h:.0f} cm deep")
    print(f"  Basket       : ({basket_x:.0f}, {basket_y:.0f}) cm  |  3pt r={r_3pt:.0f} cm")
    print(f"  FT box       : x {paint_l:.0f}-{paint_r:.0f} cm,  y 0-{ft_y:.0f} cm")
    print(f"  Camera dist  : {cam_dist:.0f} cm from baseline")
    print()
    print("  REQUIRED (left-click all 11, in order):")
    print("  - pts  1-4  (FT box): click FIRST and most precisely")
    print("  - pts  5-6  (sidelines, far): lock court WIDTH scale")
    print("  - pts  7-9  (3pt arc): improve mid-court accuracy")
    print("  - pts 10-11 (sidelines, near): fix sideline ANGLE - critical!")
    for lbl, pt in zip(REQUIRED_LABELS, REF_REQUIRED):
        print(f"    {lbl}  →  ({pt[0]:.0f}, {pt[1]:.0f}) cm")
    print()
    print("  OPTIONAL 12-14 - half-court / centre-circle (skip if off-camera):")
    for lbl, pt in zip(OPT_HALF_LABELS, REF_OPT_HALF):
        print(f"    {lbl}  →  ({pt[0]:.0f}, {pt[1]:.0f}) cm")
    print()
    print("  TIPS:")
    print("   Click FT anchors (pts 1-4) FIRST and as precisely as possible")
    print("   Far sideline corners (pts 5-6) lock court WIDTH")
    print("   Near sideline pts (10-11) fix the sideline ANGLE - without these")
    print("   the sidelines drawn by the homography will fan in the wrong direction")
    print("   After saving, the sidelines in the preview should align with the court stripes")
    print("=" * 70)

    display_img = _redraw(base_img)
    wname = "B_AI Calibration"
    cv2.namedWindow(wname, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(wname, _mouse_cb, base_img)
    cv2.imshow(wname, display_img)

    h_img, w_img = base_img.shape[:2]

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
            required_pts.clear();  optional_pts.clear();  _pending_opt = None
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

        # Zoom keys
        elif key in (ord('z'), ord('Z')):
            _, _, ww, wh = cv2.getWindowImageRect(wname)
            cx, cy = (ww or w_img)//2, (wh or h_img)//2
            _zoom_at(cx, cy, _ZOOM_STEP, ww or w_img, wh or h_img, w_img, h_img)
            display_img = _redraw(base_img)
            cv2.imshow(wname, display_img)
        elif key in (ord('x'), ord('X')):
            _, _, ww, wh = cv2.getWindowImageRect(wname)
            cx, cy = (ww or w_img)//2, (wh or h_img)//2
            _zoom_at(cx, cy, 1.0/_ZOOM_STEP, ww or w_img, wh or h_img, w_img, h_img)
            display_img = _redraw(base_img)
            cv2.imshow(wname, display_img)
        elif key == ord('0'):
            _reset_zoom()
            display_img = _redraw(base_img)
            cv2.imshow(wname, display_img)

        # Arrow-key pan (Linux)
        elif key in (81, 83, 82, 84):
            step = max(20, int(50 / _zoom))
            _, _, ww, wh = cv2.getWindowImageRect(wname)
            if key == 81:   _pan_x -= step
            elif key == 83: _pan_x += step
            elif key == 82: _pan_y -= step
            elif key == 84: _pan_y += step
            _clamp_pan(h_img, w_img, wh or h_img, ww or w_img)
            display_img = _redraw(base_img)
            cv2.imshow(wname, display_img)
        # Windows arrow keys
        elif key == 0:
            ext = cv2.waitKey(0) & 0xFF
            step = max(20, int(50 / _zoom))
            _, _, ww, wh = cv2.getWindowImageRect(wname)
            if ext == 75:   _pan_x -= step
            elif ext == 77: _pan_x += step
            elif ext == 72: _pan_y -= step
            elif ext == 80: _pan_y += step
            _clamp_pan(h_img, w_img, wh or h_img, ww or w_img)
            display_img = _redraw(base_img)
            cv2.imshow(wname, display_img)

        if cv2.getWindowProperty(wname, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()