import math
from pathlib import Path

import cv2
import numpy as np
from fastapi import APIRouter, HTTPException
from fastapi.responses import Response, FileResponse

router = APIRouter(prefix="/calibration", tags=["calibration"])

# ── FIBA constants ────────────────────────────────────────────────────────────
COURT_W  = 1500; HALF_H   = 1400
BASKET_X =  750; BASKET_Y =  157
R_3PT    =  675; PAINT_L  =  505; PAINT_R = 995; FT_Y = 575
CAM_DIST = 1400

val   = max(0.0, R_3PT**2 - BASKET_Y**2)
X3L   = BASKET_X - math.sqrt(val)
X3R   = BASKET_X + math.sqrt(val)
Y3TOP = BASKET_Y + R_3PT

REF = np.array([
    [PAINT_L,   0      ],  # 1
    [PAINT_R,   0      ],  # 2
    [PAINT_L,   FT_Y   ],  # 3
    [PAINT_R,   FT_Y   ],  # 4
    [0,         0      ],  # 5
    [COURT_W,   0      ],  # 6
    [X3L,       0      ],  # 7
    [X3R,       0      ],  # 8
    [BASKET_X,  Y3TOP  ],  # 9
    [0,         CAM_DIST], # 10
    [COURT_W,   CAM_DIST], # 11
], dtype=np.float32)

TRACKER_DIR = Path(__file__).parent / "tracker"
FRAME_PATH  = TRACKER_DIR / "calib_frame.jpg"
PREVIEW_PATH = TRACKER_DIR / "reprojection_preview.png"

WEIGHT_MAP = {0:8, 1:8, 2:8, 3:8, 4:4, 5:4, 6:2, 7:2, 8:2, 9:4, 10:4}


# ── Helpers ───────────────────────────────────────────────────────────────────

def _proj_court(pt, H):
    p = np.array([[pt]], dtype=np.float32)
    return cv2.perspectiveTransform(p, H)[0][0]


def _proj_img(cx, cy, H_inv):
    pt = np.array([[[float(cx), float(cy)]]], dtype=np.float32)
    px = cv2.perspectiveTransform(pt, H_inv)
    return (int(round(px[0][0][0])), int(round(px[0][0][1])))


def _compute_homography(src: np.ndarray):
    """Two-pass weighted homography matching calibrate.py exactly."""

    # Pass 1 — FT box only
    src1w = np.vstack([src[:4]] * 6)
    dst1w = np.vstack([REF[:4]] * 6)
    H1, _ = cv2.findHomography(src1w, dst1w, cv2.RANSAC, 5.0)
    if H1 is None:
        return None, None

    # Derive symmetrised sideline targets
    x_fl, y_fl = _proj_court(src[4],  H1)
    x_fr, y_fr = _proj_court(src[5],  H1)
    x_nl, y_nl = _proj_court(src[9],  H1)
    x_nr, y_nr = _proj_court(src[10], H1)

    x_left  = (x_fl + x_nl) / 2.0
    x_right = (x_fr + x_nr) / 2.0
    far_y   = float(np.clip((y_fl + y_fr) / 2.0, -FT_Y / 2, FT_Y / 2))
    near_y  = float(np.clip((y_nl + y_nr) / 2.0,  FT_Y + 50, HALF_H))

    derived = {
        4:  [x_left,  far_y ],
        5:  [x_right, far_y ],
        9:  [x_left,  near_y],
        10: [x_right, near_y],
    }

    # Pass 2 — all 11 pts weighted
    src_w_list, dst_w_list = [], []
    for i in range(11):
        dst_pt = derived[i] if i in derived else REF[i].tolist()
        for _ in range(WEIGHT_MAP.get(i, 1)):
            src_w_list.append(src[i])
            dst_w_list.append(dst_pt)

    H, _ = cv2.findHomography(
        np.array(src_w_list, dtype=np.float32),
        np.array(dst_w_list, dtype=np.float32),
        cv2.RANSAC, 5.0
    )
    return H, derived


def _draw_preview(frame: np.ndarray, H: np.ndarray, src: np.ndarray, derived: dict):
    H_inv = np.linalg.inv(H)

    def pi(cx, cy):
        return _proj_img(cx, cy, H_inv)

    def pc(pt):
        return _proj_court(pt, H)

    x_left  = (pc(src[4])[0]  + pc(src[9])[0])  / 2
    x_right = (pc(src[5])[0]  + pc(src[10])[0]) / 2

    # Grid
    for gy in range(0, HALF_H + 200, 200):
        cv2.line(frame, pi(x_left, gy), pi(x_right, gy), (50, 50, 50), 1)
    for i in range(9):
        gx = x_left + (x_right - x_left) * i / 8.0
        cv2.line(frame, pi(gx, 0), pi(gx, HALF_H), (50, 50, 50), 1)

    # Sidelines + baseline
    cv2.line(frame, pi(x_left,  0),     pi(x_left,  HALF_H), (0, 255, 255), 2)
    cv2.line(frame, pi(x_right, 0),     pi(x_right, HALF_H), (0, 255, 255), 2)
    cv2.line(frame, pi(x_left,  0),     pi(x_right, 0),      (0, 255, 255), 2)

    # FT box
    for (ax, ay), (bx, by) in [
        ((PAINT_L, 0),     (PAINT_R, 0)    ),
        ((PAINT_L, FT_Y),  (PAINT_R, FT_Y) ),
        ((PAINT_L, 0),     (PAINT_L, FT_Y) ),
        ((PAINT_R, 0),     (PAINT_R, FT_Y) ),
    ]:
        cv2.line(frame, pi(ax, ay), pi(bx, by), (255, 120, 0), 2)

    # Half-court line
    cv2.line(frame, pi(x_left, HALF_H), pi(x_right, HALF_H), (0, 220, 60), 3)

    # 3pt arc
    arc_pts = []
    for deg in range(-90, 91, 2):
        rad = math.radians(deg)
        cx2 = BASKET_X + R_3PT * math.sin(rad)
        cy2 = BASKET_Y + R_3PT * math.cos(rad)
        arc_pts.append(pi(cx2, cy2))
    for i in range(len(arc_pts) - 1):
        cv2.line(frame, arc_pts[i], arc_pts[i + 1], (0, 180, 255), 2)

    # Clicked points (green) + reprojected expected (blue) + error line
    for i, pt in enumerate(src):
        cv2.circle(frame, (int(pt[0]), int(pt[1])), 7, (0, 255, 0), -1)
        cv2.putText(frame, str(i + 1), (int(pt[0]) + 9, int(pt[1]) - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        dst_pt = derived[i] if i in derived else REF[i].tolist()
        rep = pi(dst_pt[0], dst_pt[1])
        cv2.circle(frame, rep, 5, (255, 0, 0), -1)
        cv2.line(frame, (int(pt[0]), int(pt[1])), rep, (0, 100, 255), 1)

    return frame


# ── Routes ────────────────────────────────────────────────────────────────────

@router.get("/frame/{file_id}")
def get_calibration_frame(file_id: str):
    from app import Config  # import here to avoid circular imports
    input_files = list(Config.UPLOAD_DIR.glob(f"{file_id}.*"))
    if not input_files:
        raise HTTPException(404, "Video not found.")

    cap = cv2.VideoCapture(str(input_files[0]))
    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = min(int(total * 0.10), total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, img = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(500, "Failed to extract frame.")

    if img.shape[1] > 1280:
        scale = 1280 / img.shape[1]
        img = cv2.resize(img, (1280, int(img.shape[0] * scale)))

    TRACKER_DIR.mkdir(exist_ok=True)
    cv2.imwrite(str(FRAME_PATH), img)

    _, buf = cv2.imencode(".jpg", img)
    return Response(content=buf.tobytes(), media_type="image/jpeg")


@router.post("/save")
async def save_calibration(data: dict):
    req = data.get("required_pts", [])
    if len(req) < 11:
        raise HTTPException(400, f"Need 11 points, got {len(req)}.")

    src = np.array(req[:11], dtype=np.float32)
    H, derived = _compute_homography(src)
    if H is None:
        raise HTTPException(500, "Homography computation failed. Check your points.")

    TRACKER_DIR.mkdir(exist_ok=True)
    np.save(str(TRACKER_DIR / "homography.npy"),       H)
    np.save(str(TRACKER_DIR / "homography_scale.npy"), np.array([1.0],          dtype=np.float32))
    np.save(str(TRACKER_DIR / "half_court_y.npy"),     np.array([float(HALF_H)], dtype=np.float32))

    # Reload in running app
    try:
        import app as _app
        _app.homography_matrix = H
        _app.homography_scale  = 1.0
    except Exception:
        pass

    # Draw preview on saved frame
    if FRAME_PATH.exists():
        frame = cv2.imread(str(FRAME_PATH))
        if frame is not None:
            frame = _draw_preview(frame, H, src, derived)
            cv2.imwrite(str(PREVIEW_PATH), frame)

    return {"status": "ok"}


@router.get("/preview")
def get_preview():
    if not PREVIEW_PATH.exists():
        raise HTTPException(404, "Preview not generated yet.")
    return FileResponse(str(PREVIEW_PATH), media_type="image/png")