"""
Usage: python generate_report.py processed/<file_id>_stats.csv
Outputs: same folder, same name .png

NOTE: shot_positions in the CSV must have a scored flag: (x,y,1) or (x,y,0)
If your CSV only has (x,y) pairs, all dots will be drawn as red (unknown).
"""
import sys, csv, re, cv2, numpy as np
from pathlib import Path

MINIMAP_PATH = Path(__file__).parent / "tracker" / "minimap.png"
COURT_W, COURT_H = 1500, 1400
F_LEFT, F_RIGHT, F_TOP, F_BOT = 0.005, 0.995, 0.005, 0.995

def court_to_px(x_cm, y_cm, mm_w, mm_h):
    cl = F_LEFT * mm_w;  cr = F_RIGHT * mm_w
    ct = F_TOP  * mm_h;  cb = F_BOT   * mm_h
    px = int(np.clip(cl + (x_cm / COURT_W) * (cr - cl), cl, cr))
    py = int(np.clip(ct + (y_cm / COURT_H) * (cb - ct), ct, cb))
    return px, py

def generate(csv_path: Path, player_id: str = None):
    minimap = cv2.imread(str(MINIMAP_PATH))
    if minimap is None:
        raise FileNotFoundError(f"minimap.png not found at {MINIMAP_PATH}")

    mm_h, mm_w = minimap.shape[:2]
    canvas = minimap.copy()
    total_shots = total_scored = 0

    with open(csv_path, newline="") as f:
        for row in csv.DictReader(f):
            if row["player_id"] != player_id:
                continue
            
            total_shots  += int(row["shots"])
            total_scored += int(row["baskets"])
            for m in re.finditer(r"\(([^)]+)\)", row["shot_positions"]):
                parts = m.group(1).split(",")
                x_cm, y_cm = float(parts[0]), float(parts[1])
                scored = bool(int(parts[2])) if len(parts) >= 3 else False
                px, py = court_to_px(x_cm, y_cm, mm_w, mm_h)
                color = (0, 200, 0) if scored else (0, 0, 220)  # green / red BGR
                cv2.circle(canvas, (px, py), 9, (0, 0, 0), -1)  # outline
                cv2.circle(canvas, (px, py), 7, color,     -1)

    bar = np.full((50, mm_w, 3), 30, dtype=np.uint8)
    label = f"Player id: {player_id}    Shots: {total_shots}    Scored: {total_scored}    Missed: {total_shots - total_scored}"
    (lw, _), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    cv2.putText(bar, label, ((mm_w - lw) // 2, 33), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    out = csv_path.with_suffix(".png")
    cv2.imwrite(str(out), np.vstack((canvas, bar)))
    print(f"✅ Saved → {out}")

if __name__ == "__main__":
    path = input("CSV path: ").strip()
    pid  = input("Player ID (leave blank for all): ").strip() or None
    generate(Path(path), pid)