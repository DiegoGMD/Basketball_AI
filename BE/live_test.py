# live_test.py  —  run: python live_test.py
import cv2
import time
from pathlib import Path
from collections import deque, defaultdict
from ultralytics import YOLO

MODEL_PATH   = Path("basketball_training/yolo26m_5classes/weights/best.pt")
TRACKER_PATH = Path("tracker/botsort.yaml")

THRESHOLDS = {0: 0.6, 1: 0.25, 2: 0.7, 3: 0.7, 4: 0.7}
COLORS     = {0: (0,165,255), 1: (0,215,255), 2: (0,255,0), 3: (0,0,255), 4: (255,100,0)}
CLASSES    = {0:"Ball", 1:"Ball in Basket", 2:"Player", 3:"Basket", 4:"Player Shooting"}

SHOT_COOLDOWN   = 1.0
BASKET_COOLDOWN = 1.0

# ── Model ──────────────────────────────────────────────────────────────────
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"❌ Model not found at: {MODEL_PATH.resolve()}")
if not TRACKER_PATH.exists():
    raise FileNotFoundError(f"❌ Tracker config not found at: {TRACKER_PATH.resolve()}")

print("🔄 Loading YOLO model...")
model = YOLO(str(MODEL_PATH))
print("✅ Model loaded.")

# ── Camera ─────────────────────────────────────────────────────────────────
CAMERA_INDEX = 0
print(f"🎥 Opening camera index {CAMERA_INDEX}...")
cap = cv2.VideoCapture(CAMERA_INDEX)

if not cap.isOpened():
    raise RuntimeError(
        f"❌ Could not open camera {CAMERA_INDEX}. "
        "Try changing CAMERA_INDEX to 1 or 2 at the top of the script."
    )

# Grab a test frame to confirm the camera actually streams
ok, test_frame = cap.read()
if not ok or test_frame is None:
    raise RuntimeError(
        f"❌ Camera {CAMERA_INDEX} opened but returned no frame. "
        "It may be in use by another app (Teams, Zoom, browser, etc.)."
    )

h, w = test_frame.shape[:2]
print(f"✅ Camera {CAMERA_INDEX} ready — {w}x{h}. Press Q to quit.")

# ── Main loop ──────────────────────────────────────────────────────────────
shots, baskets = 0, 0
last_shot = last_basket = 0.0
track_history = defaultdict(lambda: deque(maxlen=30))
frame_count = 0

conf_ranges = defaultdict(lambda: {"min": float("inf"), "max": float("-inf")})
while cap.isOpened():
    ok, frame = cap.read()
    if not ok:
        print("⚠️  Frame grab failed — camera disconnected?")
        break

    frame_count += 1
    now = time.time()

    try:
        results = model.track(
            frame,
            persist=True,
            tracker=str(TRACKER_PATH),
            verbose=False,
            conf=0.1,
        )
    except Exception as e:
        print(f"⚠️  Model inference error on frame {frame_count}: {e}")
        continue

    detections_this_frame = 0

    if results and results[0].boxes is not None:
        boxes   = results[0].boxes
        classes = boxes.cls.int().cpu().tolist()
        confs   = boxes.conf.cpu().tolist()
        ids     = boxes.id.int().cpu().tolist() if boxes.id is not None else [None]*len(classes)
        xyxy    = boxes.xyxy.cpu().numpy()
        xywh    = boxes.xywh.cpu().numpy()

        for cls, conf, tid, box_xyxy, box_xywh in zip(classes, confs, ids, xyxy, xywh):
            if conf < THRESHOLDS.get(cls, 0.3):
                continue

            conf_ranges[cls]["min"] = min(conf_ranges[cls]["min"], conf)
            conf_ranges[cls]["max"] = max(conf_ranges[cls]["max"], conf)

            detections_this_frame += 1
            x1, y1, x2, y2 = map(int, box_xyxy)
            cx, cy = float(box_xywh[0]), float(box_xywh[1])
            color  = COLORS[cls]
            label  = f"{CLASSES[cls]} {conf:.2f}" + (f" #{tid}" if tid else "")

            cv2.rectangle(frame, (x1,y1), (x2,y2), color, 2)
            cv2.putText(frame, label, (x1, y1-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            if tid is not None:
                track_history[tid].append((int(cx), int(cy)))
                pts = list(track_history[tid])
                for i in range(1, len(pts)):
                    cv2.line(frame, pts[i-1], pts[i], color, 2)

            if cls == 4 and now - last_shot > SHOT_COOLDOWN:
                shots += 1
                last_shot = now
                print(f"🏀 Shot detected! Total: {shots}")
            if cls == 1 and now - last_basket > BASKET_COOLDOWN:
                baskets += 1
                last_basket = now
                print(f"🎯 Basket! Total: {baskets}")

    # Log to terminal every 60 frames so you know it's alive
    if frame_count % 60 == 0:
        print(f"📷 Frame {frame_count} — detections this frame: {detections_this_frame} | Shots: {shots} | Baskets: {baskets}")

    acc = f"{(baskets/shots*100):.0f}%" if shots else "—"
    cv2.putText(frame, f"Shots: {shots}  Baskets: {baskets}  Acc: {acc}",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)

    cv2.imshow("Basketball AI — Live", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("👋 Quit by user.")
        break

print("\n📊 Confidence ranges per class:")
for cls, rng in sorted(conf_ranges.items()):
    print(f"  {CLASSES[cls]}: min={rng['min']:.2f}  max={rng['max']:.2f}")
cap.release()
cv2.destroyAllWindows()
print(f"✅ Done. Processed {frame_count} frames.")