# Basketball AI Tracker

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![es](https://img.shields.io/badge/lang-es-red.svg)](README.es.md)

> [!NOTE] This is repository a slight modification of sPappalard's one named [SwishAI](https://github.com/sPappalard/SwishAI/tree/master)

A comprehensive computer vision system for real-time basketball training analysis and player tracking. This project combines deep learning object detection with video processing to automatically identify players, basketballs, and court elements, providing detailed training statistics.

## Overview

Basketball AI Tracker uses YOLO (You Only Look Once) neural networks to detect and track basketball players and balls in training videos. The system processes video frames in real-time, generates annotated output videos with tracking overlays, produces a basketball court minimap visualization, and exports detailed training statistics in CSV format.

**Key Capabilities:**
- Real-time detection of players, basketballs, and court elements
- Multi-player tracking across frames
- Automatic shot and basket counting
- Live training statistics (per-player breakdown)
- Court minimap visualization with homography transformation
- Video output with HUD (heads-up display) and performance effects
- Batch processing with progress tracking
- Automatic cleanup of processed files

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Application](#running-the-application)
- [Usage](#usage)
  - [Processing a Basketball Video](#processing-a-basketball-video)
  - [API Endpoints](#api-endpoints)
- [Project Structure](#project-structure)
- [Technology Stack](#technology-stack)
- [Architecture](#architecture)
  - [Model Architecture](#model-architecture)
  - [Pipeline Architecture](#pipeline-architecture)
  - [Technical Details](#technical-details)
- [Training & Performance](#training--performance)
  - [Training a Custom Model](#training-a-custom-model)
  - [Model Performance](#model-performance)
  - [Improving Model Performance](#improving-model-performance)
- [Configuration](#configuration)
- [Performance Optimization](#performance-optimization)
- [Utility Scripts](#utility-scripts)
- [Development](#development)
- [Troubleshooting](#troubleshooting)
- [Output Files](#output-files)
- [Limitations & Future Enhancements](#limitations--future-enhancements)
- [CORS Configuration](#cors-configuration)
- [License](#license)
- [Credits](#credits)


## Quick Start

### Prerequisites

- **Windows OS** (or Linux with minor modifications to setup script)
- **Python 3.8+** (available in system PATH)
- **Node.js 16+** (for frontend development)
- **Optional:** NVIDIA GPU with CUDA 11.8+ (recommended for faster inference)

### Installation

#### 1. Backend Setup (Windows)

Run the automated environment setup script:

```bash
install_env.bat
```

This script will:
- Create a Python virtual environment
- Detect your NVIDIA GPU and CUDA version
- Install PyTorch with appropriate GPU support (or CPU if no GPU found)
- Install all required dependencies
- Verify the setup

**Automatic CUDA Detection:**
The script intelligently selects the correct PyTorch build:
- CUDA 12.8+ → `cu128` (latest support)
- CUDA 12.4-12.7 → `cu124`
- CUDA 12.1-12.3 → `cu121`
- CUDA 11.8 → `cu118`
- No GPU → CPU-only PyTorch

**Manual Setup (if needed):**

```bash
cd BE
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Frontend Setup

```bash
cd FE
npm install
```

### Running the Application

#### Backend (API Server)

```bash
cd BE
.\venv\Scripts\activate           # Activate virtual environment
python app.py
```

The FastAPI server will start at `http://localhost:8000`
- API documentation: `http://localhost:8000/docs` (Swagger UI)
- Alternative docs: `http://localhost:8000/redoc` (ReDoc)

#### Frontend (Development Server)

```bash
cd FE
npm run dev
```

The Vite development server will start at `http://localhost:5173`

## Usage

### Processing a Basketball Video

1. **Open the Web Interface**
   - Navigate to `http://localhost:5173` in your browser

2. **Upload Video**
   - Select a basketball game or practice video
   - The system generates a unique file ID for tracking
   - Video is stored in `BE/uploads/` directory

3. **Configure Detection (Optional)**
   - Set custom confidence thresholds for different object classes
   - Choose tracking algorithm (ByteTrack or BoTSORT)
   - Default thresholds are optimized for most scenarios

4. **Start Processing**
   - Trigger the video processing pipeline
   - The system runs YOLO inference on all frames
   - Player and ball coordinates are tracked across frames
   - Statistics are computed in real-time

5. **Monitor Progress**
   - Poll the status endpoint for real-time progress updates
   - Live statistics update as frames are processed
   - View per-player performance metrics

6. **Download Results**
   - Download the processed video with tracking overlays and HUD
   - Download the statistics CSV report
   - Both files are packaged in a ZIP for convenience

**Processing Pipeline:**
- Video frames are extracted and resized to 640×640
- YOLO detects 5 classes: Ball, Ball in Basket, Player, Basket, Player Shooting
- MinimapRenderer projects detections onto a 2D court visualization
- HUD displays real-time statistics: shot count and basket count
- Output video includes visual effects highlighting detections

### API Endpoints

**Video Processing:**
- `POST /upload` - Upload a video file
- `POST /process/{file_id}` - Start background processing with custom parameters
- `GET /status/{file_id}` - Get processing progress and live statistics
- `GET /download-zip/{file_id}` - Download processed video + CSV report as zip

**Model Management:**
- `GET /models` - List available detection models
- `GET /health` - System health check

## Project Structure

```
Basketball_AI/
├── BE/                                 # Backend (Python)
│   ├── app.py                          # Main FastAPI application
│   ├── calibrate.py                    # Image calibration utilities
│   ├── metrics.py                      # Training metrics and analysis
│   ├── player_report.py                # Player report utilities
│   ├── requirements.txt                # Python dependencies
│   ├── train_model.py                  # YOLO model training script
│   ├── yolo.pt                      # your YOLO base model weights
│   ├── basketball-detection-srfkd-1/   # Dataset folder
│   │   ├── test/
│   │   ├── train/
│   │   └── valid/
│   ├── basketball_training/            # Trained models directory
│   │   ├── yolo26m_5classes/
│   │   └── yolo26s_5classes/
│   ├── metrics_reports/                # Performance reports
│   │   ├── overfitting_analysis.png
│   │   ├── performance_report.json
│   │   └── training_curves.png
│   ├── tracker/                        # Tracking configs and court images
│   │   ├── botsort.yaml
│   │   ├── bytetrack.yaml
│   │   ├── court.excalidraw
│   │   ├── half_court_y.npy
│   │   ├── homography.npy
│   │   ├── homography_scale.npy
│   │   ├── minimap.png
│   │   └── reprojection_preview.png
│   ├── uploads/                        # User-uploaded videos (auto-created)
│   ├── processed/                      # Processed output videos (auto-created)
│   ├── runs/                           # Processed output videos (auto-created)
│   └── venv/                           # Python virtual environment
│
├── FE/                                 # Frontend (React)
│   ├── package.json                    # Node dependencies
│   ├── package-lock.json               # npm lockfile
│   ├── vite.config.js                  # Vite configuration
│   ├── tailwind.config.js              # Tailwind CSS configuration
│   ├── postcss.config.js               # PostCSS configuration
│   ├── eslint.config.js                # ESLint configuration
│   ├── index.html                      # SPA entry HTML
│   ├── README.md                       # Frontend README
│   ├── public/                         # Public static files
│   └── src/
│       ├── App.jsx                    # Main React component
│       ├── App.css                    # Application styles
│       ├── main.jsx                   # React entry point
│       ├── index.css                  # Global styles
│       └── assets/                    # Static assets
│
├── .gitignore                         # Ignored files
├── install_env.bat                    # Windows environment setup script
├── LICENSE                            # Project licensing
├── README.md                          # Original documentation
├── README_og.md                       # Original README backup
└── README.es.md                       # Spanish documentation
```

## Technology Stack

### Backend
- **Framework:** FastAPI 0.115.0 with Uvicorn
- **Computer Vision:** YOLOv26 (via Ultralytics), OpenCV
- **Deep Learning:** PyTorch 2.5.1 with CUDA support
- **Data Processing:** NumPy, Pandas, scikit-learn
- **Visualization:** Matplotlib, Seaborn
- **Utilities:** Roboflow (dataset management), python-dotenv, PyYAML

### Frontend
- **Framework:** React 19.1.1 with Vite
- **Styling:** Tailwind CSS with PostCSS/Autoprefixer
- **UI Components:** Lucide React (icons), React QR Code
- **Linting:** ESLint

## Architecture

### Model Architecture

The system uses YOLOv26 trained on basketball-specific datasets with 5 detection classes:

    0 - "Ball" 
    1 - "Ball in Basket"
    2 - "Player"
    3 - "Basket"
    4 - "Player Shooting"

#### Detection Classes

| ID | Class | Function | Default Confidence |
|----|-------|----------|-------------------|
| 0 | Ball | Track basketball | 0.60 |
| 1 | Ball in Basket | Detect successful shot | 0.25 |
| 2 | Player | Identify all players | 0.70 |
| 3 | Basket | Localize hoop | 0.70 |
| 4 | Player Shooting | Identify shooter | 0.70 |

### Pipeline Architecture

#### Frame-by-Frame Processing

```
Video Input
    ↓
[Split into Frames]
    ↓
[YOLO Detection] → Bounding boxes + confidence scores
    ↓
[Tracking] → Assign stable IDs across frames
    ↓
[Statistics] → Calculate shots, baskets, per-player metrics
    ↓
[Visualization] → Draw detections, HUD, minimap, effects
    ↓
[Write Frame] → Encode back into video
    ↓
Video Output (MP4)
```

#### Core Components

- **Config** - Centralized configuration management
- **GameStats** - Accumulates shot/basket counts with per-player breakdown
- **Visualizer** - OpenCV drawing utilities for HUD, effects, and court visualization
- **MinimapRenderer** - Projects 2D detections onto court using homography transformation
- **VideoProcessor** - Main pipeline: YOLO inference + output video encoding
- **AutoCleanup** - Daemon thread for automatic cleanup of old processed files

### Technical Details

#### Physics & Cooldown Logic

**Problem**: Single shooting motion detected in 10+ consecutive frames

**Solution**: Temporal cooldown windows

```python
SHOT_COOLDOWN = 1.0     # seconds - wait after detecting a shot
BASKET_COOLDOWN = 1.0   # seconds - wait after basket scoring
ANIMATION_DURATION = 1  # seconds - pulse effect duration
```

**How it works:**
1. "Player Shooting" detected → start shot cooldown timer
2. During cooldown, ignore new shot detections
3. "Ball in Basket" detection → basket only counts if outside basket cooldown
4. Prevent duplicate counting across frame boundaries

#### Custom Augmentation Strategy

Training augmentation optimized for sports footage:

```python
AUGMENTATION = {
    # Color & Lighting
    'hsv_h': 0.015,      # Hue variation (color shifts)
    'hsv_s': 0.4,        # Saturation (orange ball in any light)
    'hsv_v': 0.2,        # Value/brightness (shadows, flares)
    
    # Geometry & Position
    'degrees': 10,       # Rotation (camera angles)
    'translate': 0.1,    # Translation (object movement)
    'scale': 0.5,        # Scaling (depth variation)
    'shear': 2.0,        # Shearing (perspective)
    
    # Advanced
    'flipud': 0.5,       # Vertical flip
    'fliplr': 0.5,       # Horizontal flip
    'mosaic': 1.0,       # Mosaic augmentation
    'mixup': 0.15        # Mixup (blend images)
}
```

**Why**: Sports videos have high motion, varied lighting, and crowded scenes.

#### Court Projection

The minimap uses homography transformation to:
1. Map 3D court coordinates to 2D video frame
2. Project detected players onto bird's-eye court view
3. Show real-time player positions and shot locations

**Calibration Requirements:**
- 4+ court corner points from reference image
- Corresponding image coordinates
- Generated transformation matrix (`homography.npy`)

**Court Reference Diagram:**
```
                          baseline
    [5]───[7]───────────[1]───────[2]───────────[8]───[6]
  L  |     |             |  ─[B]─  |             |     | R
     |     |             |         |             |     |
  s  |     |             | ······· |             |     | s
  i  [10]  |             ··       ··             |  [11] i
  d  |?    |            [3]───────[4]            |    ?| d
  e  |?     ··           ··       ··           ··     ?| e
  l  |??      ···          ·······          ···      ??| l
  i  |??         ···                     ···         ??| i
  n  |???           ·········[9]·········           ???| n
  e  |????                                         ????| e
     |???????                                   ???????|
     |???????????????                   ???????????????| 
     ────────────────────────[C]────────────────────────
```

**Key Points:**
- **[1], [2]** - Baseline corners (where shooters stand)
- **[3], [4]** - Basket/hoop area corners
- **[5], [6]** - Sideline baseline intersections
- **[7], [8]** - Sideline extended baseline intersections
- **[9]** - Center court mark
- **[10], [11]** - Sideline extended points
- **[B]** - Basketball position example
- **[C]** - Court center
- **[?]** - Area not in camera

## Training & Performance

### Training a Custom Model

```bash
cd BE
.\venv\Scripts\activate
python train_model.py
```

The training script:
- Automatically validates datasets
- Verifies GPU hardware capabilities
- Implements graceful interrupt handling (Ctrl+C)
- Provides custom metrics visualization
- Uses optimized hyperparameters for basketball motion tracking

**Configuration (edit `train_model.py`):**
- `PROJECT_NAME` - Output directory for training runs
- `RUN_NAME` - Identifier for this training session
- `DATASET_DIR` - Path to your Roboflow dataset
- `BASE_MODEL` - Starting model (yolo26s.pt, yolo26m.pt, yolo26l.pt)
- `WORKERS` - Data loader workers (0 for Windows compatibility)
- `DEVICE` - GPU index (0 for first GPU)
- `SEED` - Random seed for reproducibility

### Model Performance

#### Training Configuration

**Hardware Used:**
- GPU: NVIDIA RTX 3080 8GB
- CPU: Intel i7-12700H
- RAM: 32GB DDR4

**Model Specifications:**
- **Architecture**: YOLOv26m (medium variant)
- **Input Size**: 640×640 pixels
- **Epochs**: 200
- **Batch Size**: 8
- **Optimizer**: SGD
- **Learning Rate**: 0.01 → 0.0005

#### Performance Metrics

**Overall Accuracy** (Epoch 200):
- **mAP50**: 0.909 (Mean Average Precision at IoU ≥ 0.5)
- **mAP50-95**: 0.623 (Average across IoU thresholds)
- **Precision**: 0.878
- **Recall**: 0.861

**Per-Class Performance:**

| Class | Precision | Recall | mAP50 | Strength |
|-------|-----------|--------|-------|----------|
| Ball | 0.80 | 0.88 | 0.847 | Robust detection |
| Ball in Basket | 0.51 | 0.36 | 0.932 | Rare but distinctive |
| Player | 0.86 | 0.85 | 0.928 | Excellent tracking |
| Basket | 0.91 | 0.91 | 0.966 | Very reliable |
| Player Shooting | 0.76 | 0.34 | 0.873 | Rare pose |

#### Training Curves

Training metrics visualizations available in `FE/public/`:
- `results.png` - Loss curves (box, class, total)
- `confusionMatrix.png` - Classification accuracy
- `normalizedMatrix.png` - Normalized confusion matrix
- `PR_curve.png` - Precision-recall tradeoff
- `P_curve.png` - Precision vs confidence threshold
- `R_curve.png` - Recall vs confidence threshold
- `F1_curve.png` - F1-score optimization curve

### Improving Model Performance

For better accuracy with better hardware:

```python
# In train_model.py Config class:
EPOCHS = 300              # Longer training
BATCH_SIZE = 16           # Larger batches (RTX 3080+)
BASE_MODEL = "yolo26m.pt" # Larger model (medium)
# or
BASE_MODEL = "yolo26l.pt" # Large model (24GB+ VRAM)
```

**Advanced Techniques:**
- Extended training (300+ epochs)
- Larger model variants (YOLOv26m, YOLOv26l)
- Additional data augmentation
- Fine-tuning on specific courts
- Ensemble methods

## Configuration

### Environment Variables

Create a `.env` file in the `BE/` directory if you need custom environment settings:

```bash
# BE/.env (optional)
MODEL_DEVICE=0              # GPU device index (0 for first GPU, -1 for CPU)
LOG_LEVEL=INFO              # Logging level
CLEANUP_ENABLED=true        # Auto-cleanup of old files
```

The application will use defaults if `.env` is not present. See `app.py` Config class for all available settings.

### Tracker Configuration

Two tracking algorithms are available. Adjust in `BE/app.py`:

**BoTSORT (Default - Better for stable tracking):**
- File: `BE/tracker/botsort.yaml`
- Better at maintaining player IDs across occlusions
- Uses ReID (Re-Identification) model for appearance matching
- Tunable parameters:
  - `track_high_thresh` - Match confidence (0.30)
  - `track_buffer` - Frames to keep lost tracks (60)
  - `gmc_method` - Motion compensation for camera movement

**ByteTrack (Alternative - Better for fast motion):**
- File: `BE/tracker/bytetrack.yaml`
- Faster processing, handles rapid motion well
- Lower computational overhead

Switch tracker in `app.py`:
```python
TRACKER_PATH = Path(__file__).parent / "tracker" / "botsort.yaml"  # or bytetrack.yaml
```

### Important Directories

**Files to Keep/Backup:**
- `BE/basketball_training/` - Trained model weights (critical)
- `BE/tracker/` - Calibration files and tracker configs
- `FE/` - Frontend source code

**Auto-Cleanup Directories** (files older than 10 minutes are deleted):
- `BE/uploads/` - Uploaded videos
- `BE/processed/` - Processed output videos
- `BE/runs/` - Training run outputs

**Large Files (Not in Git):**
- `*.pt` - Model weight files
- `BE/basketball-detection-srfkd-1/` - Training datasets
- `BE/processed/` - Output videos
- `BE/uploads/` - Uploaded videos

See `.gitignore` for complete list of ignored files.

### Backend Configuration

Edit configuration values in `BE/app.py` (Config class):

```python
class Config:
    # Path settings
    UPLOAD_DIR = Path(__file__).parent / "uploads"
    PROCESSED_DIR = Path(__file__).parent / "processed"
    MODEL_PATH = Path(__file__).parent / "basketball_training" / "yolo26s_5classes" / "weights" / "best.pt"
    
    # Video constraints
    MAX_DURATION_SECONDS = 180   # 3 minutes max
    TEST_MODE_DURATION = 15      # Test mode length
    
    # Processing settings
    CONFIDENCE_THRESHOLD = 0.5        # YOLO confidence threshold
    IOU_THRESHOLD = 0.45              # NMS IoU threshold
    MAX_DETECTIONS = 100              # Maximum detections per frame
    
    # File retention (auto-cleanup)
    RETENTION_SECONDS = 600      # 10 minutes
    CLEANUP_INTERVAL = 60        # Check every 60s
    
    # Physics (in seconds)
    SHOT_COOLDOWN = 1.0          # Shot detection debounce
    BASKET_COOLDOWN = 1.0        # Basket detection debounce
    ANIMATION_DURATION = 1       # Pulse effect length
    
    # Detection thresholds (0.0-1.0)
    THRESHOLDS = {
        0: 0.60,   # Ball
        1: 0.25,   # Ball in Basket
        2: 0.70,   # Player
        3: 0.70,   # Basket
        4: 0.70    # Player Shooting
    }
    
    # Court dimensions (cm)
    COURT_WIDTH_CM = 1500        # FIBA half-court width
    COURT_HEIGHT_CM = 1400       # FIBA half-court depth
```

### Adjusting for Different Scenarios

**Dim Lighting:**
```python
THRESHOLDS = {
    0: 0.50,   # Lower ball threshold
    1: 0.20,   # Lower basket threshold
    2: 0.65,   # Slightly lower player
    3: 0.65,
    4: 0.65
}
```

**Crowded Indoor Court:**
```python
SHOT_COOLDOWN = 0.8      # Faster shot detection
BASKET_COOLDOWN = 1.2    # Slower basket debounce
THRESHOLDS = {
    2: 0.75,   # Higher player threshold
    4: 0.75    # Stricter shooting pose
}
```

## Performance Optimization

### GPU Acceleration

The `install_env.bat` script automatically detects and configures PyTorch for your NVIDIA GPU:

- **CUDA 12.8+** → PyTorch with cu128 support
- **CUDA 12.4-12.7** → PyTorch with cu124 support
- **CUDA 11.8+** → PyTorch with cu118 support
- **No GPU** → PyTorch CPU version

### Model Selection

Choose the appropriate model size based on your hardware:

- `yolo26n.pt` - Nano (fastest, least accurate)
- `yolo26s.pt` - Small (balanced, recommended)
- `yolo26m.pt` - Medium (slower, more accurate)

Adjust `MODEL_PATH` in Config to switch models.

**Switching Model Variant:**

Edit `BE/app.py` line 44 and update the `MODEL_PATH`:

```python
# Current (medium model)
MODEL_PATH = Path(__file__).parent / "basketball_training" / "yolo26m_5classes_2" / "weights" / "best.pt"

# To use small model (if trained)
MODEL_PATH = Path(__file__).parent / "basketball_training" / "yolo26s_5classes" / "weights" / "best.pt"
```

## Utility Scripts

### 1. Court Calibration Tool (`calibrate.py`)

Interactive tool to calibrate court homography transformation for your specific camera setup:

```bash
cd BE
.\venv\Scripts\activate

# Interactive wizard for custom courts
python calibrate.py --video uploads/sample.mp4 --setup

# FIBA preset calibration (automatic)
python calibrate.py --video uploads/sample.mp4
python calibrate.py --video uploads/sample.mp4 --frame 120
python calibrate.py --image sample_frame.jpg
```

**Controls:**
- **Left-click** → Place required calibration point
- **Right-click** → Place/skip optional points
- **U** → Undo last point
- **R** → Reset all points
- **S** → Save and preview
- **P** → Preview without saving
- **Q / ESC** → Quit

**Output:** Generates `homography.npy` and `homography_scale.npy` in `BE/tracker/`

### 2. Live Camera Testing (`live_test.py`)

Real-time basketball detection and tracking using your webcam:

```bash
cd BE
.\venv\Scripts\activate
python live_test.py
```

**Features:**
- Real-time YOLO inference on camera feed
- Live player and ball tracking
- Shot and basket detection
- Performance metrics (FPS, detection count)
- Press **Q** to quit

**Configuration:**
Edit the script to change:
- `CAMERA_INDEX` - Webcam number (0 for default)
- Confidence thresholds
- Tracker algorithm (ByteTrack/BoTSORT)

### 3. Metrics & Analysis (`metrics.py`)

Detailed performance analysis and visualization for trained models:

```bash
cd BE
.\venv\Scripts\activate
python metrics.py
```

**Generates:**
- Training history analysis
- Per-class performance breakdown
- Overfitting analysis
- Confidence threshold analysis
- Performance report JSON
- Training curves PNG visualization

**Configuration:**
Edit to analyze different trained runs:
```python
RUN_NAME = "yolo26m_5classes_2"  # Change to your run
```

### 4. Player Report Generator (`player_report.py`)

Generate per-player shot charts and statistics:

```bash
cd BE
python player_report.py processed/<file_id>_stats.csv
```

**Output:**
- Shot position visualization on court minimap
- Per-player statistics overlay
- PNG report with shot success/failure markers
- CSV-compatible input (includes shot coordinates and scoring flag)

### 5. Alternative Models (SAM3 Variants)

Two alternative backend configurations using SAM3 (Segment Anything Model 3):

**Box-based Segmentation:**
```bash
cd BE
.\venv\Scripts\activate
python app_sam3_box.py
```

**Semantic Segmentation:**
```bash
cd BE
.\venv\Scripts\activate
python app_sam3_semantic.py
```

These are experimental variants for advanced segmentation use cases. Configuration similar to main `app.py`.

## Development

### Backend Development

```bash
cd BE
.\venv\Scripts\activate
python app.py --reload  # Auto-reload on file changes (if using Uvicorn directly)
```

### Frontend Development

```bash
cd FE
npm run dev
npm run lint  # Run ESLint
npm run build # Build for production
```

### Testing & Validation

**Linting:**
```bash
cd FE
npm run lint
```

**Building for Production:**
```bash
cd FE
npm run build
```

### Deployment Notes

**Backend for Production:**

Replace `python app.py` with a production ASGI server:
```bash
# Using Gunicorn with Uvicorn workers (Linux/macOS)
gunicorn app:app -w 4 -k uvicorn.workers.UvicornWorker --bind 0.0.0.0:8000

# Or use built-in reload-disabled production mode
python app.py --host 0.0.0.0 --port 8000
```

**Frontend for Production:**

```bash
cd FE
npm run build
# Outputs optimized files to dist/ directory
# Serve with: python -m http.server --directory dist 5173
```

**CORS Configuration for Different Domains:**

Edit `BE/app.py` CORSMiddleware settings to allow your production frontend URL:
```python
CORSMiddleware(
    allow_origins=["http://localhost:5173", "https://yourdomain.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

## Troubleshooting

### Installation Issues

**Python not found:**
```
Ensure Python is in your system PATH
Run: python --version
```

**Virtual environment creation fails:**
```
Check write permissions in the BE directory
Delete any existing venv folder: rmdir /s /q BE\venv
Retry the setup script
```

**PyTorch GPU detection fails:**
```
Verify NVIDIA drivers: nvidia-smi
Check CUDA compatibility: https://www.nvidia.com/drivers
The system will fall back to CPU automatically
```

### Processing Issues

**Out of memory error:**
```
Reduce frame resolution in Config
Use a smaller model (yolo26n.pt or yolo26s.pt)
Enable GPU memory optimization in app.py
```

**Slow processing:**
```
Enable GPU acceleration (check nvidia-smi)
Reduce input video resolution
Decrease confidence threshold slightly
```

### Frontend Issues

**Cannot connect to API:**
```
Verify backend is running: http://localhost:8000/docs
Check CORS configuration in app.py
Ensure frontend is on http://localhost:5173
```

**npm install fails:**
```
Delete node_modules and package-lock.json: 
  rmdir /s /q node_modules && del package-lock.json
  npm install
```

**Vite dev server won't start:**
```
Check if port 5173 is already in use
Try: npm run dev -- --host 127.0.0.1 --port 5174
```

**ESLint errors during development:**
```
Fix automatically: npm run lint -- --fix
Check configuration: FE/eslint.config.js
```

## Output Files

### Processed Video
- Format: MP4 (H.264 codec)
- Contains: YOLO detections, player tracking IDs, HUD with game statistics
- Minimap showing court perspective
- Visual effects highlighting detected objects

### Statistics CSV
- Columns: Frame number, timestamp, player IDs, detected objects, shots, baskets
- Per-player aggregated statistics
- Useful for further analysis in Excel or Python

## Limitations & Future Enhancements

**Current Limitations:**
- Requires clear basketball court visibility
- Performance varies with lighting conditions
- Real-time processing depends on GPU availability

**Planned Features:**
- Ball possession tracking
- Shot trajectory analysis
- Web-based model retraining interface

## CORS Configuration

The FastAPI backend is configured to accept requests from the React frontend:

```python
CORSMiddleware(
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Modify `app.py` if running on different ports.

## License

This project is released under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

**Why AGPL-3.0?**
This project integrates **Ultralytics YOLO**, which is licensed under AGPL-3.0. As a derivative work, RotoAI inherits this license to ensure full compliance with the open-source terms of its dependencies.

**What this means for you:**
- **Use:** You can use this software for personal, research, or commercial purposes.
- **Modify:** You can modify the source code.
- **Share:** If you distribute this software or host it as a network service (SaaS), you **must** disclose the source code of your modified version under the same AGPL-3.0 license.

## 📜 Credits

- **Original Work**: sPappalard
- **Developer**: DiegoGMD
- **Dataset**: [Roboflow Universe - Basketball Detection](https://universe.roboflow.com/basketball-6vyfz/basketball-detection-srfkd) + Custom Dataset
- **Frameworks**: Ultralytics YOLO, FastAPI, React

---

**Last Updated:** 2026-05-26  
**Version:** 1.0  
**Status:** Active Development
