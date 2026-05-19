# Basketball AI Tracker

[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)
[![es](https://img.shields.io/badge/lang-es-red.svg)](README.es.md)

> [!NOTE] This is repository a slight modification of sPappalard's one named [SwishAI](https://github.com/sPappalard/SwishAI/tree/master)

A comprehensive computer vision system for real-time basketball training analysis and player tracking. This project combines deep learning object detection with video processing to automatically identify players, basketballs, and court elements, providing detailed training statistics and visual analytics.

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

## Project Structure

```
Basketball_AI/
├── BE/                              # Backend (Python)
│   ├── app.py                       # Main FastAPI application
│   ├── train_model.py               # YOLO model training script
│   ├── metrics.py                   # Training metrics and analysis
│   ├── calibrate.py                 # Camera calibration utilities
│   ├── basketball_training/         # Trained models directory
│   │   └── yolo26s_5classes/        # Deployed model (5 detection classes)
│   ├── tracker/                     # Tracking configs and court images
│   │   ├── bytetrack.yaml           # ByteTrack configuration
│   │   ├── botsort.yaml             # BoTSORT configuration (default)
│   │   ├── minimap.png              # Court reference image
│   │   └── homography.npy           # Homography transformation matrix
│   ├── uploads/                     # User-uploaded videos (auto-created)
│   ├── processed/                   # Processed output videos (auto-created)
│   ├── requirements.txt             # Python dependencies
│   └── venv/                        # Python virtual environment (created by setup)
│
├── FE/                              # Frontend (React)
│   ├── src/
│   │   ├── App.jsx                  # Main React component
│   │   ├── App.css                  # Application styles
│   │   ├── main.jsx                 # React entry point
│   │   ├── index.css                # Global styles
│   │   └── assets/                  # Static assets
│   ├── public/                      # Public static files
│   ├── package.json                 # Node dependencies
│   ├── vite.config.js               # Vite configuration
│   ├── tailwind.config.js           # Tailwind CSS configuration
│   └── eslint.config.js             # ESLint configuration
│
├── install_env.bat                  # Windows environment setup script
├── LICENSE / LICENSE.txt            # Project licensing
└── README.md                        # Original documentation
```

## Getting Started

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

3. **Configure Detection (Optional)**
   - Set custom confidence thresholds for different object classes
   - Choose tracking algorithm (ByteTrack or BoTSORT)

4. **Start Processing**
   - Trigger the video processing pipeline
   - The system runs YOLO inference on all frames
   - Player and ball coordinates are tracked across frames

5. **Monitor Progress**
   - Poll the status endpoint for real-time progress updates
   - Live statistics update as frames are processed

6. **Download Results**
   - Download the processed video with tracking overlays and HUD
   - Download the CSV report with per-frame and per-player statistics

### API Endpoints

**Video Processing:**
- `POST /upload` - Upload a video file
- `POST /process/{file_id}` - Start background processing with custom parameters
- `GET /status/{file_id}` - Get processing progress and live statistics
- `GET /download-zip/{file_id}` - Download processed video + CSV report as zip

**Model Management:**
- `GET /models` - List available detection models
- `GET /health` - System health check

## Model Architecture

The system uses YOLOv11 trained on basketball-specific datasets with 5 detection classes:

1. **Player** - Basketball player
2. **Ball** - Basketball
3. **Hoop** - Basketball hoop/rim
4. **Court** - Court boundaries/markings
5. **Backboard** - Backboard structure

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

## Pipeline Architecture

### 1. Frame-by-Frame Processing

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

### 2. Core Components

- **Config** - Centralized configuration management
- **GameStats** - Accumulates shot/basket counts with per-player breakdown
- **Visualizer** - OpenCV drawing utilities for HUD, effects, and court visualization
- **MinimapRenderer** - Projects 2D detections onto court using homography transformation
- **VideoProcessor** - Main pipeline: YOLO inference + output video encoding
- **AutoCleanup** - Daemon thread for automatic cleanup of old processed files

## Configuration

Edit configuration values in `BE/app.py` (Config class):

```python
class Config:
    # Path settings
    UPLOAD_DIR = Path(__file__).parent / "uploads"
    PROCESSED_DIR = Path(__file__).parent / "processed"
    MODEL_PATH = Path(__file__).parent / "basketball_training" / "yolo26s_5classes" / "weights" / "best.pt"
    
    # Processing settings
    CONFIDENCE_THRESHOLD = 0.5        # YOLO confidence threshold
    IOU_THRESHOLD = 0.45              # NMS IoU threshold
    MAX_DETECTIONS = 100              # Maximum detections per frame
    
    # File management
    RETENTION_SECONDS = 3600          # Keep processed files for 1 hour
    CLEANUP_INTERVAL = 60             # Check for cleanup every 60 seconds
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

## Dataset & Training

The project uses Roboflow for dataset management:

```bash
cd BE
.\venv\Scripts\activate
```

Then configure and run `train_model.py` to train on your basketball dataset.

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

**Last Updated:** 2026-05-19  
**Version:** 1.0  
**Status:** Active Development
