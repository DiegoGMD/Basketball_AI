# Basketball AI Tracker

[![es](https://img.shields.io/badge/lang-es-red.svg)](README.es.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)

> [!NOTE] Este repositorio es una ligera modificación del de sPappalard, llamado [SwishAI](https://github.com/sPappalard/SwishAI/tree/master)

Un "compresnsible" sistema de computer vision para el analisis en tiempo real de entrenamientos de baloncesto y seguimiento de jugadores. Este proyecto combina detección de objetos usando deep learning con procesamiento de vídeo para identificar automáticamente a jugadres, pelotas de baloncesto y otros elementos de la pista, proporcionando detalladas estadisticas sobre el entrenamiento. 

## Resumen

Basketball AI Tracker usa las redes neuronales de YOLO (You Only Look Once) para detectar y hacer un seguimiento de los jugadores y las pelotas en vídeos de entrenamientos. El sistema procesa cada fotograma en tiempo real, genera vídeos con anotaciones y marcas de seguimiento, genera una visualización con minimapa, y exporta detalladas estadisticas en formato CSV.  

**Capacidades Clave**
- Detección en tiempo real de jugadores, pelotas y elementos de la pista
- Seguimiento multi-persona
- Conteo automático de los tiros y canastas
- Estadisticas del entrenamiento en vivo (por jugador)
- Visualización con minimapa y transformación de homografía
- Salida en vídeo con interfaz y efectos visuales
- Procesamiento en batch con seguimiento del proceso
- Limpieza automática de los archivos procesados 

## Stack Tecnológico

### Backend
- **Framework:** FastAPI 0.115.0 con Uvicorn
- **Computer Vision:** YOLOv26 (via Ultralytics), OpenCV
- **Deep Learning:** PyTorch 2.5.1 con soporte CUDA
- **Procesamiento de Datos:** NumPy, Pandas, scikit-learn
- **Visualización:** Matplotlib, Seaborn
- **Utilidades:** Roboflow (dataset management), python-dotenv, PyYAML

### Frontend
- **Framework:** React 19.1.1 with Vite
- **Styling:** Tailwind CSS with PostCSS/Autoprefixer
- **Components de la Interfaz:** Lucide React (icons), React QR Code
- **Linting:** ESLint

## Estructura del Proyecto

```
Basketball_AI/
├── BE/                                 # Backend (Python)
│   ├── app.py                          # Principal aplicación FastAPI
│   ├── calibrate.py                    # Programa de calibración de la imagen
│   ├── metrics.py                      # Metricas de entrenamiento y análisis
│   ├── player_report.py                # Programa de informe de jugador
│   ├── requirements.txt                # Dependéncias de Python
│   ├── train_model.py                  # Programa de entrenamiento del modelo de YOLO
│   ├── yolo.pt                         # Pesos de tu modelo de YOLO
│   ├── basketball-detection-srfkd-1/   # Directorio del dataset
│   │   ├── test/
│   │   ├── train/
│   │   └── valid/
│   ├── basketball_training/            # Directorio de los modelos entrenados
│   │   └── yolo_5classes/
│   ├── metrics_reports/                # Informes de funcionameinto
│   │   ├── overfitting_analysis.png
│   │   ├── performance_report.json
│   │   └── training_curves.png
│   ├── tracker/                        # Configuraciones del seguimiento y minimapa
│   │   ├── botsort.yaml
│   │   ├── bytetrack.yaml
│   │   ├── court.excalidraw
│   │   ├── half_court_y.npy
│   │   ├── homography.npy
│   │   ├── homography_scale.npy
│   │   ├── minimap.png
│   │   └── reprojection_preview.png
│   ├── uploads/                        # Vídeos subidos por el usuario (auto-created)
│   ├── processed/                      # Procesada salida de vídeo (auto-created)
│   ├── runs/                           # Procesada salida de vídeo (auto-created)
│   └── venv/                           # Entorno virtual de Python
│
├── FE/                                 # Frontend (React)
│   ├── package.json                    # Dependencias de Node
│   ├── package-lock.json               # npm lockfile
│   ├── vite.config.js                  # Configuración de Vite
│   ├── tailwind.config.js              # Configuración de Tailwind CSS
│   ├── postcss.config.js               # Configuración de PostCSS
│   ├── eslint.config.js                # Configuración de ESLint
│   ├── index.html                      # SPA entrada HTML
│   ├── README.md                       # Frontend README
│   ├── public/                         # Public static files
│   └── src/
│       ├── App.jsx                    # Componente principal de React
│       ├── App.css                    # Estilos de la aplicación
│       ├── main.jsx                   # Punto de entrada de React
│       ├── index.css                  # Estilos globales
│       └── assets/                    # Recursos estáticos
├── .gitignore                         # Archivos ignorados
├── install_env.bat                    # Script de la instalacion del entorno para Windows
├── LICENSE                            # Licecia del proyecto
├── README.md                          # Documentación original
└── README.es.md                       # Documentación en español
```

## Preparación

### Prerequisitos

- **Windows OS** (o Linux con modificaciones al script de configuración)
- **Python 3.8+** (disponible en la ruta del sistema)
- **Node.js 16+** (para el desarrollo frontend)
- **Opcional:** NVIDIA GPU con CUDA 11.8+ (recommendado para una inferencia más rápida)

### Instalación

#### 1. Configuración Backend (Windows)

Ejecuta el script de configuración automatica de entorno:

```bash
install_env.bat
```

Este script:
- Crea el entorno virtual de Python
- Detecta tu GPU de NVIDIA y tu versión de CUDA
- Instala PyTorch con el apropiado soporte para la GPU (o CPU si no encuantra GPU)
- Instala todas las dependencias necesarias
- Verifica la configuración

**Configuración Manual (si se necesita):**

```bash
cd BE
python -m venv venv
.\venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### 2. Configuración Frontend

```bash
cd FE
npm install
```

### Ejecutando la Aplicación

#### Backend (Servidor API)

```bash
cd BE
.\venv\Scripts\activate           # Activa el entorno virtual
python app.py
```
El servidor FastAPI se iniciará en `http://localhost:8000` 
- Documentación de la API: `http://localhost:8000/docs` (Swagger UI)
- documentos alternativos:  `http://localhost:8000/redoc` (ReDoc)

#### Frontend (Servidor de Desarrollo)

```bash
cd FE
npm run dev
```

El servidor de desarrollo de Vite se iniciará en `http://localhost:5173`

## Uso

### Preocesado del Vídeo de Baloncesto

1. **Accede a la Interfaz Web**
    - Navega hasta `http://localhost:5173` en tu navegador

2. **Sube el Vídeo**
    - Selecciona el vídeo de la sesión de entrenamiento
    - El sistema genera un identificador úniuco para el seguimiento

3. **Configura la Detección (Opcional)**
    - Cofigura el umbral de la confianza para distintas clases de objetos
    - Elije el algoritmo de seguimiento (ByteTrack o BoTSORT)

4. **Inicia el Procesamiento**
    - Activa la cadena de procesamiento del vídeo
    - El programa ejecuta a inferencia de YOLO en todos los fotogramas
    - Las coordenadas del jugador y la pelota se van siguiendo fotograma a fotograma  

5. **Monitorea el Progreso**
    - Poll the status endpoint for real-time progress updates
    - Live statistics update as frames are processed

6. **Descarga los Resultados**
    - Descarga el vídeo procesado con el seguimiento y la interfaz
    - Descarga el informe en CSV de las estadisticas

### API Endpoints

**Procesamiento del Vídeo:**
- `POST /upload` - Sube el archivo de vídeo
- `POST /process/{file_id}` - Inicia el procesamiento de fondo con parametros personalizados
- `GET /status/{file_id}` - Consigue el progreso de procesado y las estadisticas en vivo
- `GET /download-zip/{file_id}` - Descarga el vídeo procesado + el informe CSV como un zip

**Gestion del Modelo:**
- `GET /models` - Lista los modelos de detección disponibles
- `GET /health` - Revisión de salud del sistema

## Architectura del Modelo

El programa usa YOLOv26 entrenado en datasets especificos de baloncesto con 5 clases a detectar: 

    0 - "Pelota" 
    1 - "Pelota dentro de la canasta"
    2 - "Jugador/a"
    3 - "Canasta"
    4 - "Jugador/a lanzando"

### Entrnando un Modelo Personalizado

```bash
cd BE
.\venv\Scripts\activate
python train_model.py
```
El script de entrenamiento:
- Valida datasets automaticamente
- Verifica compatibilidades con la GPU
- Implementa un manejo correcto de interrupciones (Ctrl+C)
- Provee visualización de metricas personalizadas
- Usa hiperparametros optimizados para el seguimiento en el baloncesto

**Configuración (editar `train_model.py`):**
- `PROJECT_NAME` - Directorio de salida para las ejecuciones del entrenamiento
- `RUN_NAME` - Identificador para la sesión de entrenamiento
- `DATASET_DIR` - Ruta al dataset
- `BASE_MODEL` - Modelo de inicio o base (yolo26s.pt, yolo26m.pt, yolo26l.pt)
- `WORKERS` - Procesos del cargador de datos (0 para compatibilidad con Windows)
- `DEVICE` - GPU index (0 for first GPU)
- `SEED` - Semilla aleatoria para la reproducibilidad

## Pipeline Architecture

### 1. Procesamiento Fotograma a Fotograma

```
Entrda de Vídeo
    ↓
[Separar en fotogramas]
    ↓
[Detección con YOLO] → Cajas Delimitdoras + Puntuacion de Confianza
    ↓
[Seguimiento] → Asigne un identificador estable a través de los fotogramas
    ↓
[Estadísticas] → calcula tiros y canastas por jugador
    ↓
[Visualización] → Dibuja las detecciones, la interfaz, el minimapa i los efectos
    ↓
[Escribir Fotogramas] → Codifica en un Vídeo
    ↓
Vídeo de Salida (MP4)
```

### 2. Componetes Principales

- **Config** - Gestión Centralizada de la Configuración
- **GameStats** - Acumula el contador de tiros y canastas por jugador
- **Visualizer** - Funcionalidades de OpenCV para dibujar la interfaz, efectos y vilialización de la pista
- **MinimapRenderer** - Proyecta las detecciones en 2D en la pista usando transformacion de homografía
- **VídeoProcessor** - Cadena de procesamiento principal: inferencia YOLO + salida del vídeo Main pipeline: YOLO inference + output vídeo encoding
- **AutoCleanup** - Hilo en segundo plano para la limoieza automatica de archivos procesados antiguos

## Configuración

Edita los archivos de configuración en `BE/app.py` (clase Config):

```python
class Config:
    # Path settings
    UPLOAD_DIR = Path(__file__).parent / "uploads"
    PROCESSED_DIR = Path(__file__).parent / "processed"
    MODEL_PATH = Path(__file__).parent / "basketball_training" / "yolo26s_5classes" / "weights" / "best.pt"
    
    # Ajusted de procesamiento
    CONFIDENCE_THRESHOLD = 0.5        # Umbral de confianza de YOLO
    IOU_THRESHOLD = 0.45              # Umbral NMS IoU
    MAX_DETECTIONS = 100              # Detecciones maximas por fotograma
    
    # Gestión de archivos
    RETENTION_SECONDS = 3600          # Guarda los archivos procesados por una hora
    CLEANUP_INTERVAL = 60             # Revisa para limpiar cada 60 segundos
```

## Optimización de "Performance"

### Aceleración por GPU

El script `install_env.bat` automaticamente detecta y configura PyTorch para tu GPU NVIDIA: 

- **CUDA 12.8+** → PyTorch con soporte cu128
- **CUDA 12.4-12.7** → PyTorch con soporte cu124
- **CUDA 11.8+** → PyTorch con soporte cu118
- **No GPU** → PyTorch versión CPU

### Selección de Modelo

Elige el modelo apropiado basándote en tu hardware:

- `yolo26n.pt` - Nano (rápido, menos preciso)
- `yolo26s.pt` - Small (equilibrado, recomendado)
- `yolo26m.pt` - Medium (lento, mas preciso)

Ajusta `MODEL_PATH` en Config para cambiar de modelos.

## Detección de Errores

### Problemas de Instalación

**Python no encontrado:**
```
Asegurate de que Python esta en la ruta del sistema
Ejecuta: python --version
```

**Falla la creación del entorno virtual:**
```
Revisa los permisos de escritura en el directorio BE
Elimina cualquier carpeta venv existente: rmdir /s /q BE\venv
Vuelve a intentar con el secipt de configuración
```

**PyTorch detección de GPU falla:**
```
Verifica los drivers de NVIDIA: nvidia-smi
Recisa la compatibilidad de CUDA: https://www.nvidia.com/drivers
El programa se retirará a la CPU automaticamente
```

### Problemas de Procesamiento

**Sin memoria:**
```
Reduce la resolución de los fotogramas en Config
Usa un modelo mas pequeño (yolo26n.pt or yolo26s.pt)
Habilita la optimización de memoria de la GPU en app.py
```

**Procesamiento lento:**
```
Habilita la aceleración por GPU (revisa nvidia-smi)
Reduce la resolución del video de entrada
Reduce el umbral de confianza ligeramente 
```

### Problemas con Frontend

**No se puede conectar a la API:**
```
Verifica que el backend se está ejecutando http://localhost:8000/docs
Revisa la configuración CORS en app.py
Asegurate que el frontend está activo http://localhost:5173
```

## Desarrollo

### Desarrollo Backend

```bash
cd BE
.\venv\Scripts\activate
python app.py --reload  # Recarga automaticamente si cambia el archivo (si usa Uvicorn direcamente)
```

### Frontend Development

```bash
cd FE
npm run dev
npm run lint  # Ejecuta ESLint
npm run build # Versión para producción
```

### Pruebas y Validación

**Linting:**
```bash
cd FE
npm run lint
```

**Versión para producción:**
```bash
cd FE
npm run build
```

## CORS Configuration

El backend FastAPI está configurado para aceptar solicitudes del frontend en React:

```python
CORSMiddleware(
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Modifica `app.py` para ejectar en otros puertos. 

## Output Files

### Processed Vídeo
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
