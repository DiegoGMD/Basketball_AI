# Basketball AI Tracker

[![es](https://img.shields.io/badge/lang-es-red.svg)](README.es.md)
[![en](https://img.shields.io/badge/lang-en-blue.svg)](README.md)

> [!NOTE] Este repositorio es una ligera modificación del de sPappalard, llamado [SwishAI](https://github.com/sPappalard/SwishAI/tree/master)

Un sistema comprehensivo de computer vision para el análisis en tiempo real de entrenamientos de baloncesto y seguimiento de jugadores. Este proyecto combina detección de objetos usando deep learning con procesamiento de vídeo para identificar automáticamente a jugadores, pelotas de baloncesto y otros elementos de la pista, proporcionando estadísticas detalladas sobre el entrenamiento.

## Resumen

Basketball AI Tracker usa las redes neuronales de YOLO (You Only Look Once) para detectar y hacer seguimiento de los jugadores y las pelotas en vídeos de entrenamientos. El sistema procesa cada fotograma en tiempo real, genera vídeos con anotaciones y marcas de seguimiento, produce una visualización con minimapa de la cancha, y exporta estadísticas detalladas en formato CSV.

**Capacidades Clave:**
- Detección en tiempo real de jugadores, pelotas y elementos de la pista
- Seguimiento multi-jugador entre fotogramas
- Conteo automático de tiros y canastas
- Estadísticas de entrenamiento en vivo (por jugador)
- Visualización con minimapa y transformación de homografía
- Salida en vídeo con HUD (pantalla de información) y efectos visuales
- Procesamiento por lotes con seguimiento del progreso
- Limpieza automática de archivos procesados

## Tabla de Contenidos

- [Resumen](#resumen)
- [Inicio Rápido](#inicio-rápido)
  - [Requisitos Previos](#requisitos-previos)
  - [Instalación](#instalación)
  - [Ejecutando la Aplicación](#ejecutando-la-aplicación)
- [Uso](#uso)
  - [Procesado del Vídeo de Baloncesto](#procesado-del-vídeo-de-baloncesto)
  - [API Endpoints](#api-endpoints)
- [Estructura del Proyecto](#estructura-del-proyecto)
- [Stack Tecnológico](#stack-tecnológico)
- [Arquitectura](#arquitectura)
  - [Arquitectura del Modelo](#arquitectura-del-modelo)
  - [Arquitectura del Pipeline](#arquitectura-del-pipeline)
  - [Detalles Técnicos](#detalles-técnicos)
- [Entrenamiento y Rendimiento](#entrenamiento-y-rendimiento)
  - [Entrenando un Modelo Personalizado](#entrenando-un-modelo-personalizado)
  - [Rendimiento del Modelo](#rendimiento-del-modelo)
  - [Mejorando el Rendimiento del Modelo](#mejorando-el-rendimiento-del-modelo)
- [Configuración](#configuración)
- [Optimización de Rendimiento](#optimización-de-rendimiento)
- [Desarrollo](#desarrollo)
- [Detección de Errores](#detección-de-errores)
- [Archivos de Salida](#archivos-de-salida)
- [Limitaciones y Mejoras Futuras](#limitaciones-y-mejoras-futuras)
- [Configuración CORS](#configuración-cors)
- [Licencia](#licencia)
- [Créditos](#créditos) 

## Inicio Rápido

### Requisitos Previos

- **Windows OS** (o Linux con modificaciones al script de configuración)
- **Python 3.8+** (disponible en la ruta del sistema)
- **Node.js 16+** (para el desarrollo frontend)
- **Opcional:** NVIDIA GPU con CUDA 11.8+ (recomendado para inferencia más rápida)

### Instalación

#### 1. Configuración Backend (Windows)

Ejecuta el script de configuración automatizada del entorno:

```bash
install_env.bat
```

Este script:
- Crea un entorno virtual de Python
- Detecta tu GPU de NVIDIA y versión de CUDA
- Instala PyTorch con el soporte de GPU apropiado (o CPU si no encuentra GPU)
- Instala todas las dependencias requeridas
- Verifica la configuración

**Configuración Manual (si es necesario):**

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
- Documentos alternativos: `http://localhost:8000/redoc` (ReDoc)

#### Frontend (Servidor de Desarrollo)

```bash
cd FE
npm run dev
```

El servidor de desarrollo de Vite se iniciará en `http://localhost:5173`

## Uso

### Procesado del Vídeo de Baloncesto

1. **Accede a la Interfaz Web**
   - Navega a `http://localhost:5173` en tu navegador

2. **Sube el Vídeo**
   - Selecciona un vídeo de partido o sesión de entrenamiento
   - El sistema genera un ID de archivo único para el seguimiento

3. **Configura la Detección (Opcional)**
   - Establece umbrales de confianza personalizados para diferentes clases de objetos
   - Elige el algoritmo de seguimiento (ByteTrack o BoTSORT)

4. **Inicia el Procesamiento**
   - Activa la cadena de procesamiento de vídeo
   - El sistema ejecuta inferencia de YOLO en todos los fotogramas
   - Las coordenadas de los jugadores y pelotas se rastrean entre fotogramas

5. **Monitorea el Progreso**
   - Consulta el endpoint de estado para actualizaciones de progreso en tiempo real
   - Las estadísticas en vivo se actualizan a medida que se procesan los fotogramas

6. **Descarga los Resultados**
   - Descarga el vídeo procesado con overlays de seguimiento y HUD
   - Descarga el informe de estadísticas en CSV

### API Endpoints

**Procesamiento de Vídeo:**
- `POST /upload` - Sube un archivo de vídeo
- `POST /process/{file_id}` - Inicia el procesamiento en segundo plano con parámetros personalizados
- `GET /status/{file_id}` - Obtiene el progreso de procesamiento y estadísticas en vivo
- `GET /download-zip/{file_id}` - Descarga vídeo procesado + informe CSV como zip

**Gestión del Modelo:**
- `GET /models` - Lista los modelos de detección disponibles
- `GET /health` - Verificación de salud del sistema

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
## Estructura del Proyecto

```
Basketball_AI/
├── BE/                                 # Backend (Python)
│   ├── app.py                          # Principal aplicación FastAPI
│   ├── calibrate.py                    # Utilidades de calibración de imagen
│   ├── metrics.py                      # Análisis de métricas de entrenamiento
│   ├── player_report.py                # Utilidades de informe de jugador
│   ├── requirements.txt                # Dependencias de Python
│   ├── train_model.py                  # Script de entrenamiento del modelo YOLO
│   ├── yolo.pt                         # Pesos base de tu modelo YOLO
│   ├── basketball-detection-srfkd-1/   # Carpeta del dataset
│   │   ├── test/
│   │   ├── train/
│   │   └── valid/
│   ├── basketball_training/            # Directorio de modelos entrenados
│   │   ├── yolo26m_5classes/
│   │   └── yolo26s_5classes/
│   ├── metrics_reports/                # Informes de rendimiento
│   │   ├── overfitting_analysis.png
│   │   ├── performance_report.json
│   │   └── training_curves.png
│   ├── tracker/                        # Configuraciones de seguimiento e imágenes de cancha
│   │   ├── botsort.yaml
│   │   ├── bytetrack.yaml
│   │   ├── court.excalidraw
│   │   ├── half_court_y.npy
│   │   ├── homography.npy
│   │   ├── homography_scale.npy
│   │   ├── minimap.png
│   │   └── reprojection_preview.png
│   ├── uploads/                        # Vídeos subidos por el usuario (auto-creado)
│   ├── processed/                      # Salida de vídeos procesados (auto-creado)
│   ├── runs/                           # Salida de vídeos procesados (auto-creado)
│   └── venv/                           # Entorno virtual de Python
│
├── FE/                                 # Frontend (React)
│   ├── package.json                    # Dependencias de Node
│   ├── package-lock.json               # archivo de bloqueo npm
│   ├── vite.config.js                  # Configuración de Vite
│   ├── tailwind.config.js              # Configuración de Tailwind CSS
│   ├── postcss.config.js               # Configuración de PostCSS
│   ├── eslint.config.js                # Configuración de ESLint
│   ├── index.html                      # HTML de entrada de SPA
│   ├── README.md                       # README de Frontend
│   ├── public/                         # Archivos estáticos públicos
│   └── src/
│       ├── App.jsx                    # Componente principal de React
│       ├── App.css                    # Estilos de la aplicación
│       ├── main.jsx                   # Punto de entrada de React
│       ├── index.css                  # Estilos globales
│       └── assets/                    # Recursos estáticos
│
├── .gitignore                         # Archivos ignorados
├── install_env.bat                    # Script de configuración de entorno Windows
├── LICENSE                            # Licencia del proyecto
├── README.md                          # Documentación original
├── README_og.md                       # Copia de seguridad del README original
└── README.es.md                       # Documentación en español
```

## Stack Tecnológico

### Backend
- **Framework:** FastAPI 0.115.0 con Uvicorn
- **Computer Vision:** YOLOv26 (via Ultralytics), OpenCV
- **Deep Learning:** PyTorch 2.5.1 con soporte CUDA
- **Procesamiento de Datos:** NumPy, Pandas, scikit-learn
- **Visualización:** Matplotlib, Seaborn
- **Utilidades:** Roboflow (gestión de datasets), python-dotenv, PyYAML

### Frontend
- **Framework:** React 19.1.1 con Vite
- **Estilos:** Tailwind CSS con PostCSS/Autoprefixer
- **Componentes UI:** Lucide React (iconos), React QR Code
- **Linting:** ESLint

## Arquitectura

### Arquitectura del Modelo

El sistema usa YOLOv26 entrenado en datasets específicos de baloncesto con 5 clases de detección:

    0 - "Pelota"
    1 - "Pelota en Canasta"
    2 - "Jugador"
    3 - "Canasta"
    4 - "Jugador Lanzando"

#### Clases de Detección

| ID | Clase | Función | Confianza por Defecto |
|----|-------|---------|----------------------|
| 0 | Pelota | Rastrear baloncesto | 0.60 |
| 1 | Pelota en Canasta | Detectar tiro exitoso | 0.25 |
| 2 | Jugador | Identificar todos los jugadores | 0.70 |
| 3 | Canasta | Localizar aro | 0.70 |
| 4 | Jugador Lanzando | Identificar lanzador | 0.70 |

### Arquitectura del Pipeline

#### Procesamiento Fotograma a Fotograma

```
Entrada de Vídeo
    ↓
[Dividir en Fotogramas]
    ↓
[Detección YOLO] → Cajas delimitadoras + puntuaciones de confianza
    ↓
[Seguimiento] → Asignar IDs estables entre fotogramas
    ↓
[Estadísticas] → Calcular tiros, canastas y métricas por jugador
    ↓
[Visualización] → Dibujar detecciones, HUD, minimap, efectos
    ↓
[Escribir Fotograma] → Codificar de vuelta a vídeo
    ↓
Salida de Vídeo (MP4)
```

#### Componentes Principales

- **Config** - Gestión centralizada de configuración
- **GameStats** - Acumula contadores de tiros/canastas con desglose por jugador
- **Visualizer** - Utilidades de dibujo de OpenCV para HUD, efectos y visualización de cancha
- **MinimapRenderer** - Proyecta detecciones 2D a cancha usando transformación de homografía
- **VideoProcessor** - Pipeline principal: inferencia YOLO + codificación de vídeo de salida
- **AutoCleanup** - Hilo demonio para limpieza automática de archivos procesados antiguos

### Detalles Técnicos

#### Lógica de Enfriamiento Físico

**Problema**: Movimiento de lanzamiento único detectado en 10+ fotogramas consecutivos

**Solución**: Ventanas temporales de enfriamiento

```python
SHOT_COOLDOWN = 1.0     # segundos - espera después de detectar un lanzamiento
BASKET_COOLDOWN = 1.0   # segundos - espera después de puntuar una canasta
ANIMATION_DURATION = 1  # segundos - duración del efecto de pulso
```

**Cómo funciona:**
1. "Jugador Lanzando" detectado → inicia temporizador de enfriamiento de lanzamiento
2. Durante el enfriamiento, ignora nuevas detecciones de lanzamiento
3. Detección "Pelota en Canasta" → canasta cuenta solo si fuera del enfriamiento de canasta
4. Previene conteo duplicado entre límites de fotogramas

#### Estrategia de Aumento de Datos Personalizada

Aumento de datos optimizado para video de deportes:

```python
AUGMENTATION = {
    # Color e Iluminación
    'hsv_h': 0.015,      # Variación de matiz (cambios de color)
    'hsv_s': 0.4,        # Saturación (pelota naranja en cualquier luz)
    'hsv_v': 0.2,        # Valor/brillo (sombras, destellos)
    
    # Geometría y Posición
    'degrees': 10,       # Rotación (ángulos de cámara)
    'translate': 0.1,    # Traslación (movimiento de objetos)
    'scale': 0.5,        # Escalado (variación de profundidad)
    'shear': 2.0,        # Corte (perspectiva)
    
    # Avanzado
    'flipud': 0.5,       # Volteo vertical
    'fliplr': 0.5,       # Volteo horizontal
    'mosaic': 1.0,       # Aumento de mosaico
    'mixup': 0.15        # Mixup (fusionar imágenes)
}
```

**Por qué**: Los vídeos de deportes tienen alto movimiento, iluminación variada y escenas abarrotadas.

#### Proyección de Cancha

El minimap usa transformación de homografía para:
1. Mapear coordenadas de cancha 3D a fotograma de vídeo 2D
2. Proyectar jugadores detectados a vista de ojo de pájaro
3. Mostrar posiciones de jugadores en tiempo real y ubicaciones de lanzamientos

**Requisitos de Calibración:**
- 4+ puntos de esquina de cancha de imagen de referencia
- Coordenadas de imagen correspondientes
- Matriz de transformación generada (`homography.npy`)

## Entrenamiento y Rendimiento

### Entrenando un Modelo Personalizado

```bash
cd BE
.\venv\Scripts\activate
python train_model.py
```

El script de entrenamiento:
- Valida datasets automáticamente
- Verifica capacidades de hardware de GPU
- Implementa manejo correcto de interrupciones (Ctrl+C)
- Proporciona visualización de métricas personalizadas
- Utiliza hiperparámetros optimizados para rastreo de movimiento en baloncesto

**Configuración (editar `train_model.py`):**
- `PROJECT_NAME` - Directorio de salida para ejecuciones de entrenamiento
- `RUN_NAME` - Identificador para esta sesión de entrenamiento
- `DATASET_DIR` - Ruta a tu dataset de Roboflow
- `BASE_MODEL` - Modelo base (yolo26s.pt, yolo26m.pt, yolo26l.pt)
- `WORKERS` - Trabajadores del cargador de datos (0 para compatibilidad con Windows)
- `DEVICE` - Índice de GPU (0 para primera GPU)
- `SEED` - Semilla aleatoria para reproducibilidad

### Rendimiento del Modelo

#### Configuración de Entrenamiento

**Hardware Utilizado:**
- GPU: NVIDIA GTX 1060 6GB
- CPU: Intel i7-6700K
- RAM: 16GB DDR4
- Tiempo de Entrenamiento: ~48 horas

**Especificaciones del Modelo:**
- **Arquitectura**: YOLOv26s (variante pequeña)
- **Tamaño de Entrada**: 640×640 píxeles
- **Épocas**: 200
- **Tamaño de Lote**: 8
- **Optimizador**: AdamW con decaimiento coseno
- **Tasa de Aprendizaje**: 0.01 → 0.0005

#### Métricas de Rendimiento

**Precisión General** (Época 200):
- **mAP50**: 0.909 (Precisión Promedio Media at IoU ≥ 0.5)
- **mAP50-95**: 0.623 (Promedio entre umbrales IoU)
- **Precisión**: 0.878
- **Recall**: 0.861

**Rendimiento por Clase:**

| Clase | Precisión | Recall | mAP50 | Fortaleza |
|-------|-----------|--------|-------|-----------|
| Pelota | 0.80 | 0.88 | 0.847 | Detección robusta |
| Pelota en Canasta | 0.51 | 0.36 | 0.932 | Rara pero distintiva |
| Jugador | 0.86 | 0.85 | 0.928 | Rastreo excelente |
| Canasta | 0.91 | 0.91 | 0.966 | Muy confiable |
| Jugador Lanzando | 0.76 | 0.34 | 0.873 | Pose rara |

#### Curvas de Entrenamiento

Visualizaciones de métricas de entrenamiento disponibles en `FE/public/`:
- `results.png` - Curvas de pérdida (caja, clase, total)
- `confusionMatrix.png` - Precisión de clasificación
- `normalizedMatrix.png` - Matriz de confusión normalizada
- `PR_curve.png` - Tradeoff precisión-recall
- `P_curve.png` - Precisión vs umbral de confianza
- `R_curve.png` - Recall vs umbral de confianza
- `F1_curve.png` - Curva de optimización de puntuación F1

### Mejorando el Rendimiento del Modelo

Para mejor precisión con mejor hardware:

```python
# En la clase Config de train_model.py:
EPOCHS = 300              # Entrenamiento más largo
BATCH_SIZE = 16           # Lotes más grandes (RTX 3080+)
BASE_MODEL = "yolo26m.pt" # Modelo más grande (medio)
# o
BASE_MODEL = "yolo26l.pt" # Modelo grande (24GB+ VRAM)
```

**Técnicas Avanzadas:**
- Entrenamiento extendido (300+ épocas)
- Variantes de modelo más grandes (YOLOv26m, YOLOv26l)
- Aumento de datos adicional
- Ajuste fino en canchas específicas
- Métodos de conjunto

## Configuración

Edita los valores de configuración en `BE/app.py` (clase Config):

```python
class Config:
    # Configuración de rutas
    UPLOAD_DIR = Path(__file__).parent / "uploads"
    PROCESSED_DIR = Path(__file__).parent / "processed"
    MODEL_PATH = Path(__file__).parent / "basketball_training" / "yolo26s_5classes" / "weights" / "best.pt"
    
    # Restricciones de vídeo
    MAX_DURATION_SECONDS = 180   # Máximo 3 minutos
    TEST_MODE_DURATION = 15      # Duración del modo de prueba
    
    # Configuración de procesamiento
    CONFIDENCE_THRESHOLD = 0.5        # Umbral de confianza de YOLO
    IOU_THRESHOLD = 0.45              # Umbral NMS IoU
    MAX_DETECTIONS = 100              # Detecciones máximas por fotograma
    
    # Retención de archivos (limpieza automática)
    RETENTION_SECONDS = 600      # 10 minutos
    CLEANUP_INTERVAL = 60        # Verificar cada 60s
    
    # Física (en segundos)
    SHOT_COOLDOWN = 1.0          # Debounce de detección de lanzamiento
    BASKET_COOLDOWN = 1.0        # Debounce de detección de canasta
    ANIMATION_DURATION = 1       # Duración del efecto de pulso
    
    # Umbrales de detección (0.0-1.0)
    THRESHOLDS = {
        0: 0.60,   # Pelota
        1: 0.25,   # Pelota en Canasta
        2: 0.70,   # Jugador
        3: 0.70,   # Canasta
        4: 0.70    # Jugador Lanzando
    }
    
    # Dimensiones de cancha (cm)
    COURT_WIDTH_CM = 1500        # Ancho de media cancha FIBA
    COURT_HEIGHT_CM = 1400       # Profundidad de media cancha FIBA
```

### Ajustes para Diferentes Escenarios

**Iluminación Tenue:**
```python
THRESHOLDS = {
    0: 0.50,   # Umbral de pelota más bajo
    1: 0.20,   # Umbral de canasta más bajo
    2: 0.65,   # Jugador ligeramente más bajo
    3: 0.65,
    4: 0.65
}
```

**Cancha Interior Abarrotada:**
```python
SHOT_COOLDOWN = 0.8      # Detección de lanzamiento más rápida
BASKET_COOLDOWN = 1.2    # Debounce de canasta más lento
THRESHOLDS = {
    2: 0.75,   # Umbral de jugador más alto
    4: 0.75    # Pose de lanzamiento más estricta
}
```

## Optimización de Rendimiento

### Aceleración por GPU

El script `install_env.bat` detecta y configura automáticamente PyTorch para tu GPU NVIDIA:

- **CUDA 12.8+** → PyTorch con soporte cu128
- **CUDA 12.4-12.7** → PyTorch con soporte cu124
- **CUDA 11.8+** → PyTorch con soporte cu118
- **Sin GPU** → Versión PyTorch CPU

### Selección de Modelo

Elige el tamaño de modelo apropiado según tu hardware:

- `yolo26n.pt` - Nano (más rápido, menos preciso)
- `yolo26s.pt` - Pequeño (equilibrado, recomendado)
- `yolo26m.pt` - Medio (más lento, más preciso)

Ajusta `MODEL_PATH` en Config para cambiar de modelos.

## Desarrollo

### Desarrollo Backend

```bash
cd BE
.\venv\Scripts\activate
python app.py --reload  # Recarga automática en cambios de archivo (si usa Uvicorn directamente)
```

### Desarrollo Frontend

```bash
cd FE
npm run dev
npm run lint  # Ejecutar ESLint
npm run build # Compilar para producción
```

### Pruebas y Validación

**Linting:**
```bash
cd FE
npm run lint
```

**Compilación para Producción:**
```bash
cd FE
npm run build
```

## Detección de Errores

### Problemas de Instalación

**Python no encontrado:**
```
Asegurate de que Python está en la ruta del sistema
Ejecuta: python --version
```

**Falla la creación del entorno virtual:**
```
Verifica los permisos de escritura en el directorio BE
Elimina cualquier carpeta venv existente: rmdir /s /q BE\venv
Vuelve a intentar con el script de configuración
```

**Falla la detección de GPU de PyTorch:**
```
Verifica los drivers de NVIDIA: nvidia-smi
Comprueba compatibilidad de CUDA: https://www.nvidia.com/drivers
El sistema se retirará a CPU automáticamente
```

### Problemas de Procesamiento

**Error de falta de memoria:**
```
Reduce la resolución de fotogramas en Config
Usa un modelo más pequeño (yolo26n.pt o yolo26s.pt)
Habilita optimización de memoria de GPU en app.py
```

**Procesamiento lento:**
```
Habilita aceleración de GPU (verifica nvidia-smi)
Reduce la resolución del vídeo de entrada
Reduce ligeramente el umbral de confianza
```

### Problemas con Frontend

**No se puede conectar a la API:**
```
Verifica que el backend se está ejecutando: http://localhost:8000/docs
Comprueba la configuración CORS en app.py
Asegurate de que el frontend está en http://localhost:5173
```

## Archivos de Salida

### Vídeo Procesado
- Formato: MP4 (códec H.264)
- Contiene: Detecciones YOLO, IDs de seguimiento de jugadores, HUD con estadísticas de juego
- Minimap mostrando perspectiva de cancha
- Efectos visuales destacando objetos detectados

### CSV de Estadísticas
- Columnas: Número de fotograma, marca de tiempo, IDs de jugador, objetos detectados, lanzamientos, canastas
- Estadísticas agregadas por jugador
- Útil para análisis adicional en Excel o Python

## Limitaciones y Mejoras Futuras

**Limitaciones Actuales:**
- Requiere visibilidad clara de la cancha de baloncesto
- El rendimiento varía con las condiciones de iluminación
- El procesamiento en tiempo real depende de la disponibilidad de GPU

**Características Planificadas:**
- Rastreo de posesión de pelota
- Análisis de trayectoria de lanzamiento
- Interfaz de reentrenamiento de modelo basada en web

## Configuración CORS

El backend FastAPI está configurado para aceptar solicitudes desde el frontend React:

```python
CORSMiddleware(
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

Modifica `app.py` si ejecutas en puertos diferentes.

## Licencia

Este proyecto se publica bajo la **Licencia Pública General Affero de GNU v3.0 (AGPL-3.0)**.

**¿Por qué AGPL-3.0?**
Este proyecto integra **Ultralytics YOLO**, que está bajo licencia AGPL-3.0. Como obra derivada, Basketball AI Tracker hereda esta licencia para garantizar el cumplimiento total con los términos de código abierto de sus dependencias.

**Lo que significa para ti:**
- **Usar:** Puedes usar este software para propósitos personales, de investigación o comerciales.
- **Modificar:** Puedes modificar el código fuente.
- **Compartir:** Si distribuyes este software u lo alojas como servicio de red (SaaS), **debes** divulgar el código fuente de tu versión modificada bajo la misma licencia AGPL-3.0.

## 📜 Créditos

- **Trabajo Original**: sPappalard
- **Desarrollador**: DiegoGMD
- **Dataset**: [Roboflow Universe - Basketball Detection](https://universe.roboflow.com/basketball-6vyfz/basketball-detection-srfkd) + Dataset Personalizado
- **Frameworks**: Ultralytics YOLO, FastAPI, React

---

**Última Actualización:** 2026-05-26  
**Versión:** 1.0  
**Estado:** Desarrollo Activo
