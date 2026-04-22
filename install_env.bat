@echo off
setlocal enabledelayedexpansion

echo ==============================================================
echo   Instalador de entorno YOLO - Computer Vision
echo ==============================================================
echo.

REM --- Comprobar que Python esta disponible ---------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] Python no encontrado. Asegurate de tener Python en el PATH.
    pause & exit /b 1
)

REM --- Ir al directorio BE --------------------------------------
echo [INFO] Entrando en el directorio BE...
cd /d "%~dp0BE"
if errorlevel 1 (
    echo [ERROR] No se encontro la carpeta BE junto al .bat.
    echo         Asegurate de que install_env.bat este en la raiz del proyecto.
    pause & exit /b 1
)

REM --- Borrar venv anterior si existe ---------------------------
if exist "venv" (
    echo [INFO] Borrando entorno virtual anterior...
    rmdir /s /q venv
    echo [OK] Entorno virtual anterior eliminado.
)

REM --- Crear entorno virtual ------------------------------------
echo [INFO] Creando entorno virtual...
python -m venv venv
if errorlevel 1 (
    echo [ERROR] No se pudo crear el entorno virtual.
    pause & exit /b 1
)
echo [OK] Entorno virtual creado.

REM --- Activar el entorno virtual -------------------------------
echo [INFO] Activando entorno virtual...
call venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] No se pudo activar el entorno virtual.
    pause & exit /b 1
)
echo [OK] Entorno virtual activo.
echo.

REM --- Actualizar pip -------------------------------------------
echo [INFO] Actualizando pip...
python -m pip install --upgrade pip --quiet
echo [OK] pip actualizado.
echo.

REM --- Detectar GPU y version de CUDA ---------------------------
set CUDA_VER=cpu
set TORCH_INDEX=https://download.pytorch.org/whl/cpu
set CUDA_MAJOR=0
set CUDA_MINOR=0

nvidia-smi >nul 2>&1
if errorlevel 1 (
    echo [AVISO] No se detecto GPU NVIDIA. Instalando version CPU de PyTorch.
    goto :install_torch
)

for /f "tokens=9" %%v in ('nvidia-smi ^| findstr /i "CUDA Version"') do (
    set RAW_VER=%%v
)

if not defined RAW_VER (
    echo [AVISO] No se pudo leer la version de CUDA. Usando CPU.
    goto :install_torch
)

for /f "tokens=1,2 delims=." %%a in ("!RAW_VER!") do (
    set CUDA_MAJOR=%%a
    set CUDA_MINOR=%%b
)

echo [INFO] Driver CUDA detectado: !RAW_VER!

REM --- Elegir el indice de PyTorch segun CUDA -------------------
REM     Regla: usar siempre la build mas reciente que soporte la GPU.
REM     cu128 = compatible con CUDA 12.8+ y drivers 13.x
REM     cu124 = compatible con CUDA 12.4 - 12.7
REM     cu121 = compatible con CUDA 12.1 - 12.3
REM     cu118 = compatible con CUDA 11.8

if !CUDA_MAJOR! GEQ 13 (
    set CUDA_VER=cu128
    set TORCH_INDEX=https://download.pytorch.org/whl/cu128
    goto :install_torch
)
if !CUDA_MAJOR! EQU 12 if !CUDA_MINOR! GEQ 8 (
    set CUDA_VER=cu128
    set TORCH_INDEX=https://download.pytorch.org/whl/cu128
    goto :install_torch
)
if !CUDA_MAJOR! EQU 12 if !CUDA_MINOR! GEQ 4 (
    set CUDA_VER=cu124
    set TORCH_INDEX=https://download.pytorch.org/whl/cu128
    goto :install_torch
)
if !CUDA_MAJOR! EQU 12 if !CUDA_MINOR! GEQ 1 (
    set CUDA_VER=cu121
    set TORCH_INDEX=https://download.pytorch.org/whl/cu121
    goto :install_torch
)
if !CUDA_MAJOR! EQU 11 if !CUDA_MINOR! GEQ 8 (
    set CUDA_VER=cu118
    set TORCH_INDEX=https://download.pytorch.org/whl/cu118
    goto :install_torch
)

echo [AVISO] CUDA !RAW_VER! sin build compatible. Usando CPU.

:install_torch
echo.
echo [INFO] Indice PyTorch seleccionado: %TORCH_INDEX%
echo.
echo --------------------------------------------------------------
echo  [1/3] Instalando PyTorch (ultima version disponible para %CUDA_VER%)
echo --------------------------------------------------------------

REM --- Instalar la version mas reciente disponible (sin fijar version) --
pip install torch torchvision torchaudio --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de PyTorch.
    pause & exit /b 1
)

REM --- Mostrar que version se instalo ---------------------------
echo.
for /f "tokens=2" %%v in ('pip show torch ^| findstr /i "Version"') do (
    echo [OK] PyTorch instalado: %%v
)
echo.

echo --------------------------------------------------------------
echo  [2/3] Instalando el resto de dependencias
echo --------------------------------------------------------------
pip install ^
    fastapi==0.115.0 ^
    "uvicorn[standard]==0.32.0" ^
    python-multipart==0.0.12 ^
    ultralytics ^
    opencv-python==4.10.0.84 ^
    numpy==1.26.4 ^
    "pandas>=2.0.0" ^
    "scikit-learn>=1.3.0" ^
    "Pillow>=10.0.0" ^
    "matplotlib>=3.8.0" ^
    "seaborn>=0.13.0" ^
    roboflow==1.1.47 ^
    PyYAML==6.0.2 ^
    python-dotenv==1.0.0^
    lapx==0.5.5

if errorlevel 1 (
    echo [ERROR] Fallo la instalacion de dependencias.
    pause & exit /b 1
)

echo.
echo --------------------------------------------------------------
echo  [3/3] Verificando GPU y PyTorch
echo --------------------------------------------------------------
python -c "import torch; v=torch.__version__; cuda=torch.cuda.is_available(); dev=torch.cuda.get_device_name(0) if cuda else 'CPU - sin GPU'; print('  torch    : '+v); print('  CUDA OK  : '+str(cuda)); print('  Device   : '+dev)"

if errorlevel 1 (
    echo.
    echo [AVISO] PyTorch instalado pero CUDA no disponible.
    echo         Comprueba los drivers de NVIDIA en: https://www.nvidia.com/drivers
) else (
    echo.
    echo [OK] Todo listo! GPU detectada y funcionando.
)

echo.
echo Para activar el entorno manualmente en el futuro:
echo   BE\venv\Scripts\activate
echo.
pause