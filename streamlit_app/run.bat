@echo off
REM Script para iniciar la app de Streamlit
REM Windows Batch Script

echo.
echo ================================================
echo    🔮 Churn Prediction Dashboard
echo    Iniciando aplicación Streamlit...
echo ================================================
echo.

REM Verificar que Python está instalado
python --version >nul 2>&1
if errorlevel 1 (
    echo ❌ ERROR: Python no está instalado o no está en el PATH
    echo.
    echo Por favor instala Python desde: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo ✅ Python detectado
echo.

REM Verificar que los modelos existen
if not exist "..\models\xgboost_optimized.pkl" (
    echo ⚠️  ADVERTENCIA: Modelo no encontrado
    echo    Ubicación esperada: ..\models\xgboost_optimized.pkl
    echo.
    echo    Por favor ejecuta los notebooks 03 y 05 primero para generar los modelos.
    echo.
    pause
)

REM Verificar si streamlit está instalado
python -c "import streamlit" 2>nul
if errorlevel 1 (
    echo 📦 Streamlit no está instalado. Instalando dependencias...
    echo.
    pip install -r requirements.txt
    if errorlevel 1 (
        echo ❌ ERROR: No se pudieron instalar las dependencias
        pause
        exit /b 1
    )
    echo.
    echo ✅ Dependencias instaladas correctamente
    echo.
)

echo 🚀 Iniciando Streamlit...
echo.
echo    La aplicación se abrirá automáticamente en tu navegador.
echo    URL: http://localhost:8501
echo.
echo    Presiona Ctrl+C para detener el servidor.
echo.
echo ================================================
echo.

REM Iniciar Streamlit
streamlit run app.py

pause
