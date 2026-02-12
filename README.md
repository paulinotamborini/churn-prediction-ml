# 🎯 Churn Prediction ML Project

Sistema completo de Machine Learning para predecir abandono de clientes en telecomunicaciones.

## 📊 Objetivo del Proyecto

Desarrollar un modelo de clasificación que identifique clientes con alto riesgo de churn, permitiendo estrategias de retención proactivas.

## 🛠️ Stack Tecnológico

- **Python 3.9+**
- **Pandas, NumPy** - Análisis de datos
- **Scikit-learn, XGBoost** - Machine Learning
- **FastAPI** - API REST
- **Streamlit** - Dashboard interactivo
- **Docker** - Containerización
- **MLflow** - Experiment tracking

## 🚀 Configuración del Entorno
```bash
# Clonar repositorio
git clone https://github.com/TU-USUARIO/churn-prediction-ml.git
cd churn-prediction-ml

# Crear y activar entorno virtual
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate   # Windows

# Instalar dependencias
pip install -r requirements.txt
```

## 📈 Progreso del Proyecto

- [x] Configuración inicial
- [ ] Análisis exploratorio de datos
- [ ] Feature engineering
- [ ] Desarrollo de modelos
- [ ] Evaluación y optimización
- [ ] Deployment
- [ ] Dashboard

## 📝 Autor

Paulino Tamborini - Data Scientist

---

**Fecha de inicio:** [Fecha de hoy]
```

### 6.2 Crear .gitignore (si no se creó automáticamente)

Crea un archivo `.gitignore` con esto:
```
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
venv/
env/

# Jupyter
.ipynb_checkpoints

# Data
data/raw/*.csv
data/processed/*.csv
*.pkl
*.joblib

# Models
models/saved_models/*.pkl
models/saved_models/*.h5

# MLflow
mlruns/
mlartifacts/

# IDEs
.vscode/
.idea/
*.swp
*.swo

# OS
.DS_Store
Thumbs.db

# Environment variables
.env