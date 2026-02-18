# 🔮 Churn Prediction System - Telecom Customer Analytics

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-00ADD8?style=for-the-badge&logo=xgboost&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.3+-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.30-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)

**Sistema completo de predicción de churn con Machine Learning, API REST y Dashboard interactivo**

[Ver Demo](#-demo) · [Documentación](#-documentación) · [Instalación](#-instalación-rápida)

</div>

---

## 📊 Resultados del Modelo

<div align="center">

| Métrica | Score | 
|---------|-------|
| **ROC-AUC** | 0.88 🎯 |
| **Accuracy** | 85.2% ✅ |
| **Precision** | 83.5% 📊 |
| **Recall** | 79.8% 🎪 |
| **F1-Score** | 81.6% 🎭 |

</div>

---

## 🎯 Descripción del Proyecto

Sistema end-to-end de Machine Learning para predecir la probabilidad de que un cliente de telecomunicaciones abandone el servicio (churn). 

**Características principales:**
- 📈 Análisis Exploratorio completo con visualizaciones interactivas
- 🔧 Feature Engineering avanzado (10+ variables derivadas)
- 🤖 Múltiples modelos ML con optimización de hiperparámetros
- ⚡ API REST con FastAPI para predicciones en tiempo real
- 🎨 Dashboard interactivo con Streamlit
- 🐳 Containerización con Docker
- 📊 Interpretabilidad con SHAP values

---

## 🏗️ Arquitectura del Proyecto

```
churn-prediction/
├── 📊 data/                    # Datos
├── 📓 notebooks/               # Análisis y modelado
├── 🤖 models/                  # Modelos entrenados
├── ⚡ api/                     # FastAPI REST API
├── 🎨 streamlit_app/          # Dashboard
└── 🐳 Docker/                 # Containerización
```

---

## 💻 Instalación Rápida

```bash
# 1. Clonar repositorio
git clone https://github.com/tu-usuario/churn-prediction.git
cd churn-prediction

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar notebooks (01-05)
jupyter notebook

# 4. Ejecutar API
cd api && uvicorn main:app --reload

# 5. Ejecutar Dashboard
cd streamlit_app && streamlit run app.py
```

---

## 🚀 Características

### 🎯 Predicción de Churn
- Predicción individual y en lote
- Clasificación de riesgo (Bajo/Medio/Alto/Crítico)
- Recomendaciones personalizadas

### 📊 Dashboard Interactivo
- Visualizaciones con Plotly
- KPIs en tiempo real
- Análisis de tendencias

### ⚡ API REST
- Endpoints RESTful
- Validación con Pydantic
- Documentación Swagger

---

## 📈 Metodología

1. **EDA**: Análisis exploratorio completo
2. **Feature Engineering**: 10+ variables derivadas
3. **Modelado**: LR → RF → GB → XGBoost
4. **Optimización**: RandomizedSearchCV
5. **Evaluación**: Cross-validation, SHAP, calibración

---

## 🛠️ Stack Tecnológico

**ML:** Python, Pandas, NumPy, Scikit-learn, XGBoost  
**Visualización:** Matplotlib, Seaborn, Plotly  
**API:** FastAPI, Uvicorn, Pydantic  
**Dashboard:** Streamlit  
**DevOps:** Docker, Docker Compose

---

## 📄 Licencia

MIT License - Ver [LICENSE](LICENSE) para más detalles.

---

## 👤 Autor

**PAULINO TAMBORINI**
- LinkedIn: [https://www.linkedin.com/in/paulino-tamborini-41a60b272/]
- GitHub: [https://github.com/paulinotamborini]
- Email: paulinotamborini044@gmail.com

---

<div align="center">

**⭐ Si este proyecto te fue útil, considera darle una estrella! ⭐**

Made with ❤️ and ☕

</div>
