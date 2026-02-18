# 🔮 Churn Prediction - Telco Customer

Proyecto de Machine Learning para predecir la probabilidad de que un cliente abandone el servicio (churn).

## 📊 Dataset

- **Fuente**: Telco Customer Churn Dataset
- **Registros**: ~7,000 clientes
- **Variables**: 21 features (demográficas, servicios, facturación)
- **Target**: Churn (Yes/No)

## 🎯 Objetivo

Desarrollar un modelo predictivo para identificar clientes con alta probabilidad de churn, permitiendo:
- Estrategias proactivas de retención
- Segmentación de clientes en riesgo
- Optimización de recursos de marketing

## 📁 Estructura del Proyecto

```
churn-prediction/
├── data/
│   ├── raw/              # Datos originales
│   └── processed/        # Datos procesados
├── notebooks/            # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_categorical_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   └── 04_modeling.ipynb
├── models/               # Modelos entrenados
├── src/                  # Código fuente
└── reports/              # Reportes y visualizaciones
```

## 🔄 Flujo de Trabajo

1. **EDA** - Análisis exploratorio de datos
2. **Análisis Categórico** - Relaciones entre variables y churn
3. **Feature Engineering** - Creación de nuevas variables
4. **Modelado** - Entrenamiento de modelos ML

## 🤖 Modelos Implementados

- Logistic Regression (baseline)
- Random Forest
- Gradient Boosting
- XGBoost

## 📈 Métricas Principales

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC

## 🛠️ Tecnologías

- Python 3.8+
- Pandas, NumPy
- Scikit-learn
- XGBoost
- Matplotlib, Seaborn, Plotly
- Jupyter Notebook

## 🚀 Instalación

```bash
# Clonar repositorio
git clone <tu-repo>
cd churn-prediction

# Instalar dependencias
pip install -r requirements.txt

# Ejecutar notebooks
jupyter notebook
```

## 📊 Resultados

(Completar después del análisis)

## 👤 Autor

Tu nombre

## 📝 Licencia

MIT
