# 📁 Estructura del Proyecto - Churn Prediction

## Estructura Recomendada

```
churn-prediction/
│
├── data/
│   ├── raw/                          # Datos originales sin procesar
│   │   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│   │
│   └── processed/                    # Datos procesados y listos para modelado
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── X_train_scaled.csv
│       ├── X_test_scaled.csv
│       ├── y_train.csv
│       └── y_test.csv
│
├── notebooks/                        # Jupyter notebooks para análisis
│   ├── 01_eda.ipynb
│   ├── 02_categorical_analysis.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_modeling.ipynb
│   └── 05_optimization.ipynb (próximo)
│
├── models/                           # Modelos entrenados
│   ├── logistic_regression_model.pkl
│   ├── random_forest_model.pkl
│   ├── gradient_boosting_model.pkl
│   ├── xgboost_model.pkl
│   ├── scaler.pkl
│   └── model_comparison_results.csv
│
├── src/                              # Código fuente (opcional para producción)
│   ├── __init__.py
│   ├── preprocessing.py
│   ├── features.py
│   ├── model.py
│   └── predict.py
│
├── reports/                          # Reportes y visualizaciones
│   └── figures/
│
├── requirements.txt                  # Dependencias del proyecto
├── README.md                         # Documentación del proyecto
└── .gitignore                        # Archivos a ignorar en git

```

## 📋 Descripción de Carpetas

### 📂 `data/`
- **`raw/`**: Datos originales sin modificar. Nunca edites estos archivos.
- **`processed/`**: Datos limpios, transformados y listos para machine learning.

### 📓 `notebooks/`
- Notebooks de Jupyter para análisis exploratorio, experimentación y visualización.
- Numerados para seguir el flujo del proyecto.

### 🤖 `models/`
- Modelos entrenados guardados como archivos `.pkl`
- Transformadores (scalers, encoders, etc.)
- Resultados de comparación de modelos

### 💻 `src/`
- Código Python reutilizable y modularizado
- Scripts para poner el modelo en producción
- Útil cuando quieres desplegar el modelo como API o servicio

### 📊 `reports/`
- Reportes finales, presentaciones
- Gráficos y visualizaciones para stakeholders

## 🎯 Flujo de Trabajo

1. **EDA** → `notebooks/01_eda.ipynb`
   - Carga datos desde `data/raw/`
   - Análisis exploratorio inicial

2. **Feature Engineering** → `notebooks/03_feature_engineering.ipynb`
   - Crea nuevas variables
   - Guarda datasets procesados en `data/processed/`
   - Guarda scaler en `models/`

3. **Modelado** → `notebooks/04_modeling.ipynb`
   - Carga datos desde `data/processed/`
   - Entrena múltiples modelos
   - Guarda modelos en `models/`

4. **Optimización** → `notebooks/05_optimization.ipynb` (próximo)
   - Hyperparameter tuning
   - Interpretabilidad

## 🚀 Ventajas de Esta Estructura

✅ **Organización**: Cada tipo de archivo en su lugar
✅ **Reproducibilidad**: Fácil seguir el flujo del proyecto
✅ **Colaboración**: Otros pueden entender rápidamente el proyecto
✅ **Escalabilidad**: Fácil agregar nuevos componentes
✅ **Git-friendly**: Puedes hacer `.gitignore` de `data/` y `models/` para no subir archivos pesados

## 📝 Comandos Útiles

```bash
# Crear estructura de carpetas
mkdir -p data/raw data/processed models notebooks src reports/figures

# Mover datos crudos
mv WA_Fn-UseC_-Telco-Customer-Churn.csv data/raw/

# Git - ignorar archivos pesados
echo "data/" >> .gitignore
echo "models/*.pkl" >> .gitignore
echo ".ipynb_checkpoints/" >> .gitignore
```

## 🔄 Próximos Pasos

1. ✅ EDA básico
2. ✅ Análisis categórico
3. ✅ Feature Engineering
4. ✅ Modelado
5. ⏳ Optimización de hiperparámetros
6. ⏳ Interpretabilidad (SHAP)
7. ⏳ Deployment (API con FastAPI/Flask)

---

**Nota**: Esta es la estructura estándar de la industria para proyectos de Data Science y Machine Learning.
