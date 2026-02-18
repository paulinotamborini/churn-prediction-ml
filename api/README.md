# 🚀 Churn Prediction API

API REST construida con FastAPI para predecir la probabilidad de churn de clientes de telecomunicaciones.

## 📋 Características

- ✅ Predicción individual de churn
- ✅ Predicción en lote (hasta 100 clientes)
- ✅ Validación automática de entrada con Pydantic
- ✅ Documentación interactiva (Swagger UI)
- ✅ Logging de predicciones
- ✅ Cálculo de nivel de riesgo
- ✅ CORS habilitado para desarrollo

## 🛠️ Instalación

### 1. Instalar dependencias

```powershell
# Crear entorno virtual (opcional pero recomendado)
python -m venv venv
.\venv\Scripts\activate

# Instalar dependencias
pip install -r api/requirements.txt
```

### 2. Verificar que los modelos existen

Asegúrate de tener estos archivos en la carpeta `models/`:
- `xgboost_optimized.pkl` - Modelo XGBoost optimizado
- `scaler.pkl` - Scaler para normalización

Si no los tienes, ejecuta los notebooks 03 y 05 primero.

## 🚀 Uso

### Iniciar el servidor

```powershell
# Opción 1: Desarrollo (con auto-reload)
uvicorn api.main:app --reload

# Opción 2: Producción
uvicorn api.main:app --host 0.0.0.0 --port 8000
```

El servidor estará disponible en: `http://localhost:8000`

### Documentación Interactiva

Una vez que el servidor esté corriendo, accede a:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## 📡 Endpoints

### 1. Health Check

```http
GET /
```

**Response:**
```json
{
  "status": "healthy",
  "message": "Churn Prediction API is running",
  "model_loaded": true,
  "timestamp": "2024-02-18T10:30:00"
}
```

### 2. Model Info

```http
GET /info
```

**Response:**
```json
{
  "model_info": {
    "model_type": "XGBoost Classifier",
    "version": "1.0.0",
    "trained_date": "2024-02-18",
    "features_count": 30,
    "accuracy": 0.85,
    "roc_auc": 0.88
  },
  "status": "ready"
}
```

### 3. Predicción Individual

```http
POST /predict
Content-Type: application/json
```

**Request Body:**
```json
{
  "tenure": 12,
  "MonthlyCharges": 70.0,
  "TotalCharges": 840.0,
  "gender": "Male",
  "SeniorCitizen": 0,
  "Partner": "Yes",
  "Dependents": "No",
  "PhoneService": "Yes",
  "PaperlessBilling": "Yes",
  "MultipleLines": "Yes",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "Yes",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "Yes",
  "StreamingMovies": "Yes",
  "Contract": "Month-to-month",
  "PaymentMethod": "Electronic check"
}
```

**Response:**
```json
{
  "churn_probability": 0.7532,
  "churn_prediction": "Churn",
  "risk_level": "High",
  "confidence": 0.5064,
  "timestamp": "2024-02-18T10:35:00"
}
```

### 4. Predicción en Lote

```http
POST /predict_batch
Content-Type: application/json
```

**Request Body:**
```json
{
  "customers": [
    {
      "tenure": 12,
      "MonthlyCharges": 70.0,
      ...
    },
    {
      "tenure": 48,
      "MonthlyCharges": 55.0,
      ...
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "churn_probability": 0.7532,
      "churn_prediction": "Churn",
      "risk_level": "High",
      "confidence": 0.5064,
      "timestamp": "2024-02-18T10:35:00"
    },
    {
      "churn_probability": 0.2145,
      "churn_prediction": "No Churn",
      "risk_level": "Low",
      "confidence": 0.5710,
      "timestamp": "2024-02-18T10:35:00"
    }
  ],
  "total_customers": 2,
  "high_risk_count": 1,
  "timestamp": "2024-02-18T10:35:00"
}
```

## 🧪 Testing

### Ejecutar tests

```powershell
# Asegúrate de que el servidor esté corriendo primero
# En otra terminal:
python api/test_api.py
```

### Probar con curl (Windows PowerShell)

```powershell
# Health check
curl http://localhost:8000/

# Predicción
curl -X POST http://localhost:8000/predict `
  -H "Content-Type: application/json" `
  -d '{
    "tenure": 12,
    "MonthlyCharges": 70.0,
    "TotalCharges": 840.0,
    "gender": "Male",
    "SeniorCitizen": 0,
    "Partner": "Yes",
    "Dependents": "No",
    "PhoneService": "Yes",
    "PaperlessBilling": "Yes",
    "MultipleLines": "Yes",
    "InternetService": "Fiber optic",
    "OnlineSecurity": "No",
    "OnlineBackup": "Yes",
    "DeviceProtection": "No",
    "TechSupport": "No",
    "StreamingTV": "Yes",
    "StreamingMovies": "Yes",
    "Contract": "Month-to-month",
    "PaymentMethod": "Electronic check"
  }'
```

### Probar con Python

```python
import requests

# Hacer predicción
response = requests.post(
    "http://localhost:8000/predict",
    json={
        "tenure": 12,
        "MonthlyCharges": 70.0,
        "TotalCharges": 840.0,
        "gender": "Male",
        "SeniorCitizen": 0,
        "Partner": "Yes",
        "Dependents": "No",
        "PhoneService": "Yes",
        "PaperlessBilling": "Yes",
        "MultipleLines": "Yes",
        "InternetService": "Fiber optic",
        "OnlineSecurity": "No",
        "OnlineBackup": "Yes",
        "DeviceProtection": "No",
        "TechSupport": "No",
        "StreamingTV": "Yes",
        "StreamingMovies": "Yes",
        "Contract": "Month-to-month",
        "PaymentMethod": "Electronic check"
    }
)

print(response.json())
```

## 📊 Niveles de Riesgo

| Probabilidad | Nivel de Riesgo | Descripción |
|--------------|-----------------|-------------|
| 0.0 - 0.3    | **Low**         | Cliente estable, baja probabilidad de churn |
| 0.3 - 0.5    | **Medium**      | Cliente en zona de atención |
| 0.5 - 0.7    | **High**        | Cliente en riesgo, requiere intervención |
| 0.7 - 1.0    | **Critical**    | Cliente con muy alta probabilidad de churn |

## 🔒 Seguridad (Para Producción)

Para producción, considera agregar:

1. **Autenticación JWT**
   ```python
   from fastapi.security import HTTPBearer
   ```

2. **Rate Limiting**
   ```python
   from slowapi import Limiter
   ```

3. **HTTPS**
   ```python
   uvicorn api.main:app --ssl-keyfile=key.pem --ssl-certfile=cert.pem
   ```

4. **Variables de entorno**
   ```python
   from pydantic_settings import BaseSettings
   ```

## 📝 Estructura del Proyecto

```
api/
├── main.py              # API principal
├── requirements.txt     # Dependencias
├── test_api.py         # Tests
└── README.md           # Esta documentación

models/
├── xgboost_optimized.pkl
└── scaler.pkl
```

## 🐛 Troubleshooting

### Error: "Model not found"

**Solución:** Ejecuta los notebooks 03 y 05 para generar los modelos.

### Error: "Cannot connect to API"

**Solución:** Verifica que el servidor esté corriendo con `uvicorn api.main:app --reload`

### Error: "Validation error"

**Solución:** Verifica que todos los campos requeridos estén presentes y tengan el formato correcto.

## 📚 Recursos

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [Pydantic Documentation](https://docs.pydantic.dev/)
- [Uvicorn Documentation](https://www.uvicorn.org/)

## 👤 Autor

PAULINO TAMBORINI - Proyecto de Churn Prediction

## 📄 Licencia

MIT
