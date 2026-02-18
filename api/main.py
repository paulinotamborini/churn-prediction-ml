"""
API de Predicción de Churn - FastAPI
Autor: Tu nombre
Fecha: 2024

Esta API permite predecir la probabilidad de churn de clientes
utilizando el modelo XGBoost entrenado.

Endpoints:
    - GET /: Health check
    - GET /info: Información del modelo
    - POST /predict: Predicción individual
    - POST /predict_batch: Predicción en lote
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear app FastAPI
app = FastAPI(
    title="Churn Prediction API",
    description="API para predecir la probabilidad de churn de clientes de telecomunicaciones",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Configurar CORS (permitir requests desde cualquier origen en desarrollo)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Cargar modelo y scaler al inicio
try:
    MODEL = joblib.load(r"C:\Proyecto Churn\churn-prediction-ml\notebooks\models\xgboost_optimized.pkl")
    SCALER = joblib.load(r'C:\Proyecto Churn\churn-prediction-ml\notebooks\models\scaler.pkl')
    logger.info("✅ Modelo y scaler cargados exitosamente")
except FileNotFoundError as e:
    logger.error(f"❌ Error cargando modelo: {e}")
    MODEL = None
    SCALER = None

# Metadata del modelo
MODEL_INFO = {
    "model_type": "XGBoost Classifier",
    "version": "1.0.0",
    "trained_date": "2024-02-18",
    "features_count": 30,
    "accuracy": 0.85,
    "roc_auc": 0.88
}


# =====================================================
# Modelos de Datos (Pydantic)
# =====================================================

class CustomerInput(BaseModel):
    """Esquema de entrada para un cliente individual"""
    
    # Variables numéricas
    tenure: int = Field(..., ge=0, le=72, description="Meses como cliente (0-72)")
    MonthlyCharges: float = Field(..., ge=0, le=150, description="Cargo mensual ($)")
    TotalCharges: float = Field(..., ge=0, description="Cargo total acumulado ($)")
    
    # Variables binarias (Yes/No)
    gender: str = Field(..., description="Género: Male o Female")
    SeniorCitizen: int = Field(..., ge=0, le=1, description="Es adulto mayor: 0 o 1")
    Partner: str = Field(..., description="Tiene pareja: Yes o No")
    Dependents: str = Field(..., description="Tiene dependientes: Yes o No")
    PhoneService: str = Field(..., description="Tiene servicio telefónico: Yes o No")
    PaperlessBilling: str = Field(..., description="Facturación sin papel: Yes o No")
    
    # Variables categóricas con múltiples opciones
    MultipleLines: str = Field(..., description="Múltiples líneas: Yes, No, No phone service")
    InternetService: str = Field(..., description="Servicio de internet: DSL, Fiber optic, No")
    OnlineSecurity: str = Field(..., description="Seguridad online: Yes, No, No internet service")
    OnlineBackup: str = Field(..., description="Backup online: Yes, No, No internet service")
    DeviceProtection: str = Field(..., description="Protección de dispositivo: Yes, No, No internet service")
    TechSupport: str = Field(..., description="Soporte técnico: Yes, No, No internet service")
    StreamingTV: str = Field(..., description="TV streaming: Yes, No, No internet service")
    StreamingMovies: str = Field(..., description="Películas streaming: Yes, No, No internet service")
    Contract: str = Field(..., description="Tipo de contrato: Month-to-month, One year, Two year")
    PaymentMethod: str = Field(..., description="Método de pago: Electronic check, Mailed check, Bank transfer (automatic), Credit card (automatic)")
    
    class Config:
        schema_extra = {
            "example": {
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
        }
    
    @validator('gender')
    def validate_gender(cls, v):
        if v not in ['Male', 'Female']:
            raise ValueError('gender debe ser Male o Female')
        return v
    
    @validator('Partner', 'Dependents', 'PhoneService', 'PaperlessBilling')
    def validate_yes_no(cls, v):
        if v not in ['Yes', 'No']:
            raise ValueError('Debe ser Yes o No')
        return v
    
    @validator('Contract')
    def validate_contract(cls, v):
        if v not in ['Month-to-month', 'One year', 'Two year']:
            raise ValueError('Contract debe ser Month-to-month, One year o Two year')
        return v


class PredictionOutput(BaseModel):
    """Esquema de salida para predicción"""
    churn_probability: float = Field(..., description="Probabilidad de churn (0-1)")
    churn_prediction: str = Field(..., description="Predicción: Churn o No Churn")
    risk_level: str = Field(..., description="Nivel de riesgo: Low, Medium, High, Critical")
    confidence: float = Field(..., description="Confianza en la predicción (0-1)")
    timestamp: str = Field(..., description="Timestamp de la predicción")
    
    class Config:
        schema_extra = {
            "example": {
                "churn_probability": 0.75,
                "churn_prediction": "Churn",
                "risk_level": "High",
                "confidence": 0.95,
                "timestamp": "2024-02-18T10:30:00"
            }
        }


class BatchPredictionInput(BaseModel):
    """Esquema de entrada para predicción en lote"""
    customers: List[CustomerInput] = Field(..., max_items=100, description="Lista de clientes (máx 100)")


class BatchPredictionOutput(BaseModel):
    """Esquema de salida para predicción en lote"""
    predictions: List[PredictionOutput]
    total_customers: int
    high_risk_count: int
    timestamp: str


# =====================================================
# Funciones Auxiliares
# =====================================================

def preprocess_input(data: CustomerInput) -> pd.DataFrame:
    """
    Preprocesa los datos de entrada para que coincidan con el formato del modelo
    
    Args:
        data: CustomerInput con los datos del cliente
        
    Returns:
        DataFrame con las features procesadas
    """
    # Convertir a diccionario
    customer_dict = data.dict()
    
    # Crear DataFrame
    df = pd.DataFrame([customer_dict])
    
    # Feature Engineering (mismo que en entrenamiento)
    
    # 1. AvgMonthlyCharges
    df['AvgMonthlyCharges'] = df.apply(
        lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'],
        axis=1
    )
    
    # 2. ChargeRatio
    df['ChargeRatio'] = df.apply(
        lambda row: row['TotalCharges'] / (row['MonthlyCharges'] * row['tenure']) if row['tenure'] > 0 else 1,
        axis=1
    )
    
    # 3. TotalServices
    service_cols = ['PhoneService', 'MultipleLines', 'InternetService', 
                    'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 
                    'TechSupport', 'StreamingTV', 'StreamingMovies']
    
    df['TotalServices'] = 0
    for col in service_cols:
        if col in df.columns:
            df['TotalServices'] += (~df[col].isin(['No', 'No phone service', 'No internet service'])).astype(int)
    
    # 4. HasProtectionServices
    protection_cols = ['OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport']
    df['HasProtectionServices'] = 0
    for col in protection_cols:
        if col in df.columns:
            df['HasProtectionServices'] += (df[col] == 'Yes').astype(int)
    
    df['HasAnyProtection'] = (df['HasProtectionServices'] > 0).astype(int)
    
    # 5. HasStreamingServices
    streaming_cols = ['StreamingTV', 'StreamingMovies']
    df['HasStreamingServices'] = 0
    for col in streaming_cols:
        if col in df.columns:
            df['HasStreamingServices'] += (df[col] == 'Yes').astype(int)
    
    df['HasAnyStreaming'] = (df['HasStreamingServices'] > 0).astype(int)
    
    # 6. LongTermAutoPayment
    df['LongTermAutoPayment'] = (
        (df['Contract'].isin(['One year', 'Two year'])) & 
        (df['PaymentMethod'].isin(['Bank transfer (automatic)', 'Credit card (automatic)']))
    ).astype(int)
    
    # 7. HighRiskCustomer
    df['HighRiskCustomer'] = (
        (df['Contract'] == 'Month-to-month') & 
        (df['PaymentMethod'].isin(['Electronic check', 'Mailed check']))
    ).astype(int)
    
    # 8. HasPaperlessBilling
    df['HasPaperlessBilling'] = (df['PaperlessBilling'] == 'Yes').astype(int)
    
    # Encoding de variables binarias
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        if col in df.columns:
            df[col + '_encoded'] = (df[col] == 'Yes').astype(int)
    
    # Gender encoding
    df['Gender_encoded'] = (df['gender'] == 'Male').astype(int)
    
    # One-hot encoding para categóricas
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaymentMethod']
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
    
    # Eliminar columnas originales que no necesita el modelo
    cols_to_drop = ['gender'] + binary_cols
    df_encoded = df_encoded.drop(columns=[col for col in cols_to_drop if col in df_encoded.columns])
    
    # Asegurar que todas las columnas del modelo están presentes
    # (En producción, necesitarías guardar las columnas exactas del entrenamiento)
    
    return df_encoded


def calculate_risk_level(probability: float) -> str:
    """
    Calcula el nivel de riesgo basado en la probabilidad de churn
    
    Args:
        probability: Probabilidad de churn (0-1)
        
    Returns:
        Nivel de riesgo: Low, Medium, High, Critical
    """
    if probability < 0.3:
        return "Low"
    elif probability < 0.5:
        return "Medium"
    elif probability < 0.7:
        return "High"
    else:
        return "Critical"


# =====================================================
# Endpoints
# =====================================================

@app.get("/", tags=["Health"])
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "message": "Churn Prediction API is running",
        "model_loaded": MODEL is not None,
        "timestamp": datetime.now().isoformat()
    }


@app.get("/info", tags=["Model Info"])
async def model_info():
    """Obtener información del modelo"""
    if MODEL is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    
    return {
        "model_info": MODEL_INFO,
        "status": "ready",
        "endpoints": {
            "predict": "/predict",
            "predict_batch": "/predict_batch",
            "docs": "/docs"
        }
    }


@app.post("/predict", response_model=PredictionOutput, tags=["Predictions"])
async def predict_churn(customer: CustomerInput):
    """
    Predecir probabilidad de churn para un cliente individual
    
    Args:
        customer: Datos del cliente
        
    Returns:
        Predicción de churn con probabilidad y nivel de riesgo
    """
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Modelo o scaler no cargado")
    
    try:
        # Preprocesar datos
        df_processed = preprocess_input(customer)
        
        # Escalar features numéricas (si aplica)
        # Nota: En producción real, necesitas manejar el escalado correctamente
        
        # Hacer predicción
        probability = float(MODEL.predict_proba(df_processed)[0][1])
        prediction = "Churn" if probability >= 0.5 else "No Churn"
        risk_level = calculate_risk_level(probability)
        
        # Confianza basada en qué tan lejos está del umbral 0.5
        confidence = abs(probability - 0.5) * 2  # 0 en 0.5, 1 en 0 o 1
        
        logger.info(f"Predicción realizada: {prediction} (prob: {probability:.3f})")
        
        return PredictionOutput(
            churn_probability=round(probability, 4),
            churn_prediction=prediction,
            risk_level=risk_level,
            confidence=round(confidence, 4),
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en predicción: {str(e)}")


@app.post("/predict_batch", response_model=BatchPredictionOutput, tags=["Predictions"])
async def predict_batch(batch: BatchPredictionInput):
    """
    Predecir probabilidad de churn para múltiples clientes
    
    Args:
        batch: Lista de clientes (máximo 100)
        
    Returns:
        Predicciones para todos los clientes
    """
    if MODEL is None or SCALER is None:
        raise HTTPException(status_code=503, detail="Modelo o scaler no cargado")
    
    if len(batch.customers) > 100:
        raise HTTPException(status_code=400, detail="Máximo 100 clientes por request")
    
    try:
        predictions = []
        high_risk_count = 0
        
        for customer in batch.customers:
            # Reutilizar endpoint individual
            prediction = await predict_churn(customer)
            predictions.append(prediction)
            
            if prediction.risk_level in ["High", "Critical"]:
                high_risk_count += 1
        
        logger.info(f"Batch prediction: {len(predictions)} clientes, {high_risk_count} alto riesgo")
        
        return BatchPredictionOutput(
            predictions=predictions,
            total_customers=len(predictions),
            high_risk_count=high_risk_count,
            timestamp=datetime.now().isoformat()
        )
        
    except Exception as e:
        logger.error(f"Error en batch prediction: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error en batch prediction: {str(e)}")


# =====================================================
# Punto de entrada
# =====================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)


import requests

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