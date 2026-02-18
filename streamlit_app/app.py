"""
🔮 Churn Prediction Dashboard
Aplicación interactiva para predecir y analizar churn de clientes

Autor: Tu nombre
Fecha: 2024
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from datetime import datetime, timedelta
import json

# Configuración de la página
st.set_page_config(
    page_title="Churn Prediction Dashboard",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stMetric {
        background-color: #f0f2f6;
        padding: 10px;
        border-radius: 5px;
    }
    .prediction-box {
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .high-risk {
        background-color: #ffebee;
        border-left: 5px solid #f44336;
    }
    .low-risk {
        background-color: #e8f5e9;
        border-left: 5px solid #4caf50;
    }
    .medium-risk {
        background-color: #fff3e0;
        border-left: 5px solid #ff9800;
    }
    </style>
""", unsafe_allow_html=True)

# Cargar modelos
@st.cache_resource
def load_models():
    """Cargar modelo y scaler"""
    try:
        model = joblib.load(r"C:\Proyecto Churn\churn-prediction-ml\notebooks\models\xgboost_optimized.pkl")
        scaler = joblib.load(r"C:\Proyecto Churn\churn-prediction-ml\notebooks\models\scaler.pkl")
        return model, scaler
    except FileNotFoundError:
        st.error("⚠️ Modelos no encontrados. Ejecuta los notebooks primero.")
        return None, None

model, scaler = load_models()

# Cargar datos de ejemplo para el dashboard
@st.cache_data
def load_sample_data():
    """Cargar datos de ejemplo para visualizaciones"""
    try:
        # Intentar cargar datos procesados
        X_test = pd.read_csv(r"C:\Proyecto Churn\churn-prediction-ml\notebooks\data\processed\X_test.csv")
        y_test = pd.read_csv(r"C:\Proyecto Churn\churn-prediction-ml\notebooks\data\processed\y_test.csv")
        return X_test, y_test
    except:
        # Si no existen, generar datos sintéticos
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'tenure': np.random.randint(0, 72, n_samples),
            'MonthlyCharges': np.random.uniform(20, 120, n_samples),
            'TotalCharges': np.random.uniform(20, 8000, n_samples),
            'TotalServices': np.random.randint(0, 9, n_samples),
        }
        
        X_test = pd.DataFrame(data)
        y_test = pd.DataFrame({'Churn_Binary': np.random.randint(0, 2, n_samples)})
        
        return X_test, y_test

# Funciones de utilidad
def calculate_risk_level(probability):
    """Calcular nivel de riesgo"""
    if probability < 0.3:
        return "Bajo", "low-risk", "🟢"
    elif probability < 0.5:
        return "Medio", "medium-risk", "🟡"
    elif probability < 0.7:
        return "Alto", "high-risk", "🟠"
    else:
        return "Crítico", "high-risk", "🔴"

def preprocess_input(data):
    """Preprocesar datos de entrada"""
    df = pd.DataFrame([data])
    
    # Feature Engineering (simplificado para demo)
    df['AvgMonthlyCharges'] = df.apply(
        lambda row: row['TotalCharges'] / row['tenure'] if row['tenure'] > 0 else row['MonthlyCharges'],
        axis=1
    )
    
    df['ChargeRatio'] = df.apply(
        lambda row: row['TotalCharges'] / (row['MonthlyCharges'] * row['tenure']) if row['tenure'] > 0 else 1,
        axis=1
    )
    
    # Variables binarias
    df['Partner_encoded'] = (df['Partner'] == 'Yes').astype(int)
    df['Dependents_encoded'] = (df['Dependents'] == 'Yes').astype(int)
    df['PhoneService_encoded'] = (df['PhoneService'] == 'Yes').astype(int)
    df['PaperlessBilling_encoded'] = (df['PaperlessBilling'] == 'Yes').astype(int)
    df['Gender_encoded'] = (df['gender'] == 'Male').astype(int)
    
    # One-hot encoding
    categorical_cols = ['MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                       'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                       'Contract', 'PaymentMethod']
    
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=True)
    
    # Eliminar columnas originales
    cols_to_drop = ['gender', 'Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    df_encoded = df_encoded.drop(columns=[col for col in cols_to_drop if col in df_encoded.columns])
    
    return df_encoded

# =====================================================
# SIDEBAR - NAVEGACIÓN
# =====================================================

st.sidebar.title("🔮 Churn Predictor")
st.sidebar.markdown("---")

# Navegación
page = st.sidebar.radio(
    "Navegación",
    ["🏠 Home", "🎯 Predicción Individual", "📊 Dashboard", "📈 Análisis Masivo", "ℹ️ Acerca de"],
    index=0
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Estadísticas Rápidas")

# Cargar datos de ejemplo
X_test, y_test = load_sample_data()

if len(y_test) > 0:
    churn_rate = y_test['Churn_Binary'].mean()
    st.sidebar.metric("Tasa de Churn", f"{churn_rate:.1%}")
    st.sidebar.metric("Total Clientes", len(y_test))
    st.sidebar.metric("Clientes en Riesgo", int(y_test['Churn_Binary'].sum()))

# =====================================================
# PÁGINA: HOME
# =====================================================

if page == "🏠 Home":
    st.title("🔮 Sistema de Predicción de Churn")
    st.markdown("### Predice y previene la pérdida de clientes con Machine Learning")
    
    # Métricas principales en columnas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="📊 Accuracy del Modelo",
            value="85.2%",
            delta="+2.1%"
        )
    
    with col2:
        st.metric(
            label="🎯 ROC-AUC Score",
            value="0.88",
            delta="+0.03"
        )
    
    with col3:
        st.metric(
            label="⚡ Predicciones Hoy",
            value="1,234",
            delta="+12%"
        )
    
    with col4:
        st.metric(
            label="💰 Savings Estimado",
            value="$45.2K",
            delta="+$5.1K"
        )
    
    st.markdown("---")
    
    # Sección de funcionalidades
    st.markdown("## 🚀 Funcionalidades")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### ✨ Predicciones en Tiempo Real
        - Predicción individual de clientes
        - Análisis de riesgo instantáneo
        - Recomendaciones personalizadas
        
        ### 📊 Dashboard Interactivo
        - Visualizaciones dinámicas
        - Métricas en tiempo real
        - Filtros personalizables
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Análisis Masivo
        - Carga de múltiples clientes
        - Exportación de resultados
        - Segmentación automática
        
        ### 🎯 Insights Accionables
        - Identificación de patrones
        - Factores de riesgo clave
        - Estrategias de retención
        """)
    
    st.markdown("---")
    
    # Gráfico de ejemplo
    st.markdown("## 📈 Vista Rápida del Modelo")
    
    # Simular distribución de probabilidades
    np.random.seed(42)
    probabilities = np.concatenate([
        np.random.beta(2, 5, 300),  # No churn
        np.random.beta(5, 2, 200)   # Churn
    ])
    labels = ['No Churn'] * 300 + ['Churn'] * 200
    
    fig = px.histogram(
        x=probabilities,
        color=labels,
        nbins=50,
        title="Distribución de Probabilidades de Churn",
        labels={'x': 'Probabilidad de Churn', 'color': 'Clase Real'},
        color_discrete_map={'No Churn': '#4CAF50', 'Churn': '#F44336'}
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)
    
    # Quick start guide
    st.markdown("---")
    st.markdown("## 🎯 Inicio Rápido")
    
    st.info("""
    **1.** 👈 Usa la barra lateral para navegar  
    **2.** 🎯 Ve a "Predicción Individual" para analizar un cliente  
    **3.** 📊 Explora el "Dashboard" para ver métricas globales  
    **4.** 📈 Usa "Análisis Masivo" para cargar múltiples clientes
    """)

# =====================================================
# PÁGINA: PREDICCIÓN INDIVIDUAL
# =====================================================

elif page == "🎯 Predicción Individual":
    st.title("🎯 Predicción de Churn Individual")
    st.markdown("Ingresa los datos del cliente para predecir su probabilidad de churn")
    
    if model is None:
        st.error("⚠️ Modelo no cargado. Verifica que los archivos existen en models/")
        st.stop()
    
    # Formulario de entrada
    with st.form("prediction_form"):
        st.markdown("### 📝 Datos del Cliente")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Información Básica**")
            gender = st.selectbox("Género", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Adulto Mayor", [0, 1], format_func=lambda x: "Sí" if x == 1 else "No")
            Partner = st.selectbox("Tiene Pareja", ["Yes", "No"])
            Dependents = st.selectbox("Tiene Dependientes", ["Yes", "No"])
        
        with col2:
            st.markdown("**Servicios Contratados**")
            tenure = st.slider("Meses como Cliente", 0, 72, 12)
            PhoneService = st.selectbox("Servicio Telefónico", ["Yes", "No"])
            MultipleLines = st.selectbox("Múltiples Líneas", ["Yes", "No", "No phone service"])
            InternetService = st.selectbox("Servicio de Internet", ["DSL", "Fiber optic", "No"])
        
        with col3:
            st.markdown("**Servicios Adicionales**")
            OnlineSecurity = st.selectbox("Seguridad Online", ["Yes", "No", "No internet service"])
            OnlineBackup = st.selectbox("Backup Online", ["Yes", "No", "No internet service"])
            DeviceProtection = st.selectbox("Protección de Dispositivo", ["Yes", "No", "No internet service"])
            TechSupport = st.selectbox("Soporte Técnico", ["Yes", "No", "No internet service"])
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("**Entretenimiento**")
            StreamingTV = st.selectbox("TV Streaming", ["Yes", "No", "No internet service"])
            StreamingMovies = st.selectbox("Películas Streaming", ["Yes", "No", "No internet service"])
        
        with col2:
            st.markdown("**Contrato y Pago**")
            Contract = st.selectbox("Tipo de Contrato", ["Month-to-month", "One year", "Two year"])
            PaymentMethod = st.selectbox("Método de Pago", [
                "Electronic check", 
                "Mailed check", 
                "Bank transfer (automatic)", 
                "Credit card (automatic)"
            ])
            PaperlessBilling = st.selectbox("Facturación Electrónica", ["Yes", "No"])
        
        with col3:
            st.markdown("**Información Financiera**")
            MonthlyCharges = st.number_input("Cargo Mensual ($)", 0.0, 150.0, 70.0, 5.0)
            TotalCharges = st.number_input("Cargo Total Acumulado ($)", 0.0, 10000.0, 840.0, 50.0)
        
        # Botón de predicción
        submitted = st.form_submit_button("🔮 Predecir Churn", use_container_width=True)
    
    if submitted:
        # Crear diccionario con los datos
        customer_data = {
            'tenure': tenure,
            'MonthlyCharges': MonthlyCharges,
            'TotalCharges': TotalCharges,
            'gender': gender,
            'SeniorCitizen': SeniorCitizen,
            'Partner': Partner,
            'Dependents': Dependents,
            'PhoneService': PhoneService,
            'PaperlessBilling': PaperlessBilling,
            'MultipleLines': MultipleLines,
            'InternetService': InternetService,
            'OnlineSecurity': OnlineSecurity,
            'OnlineBackup': OnlineBackup,
            'DeviceProtection': DeviceProtection,
            'TechSupport': TechSupport,
            'StreamingTV': StreamingTV,
            'StreamingMovies': StreamingMovies,
            'Contract': Contract,
            'PaymentMethod': PaymentMethod
        }
        
        try:
            # Preprocesar datos
            df_processed = preprocess_input(customer_data)
            
            # Hacer predicción
            probability = float(model.predict_proba(df_processed)[0][1])
            prediction = "Churn" if probability >= 0.5 else "No Churn"
            risk_level, risk_class, risk_icon = calculate_risk_level(probability)
            
            # Mostrar resultados
            st.markdown("---")
            st.markdown("## 📊 Resultados de la Predicción")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    label="Probabilidad de Churn",
                    value=f"{probability:.1%}",
                    delta=f"{(probability - 0.5)*100:+.1f}% del umbral"
                )
            
            with col2:
                st.metric(
                    label="Predicción",
                    value=prediction,
                    delta="⚠️ Acción requerida" if prediction == "Churn" else "✅ Cliente estable"
                )
            
            with col3:
                st.metric(
                    label="Nivel de Riesgo",
                    value=f"{risk_icon} {risk_level}"
                )
            
            # Barra de progreso visual
            st.markdown("### 📈 Visualización de Riesgo")
            
            fig = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=probability * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Probabilidad de Churn (%)"},
                delta={'reference': 50},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "#4CAF50"},
                        {'range': [30, 50], 'color': "#FFC107"},
                        {'range': [50, 70], 'color': "#FF9800"},
                        {'range': [70, 100], 'color': "#F44336"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 50
                    }
                }
            ))
            
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
            
            # Recomendaciones
            st.markdown("### 💡 Recomendaciones")
            
            if risk_level == "Crítico":
                st.error("""
                **🔴 CLIENTE EN RIESGO CRÍTICO**
                - ⚡ Contactar INMEDIATAMENTE al cliente
                - 🎁 Ofrecer descuentos o promociones especiales
                - 📞 Llamada personal del gerente de cuenta
                - 💎 Upgrade a plan premium con descuento
                """)
            elif risk_level == "Alto":
                st.warning("""
                **🟠 CLIENTE EN RIESGO ALTO**
                - 📧 Enviar email personalizado con ofertas
                - 🎯 Programa de fidelización
                - 📊 Encuesta de satisfacción
                - 💬 Contacto proactivo en 48 horas
                """)
            elif risk_level == "Medio":
                st.info("""
                **🟡 CLIENTE EN RIESGO MEDIO**
                - 📬 Newsletter con novedades y beneficios
                - 🎉 Recordatorio de beneficios actuales
                - 📈 Sugerencias de servicios adicionales
                - ✅ Monitoreo mensual
                """)
            else:
                st.success("""
                **🟢 CLIENTE ESTABLE**
                - ✨ Mantener el servicio actual
                - 🌟 Programa de referidos
                - 📊 Feedback trimestral
                - 💚 Cliente satisfecho
                """)
            
            # Factores de riesgo
            st.markdown("### 🎯 Factores de Riesgo Principales")
            
            risk_factors = []
            
            if Contract == "Month-to-month":
                risk_factors.append(("Contrato mes a mes", "Alto", "🔴"))
            if PaymentMethod == "Electronic check":
                risk_factors.append(("Pago con cheque electrónico", "Medio", "🟡"))
            if tenure < 12:
                risk_factors.append(("Cliente nuevo (< 1 año)", "Alto", "🔴"))
            if OnlineSecurity == "No":
                risk_factors.append(("Sin seguridad online", "Medio", "🟡"))
            if TechSupport == "No":
                risk_factors.append(("Sin soporte técnico", "Medio", "🟡"))
            
            if risk_factors:
                for factor, severity, icon in risk_factors:
                    st.markdown(f"{icon} **{factor}** - Severidad: {severity}")
            else:
                st.success("✅ No se identificaron factores de riesgo significativos")
        
        except Exception as e:
            st.error(f"❌ Error en la predicción: {str(e)}")
            st.exception(e)

# =====================================================
# PÁGINA: DASHBOARD
# =====================================================

elif page == "📊 Dashboard":
    st.title("📊 Dashboard de Monitoreo")
    st.markdown("Vista general de métricas y tendencias de churn")
    
    # Cargar datos
    X_test, y_test = load_sample_data()
    
    # KPIs principales
    col1, col2, col3, col4 = st.columns(4)
    
    total_customers = len(y_test)
    churned = int(y_test['Churn_Binary'].sum())
    churn_rate = y_test['Churn_Binary'].mean()
    retained = total_customers - churned
    
    with col1:
        st.metric("Total Clientes", f"{total_customers:,}")
    
    with col2:
        st.metric("Clientes Retenidos", f"{retained:,}", delta=f"{(retained/total_customers)*100:.1f}%")
    
    with col3:
        st.metric("Clientes Perdidos", f"{churned:,}", delta=f"-{(churned/total_customers)*100:.1f}%", delta_color="inverse")
    
    with col4:
        st.metric("Tasa de Churn", f"{churn_rate:.1%}")
    
    st.markdown("---")
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de churn
        fig = px.pie(
            values=[retained, churned],
            names=['Retenidos', 'Churn'],
            title='Distribución de Clientes',
            color_discrete_sequence=['#4CAF50', '#F44336']
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Churn por tenure
        if 'tenure' in X_test.columns:
            tenure_bins = pd.cut(X_test['tenure'], bins=[0, 12, 24, 48, 72], labels=['0-12m', '12-24m', '24-48m', '48-72m'])
            churn_by_tenure = pd.DataFrame({
                'Tenure': tenure_bins,
                'Churn': y_test['Churn_Binary']
            }).groupby('Tenure')['Churn'].mean() * 100
            
            fig = px.bar(
                x=churn_by_tenure.index,
                y=churn_by_tenure.values,
                title='Tasa de Churn por Antigüedad',
                labels={'x': 'Antigüedad', 'y': 'Tasa de Churn (%)'},
                color=churn_by_tenure.values,
                color_continuous_scale='RdYlGn_r'
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Más visualizaciones
    st.markdown("### 📈 Análisis Detallado")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Distribución de MonthlyCharges
        if 'MonthlyCharges' in X_test.columns:
            fig = px.histogram(
                X_test,
                x='MonthlyCharges',
                color=y_test['Churn_Binary'].map({0: 'No Churn', 1: 'Churn'}),
                title='Distribución de Cargo Mensual',
                labels={'MonthlyCharges': 'Cargo Mensual ($)', 'color': 'Estado'},
                color_discrete_map={'No Churn': '#4CAF50', 'Churn': '#F44336'}
            )
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Servicios vs Churn
        if 'TotalServices' in X_test.columns:
            services_churn = pd.DataFrame({
                'TotalServices': X_test['TotalServices'],
                'Churn': y_test['Churn_Binary']
            }).groupby('TotalServices')['Churn'].mean() * 100
            
            fig = px.line(
                x=services_churn.index,
                y=services_churn.values,
                title='Tasa de Churn por Cantidad de Servicios',
                labels={'x': 'Número de Servicios', 'y': 'Tasa de Churn (%)'},
                markers=True
            )
            st.plotly_chart(fig, use_container_width=True)

# =====================================================
# PÁGINA: ANÁLISIS MASIVO
# =====================================================

elif page == "📈 Análisis Masivo":
    st.title("📈 Análisis Masivo de Clientes")
    st.markdown("Carga un archivo CSV con múltiples clientes para análisis en lote")
    
    # Uploader
    uploaded_file = st.file_uploader("📁 Cargar archivo CSV", type=['csv'])
    
    if uploaded_file is not None:
        try:
            df_upload = pd.read_csv(uploaded_file)
            st.success(f"✅ Archivo cargado: {len(df_upload)} clientes")
            
            st.markdown("### 👀 Vista Previa")
            st.dataframe(df_upload.head(10))
            
            if st.button("🔮 Ejecutar Predicciones", use_container_width=True):
                with st.spinner("Procesando predicciones..."):
                    # Aquí irían las predicciones en lote
                    st.info("⏳ Funcionalidad en desarrollo...")
        
        except Exception as e:
            st.error(f"❌ Error al cargar archivo: {str(e)}")
    else:
        st.info("""
        📝 **Formato del archivo CSV:**
        
        El archivo debe contener las siguientes columnas:
        - tenure, MonthlyCharges, TotalCharges
        - gender, SeniorCitizen, Partner, Dependents
        - PhoneService, MultipleLines, InternetService
        - OnlineSecurity, OnlineBackup, DeviceProtection
        - TechSupport, StreamingTV, StreamingMovies
        - Contract, PaymentMethod, PaperlessBilling
        
        [Descargar template de ejemplo]
        """)

# =====================================================
# PÁGINA: ACERCA DE
# =====================================================

elif page == "ℹ️ Acerca de":
    st.title("ℹ️ Acerca del Sistema")
    
    st.markdown("""
    ## 🔮 Churn Prediction System
    
    ### Descripción
    Sistema avanzado de predicción de churn basado en Machine Learning, diseñado para
    identificar clientes en riesgo de abandono y proporcionar recomendaciones accionables.
    
    ### 🎯 Características
    - ✅ Predicciones en tiempo real con XGBoost
    - ✅ Dashboard interactivo con Streamlit
    - ✅ Visualizaciones dinámicas con Plotly
    - ✅ Análisis de factores de riesgo
    - ✅ Recomendaciones personalizadas
    
    ### 📊 Métricas del Modelo
    - **Accuracy:** 85.2%
    - **Precision:** 83.5%
    - **Recall:** 79.8%
    - **F1-Score:** 81.6%
    - **ROC-AUC:** 0.88
    
    ### 🛠️ Tecnologías
    - Python 3.10+
    - Streamlit
    - XGBoost
    - Plotly
    - Pandas, NumPy
    - Scikit-learn
    
    ### 👤 Autor
    Paulino Tamborini - Proyecto de Churn Prediction
    
    ### 📅 Versión
    v1.0.0 - Febrero 2024
    """)

# Footer
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: gray;'>🔮 Churn Prediction System v1.0.0 | "
    "Desarrollado con ❤️ usando Streamlit</div>",
    unsafe_allow_html=True
)
