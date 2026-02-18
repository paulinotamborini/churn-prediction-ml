# 🚀 Guía de Inicio Rápido - Streamlit App

## ⚡ Inicio en 3 Pasos

### Paso 1: Instalar Dependencias

```powershell
# En la terminal de VS Code, navega a streamlit_app
cd streamlit_app

# Instala las dependencias
pip install -r requirements.txt
```

### Paso 2: Verificar Modelos

Asegúrate de tener estos archivos:
```
models/
├── xgboost_optimized.pkl  ✅
└── scaler.pkl              ✅
```

Si no los tienes, ejecuta:
- Notebook `03_feature_engineering.ipynb` (genera scaler.pkl)
- Notebook `05_evaluation_optimization.ipynb` (genera xgboost_optimized.pkl)

### Paso 3: Ejecutar la App

**Opción A: Con script (Más fácil)**
```powershell
# En Windows, doble clic en:
run.bat

# O desde terminal:
.\run.bat
```

**Opción B: Comando directo**
```powershell
streamlit run app.py
```

La app se abrirá automáticamente en: **http://localhost:8501**

---

## 🎯 Funcionalidades Principales

### 1. 🏠 Home
- Vista general del sistema
- Métricas del modelo
- Distribución de probabilidades

**Cómo usar:**
1. Abre la app
2. La página Home se carga automáticamente
3. Explora las métricas y gráficos

### 2. 🎯 Predicción Individual

Predice el churn de un cliente específico.

**Cómo usar:**
1. Click en **🎯 Predicción Individual** en la barra lateral
2. Completa el formulario:
   - **Información Básica**: Género, edad, pareja, dependientes
   - **Servicios**: Internet, teléfono, streaming, seguridad
   - **Contrato**: Tipo, método de pago
   - **Financiero**: Cargo mensual y total
3. Click en **🔮 Predecir Churn**
4. Revisa los resultados:
   - ✅ Probabilidad de churn
   - ✅ Nivel de riesgo
   - ✅ Gauge visual
   - ✅ Recomendaciones personalizadas
   - ✅ Factores de riesgo

**Ejemplo de caso de uso:**

*Cliente de Alto Riesgo:*
- Tenure: 2 meses (cliente nuevo)
- Contract: Month-to-month
- PaymentMethod: Electronic check
- OnlineSecurity: No
- TechSupport: No
- → Resultado esperado: **Alto riesgo de churn**

*Cliente Estable:*
- Tenure: 60 meses (cliente antiguo)
- Contract: Two year
- PaymentMethod: Bank transfer (automatic)
- OnlineSecurity: Yes
- TechSupport: Yes
- → Resultado esperado: **Bajo riesgo de churn**

### 3. 📊 Dashboard

Vista general de métricas y tendencias.

**Cómo usar:**
1. Click en **📊 Dashboard**
2. Explora las métricas principales
3. Interactúa con los gráficos:
   - **Hover**: Ver valores exactos
   - **Zoom**: Click y arrastra
   - **Pan**: Shift + Click y arrastra
   - **Reset**: Doble click

**Gráficos disponibles:**
- 📈 KPIs principales
- 🥧 Pie chart de distribución
- 📊 Churn por antigüedad
- 📉 Distribución de cargo mensual
- 📈 Servicios vs Churn

### 4. 📈 Análisis Masivo

Predice churn para múltiples clientes a la vez.

**Cómo usar:**
1. Click en **📈 Análisis Masivo**
2. Prepara tu CSV con las columnas requeridas
3. Click en **📁 Cargar archivo CSV**
4. Selecciona tu archivo
5. Click en **🔮 Ejecutar Predicciones**
6. Descarga los resultados

**Formato del CSV:**

```csv
tenure,MonthlyCharges,TotalCharges,gender,SeniorCitizen,Partner,Dependents,...
12,70.0,840.0,Male,0,Yes,No,...
48,55.0,2640.0,Female,0,Yes,Yes,...
```

---

## 🎨 Personalización

### Cambiar el Tema

Edita `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"        # Color principal
backgroundColor = "#FFFFFF"      # Fondo principal
secondaryBackgroundColor = "#F0F2F6"  # Fondo sidebar
textColor = "#262730"           # Color de texto
```

### Cambiar el Puerto

```powershell
streamlit run app.py --server.port 8502
```

### Desactivar Auto-reload

En `.streamlit/config.toml`:

```toml
[server]
runOnSave = false
```

---

## 🔧 Atajos de Teclado

| Atajo | Acción |
|-------|--------|
| `R` | Recargar la app |
| `Ctrl + Shift + R` | Limpiar caché y recargar |
| `Ctrl + C` | Detener el servidor |

---

## 📊 Interpretación de Resultados

### Niveles de Riesgo

| Nivel | Probabilidad | Acción |
|-------|--------------|--------|
| 🟢 **Bajo** | 0-30% | Monitoreo regular |
| 🟡 **Medio** | 30-50% | Atención preventiva |
| 🟠 **Alto** | 50-70% | Intervención necesaria |
| 🔴 **Crítico** | 70-100% | Acción inmediata |

### Factores de Riesgo Comunes

1. **Contrato mes a mes** → Alto riesgo
2. **Cliente nuevo (< 12 meses)** → Alto riesgo
3. **Pago con cheque electrónico** → Medio riesgo
4. **Sin servicios de protección** → Medio riesgo
5. **Cargo mensual muy alto** → Medio riesgo

---

## 🐛 Solución de Problemas

### La app no inicia

```powershell
# Verificar instalación de Streamlit
streamlit --version

# Si no está instalado:
pip install streamlit

# Reinstalar dependencias
pip install -r requirements.txt --upgrade
```

### Error: "Model not loaded"

```powershell
# Verificar que los modelos existen
dir ..\models\

# Deberías ver:
# xgboost_optimized.pkl
# scaler.pkl

# Si no existen, ejecuta notebooks 03 y 05
```

### Error: "Address already in use"

```powershell
# El puerto 8501 ya está en uso, usa otro puerto:
streamlit run app.py --server.port 8502
```

### Los gráficos no se muestran

```powershell
# Actualizar Plotly
pip install --upgrade plotly
```

### La app es muy lenta

```powershell
# Limpiar caché de Streamlit
# En la app, presiona: Ctrl + Shift + R

# O en terminal:
streamlit cache clear
```

---

## 💡 Tips y Trucos

### 1. **Caché de Datos**

Usa `@st.cache_data` para funciones que cargan datos:

```python
@st.cache_data
def load_data():
    return pd.read_csv('data.csv')
```

### 2. **Caché de Modelos**

Usa `@st.cache_resource` para modelos:

```python
@st.cache_resource
def load_model():
    return joblib.load('model.pkl')
```

### 3. **Estado de Sesión**

Mantén estado entre reruns:

```python
if 'counter' not in st.session_state:
    st.session_state.counter = 0

st.session_state.counter += 1
```

### 4. **Columnas Responsivas**

```python
col1, col2, col3 = st.columns([2, 1, 1])
# Las proporciones son: 50%, 25%, 25%
```

### 5. **Expanders para Organizar**

```python
with st.expander("Ver detalles"):
    st.write("Contenido oculto")
```

---

## 📚 Recursos Adicionales

- 📖 [Documentación de Streamlit](https://docs.streamlit.io/)
- 🎓 [Tutoriales de Streamlit](https://docs.streamlit.io/get-started/tutorials)
- 🌟 [Galería de Apps](https://streamlit.io/gallery)
- 💬 [Foro de Streamlit](https://discuss.streamlit.io/)

---

## ✅ Checklist de Verificación

Antes de usar la app, verifica:

- [ ] ✅ Python 3.10+ instalado
- [ ] ✅ Dependencias instaladas (`pip install -r requirements.txt`)
- [ ] ✅ Modelos generados (notebooks 03 y 05 ejecutados)
- [ ] ✅ Puerto 8501 disponible
- [ ] ✅ Navegador actualizado

---

## 🎉 ¡Listo!

Tu aplicación de Streamlit está configurada y lista para usar.

**Comandos esenciales:**

```powershell
# Iniciar app
streamlit run app.py

# Ver ayuda
streamlit --help

# Limpiar caché
streamlit cache clear

# Ver versión
streamlit --version
```

**¡Disfruta prediciendo churn!** 🔮
