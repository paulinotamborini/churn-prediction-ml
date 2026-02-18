# 🔮 Churn Prediction Dashboard - Streamlit App

Aplicación interactiva de Streamlit para predecir y analizar churn de clientes en tiempo real.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-00ADD8?style=for-the-badge&logo=xgboost&logoColor=white)

## 🌟 Características

### 🏠 **Home**
- Vista general del sistema
- Métricas principales del modelo
- Distribución de probabilidades
- Guía de inicio rápido

### 🎯 **Predicción Individual**
- Formulario interactivo para entrada de datos
- Predicción en tiempo real
- Gauge visual de probabilidad
- Nivel de riesgo (Bajo, Medio, Alto, Crítico)
- Recomendaciones personalizadas
- Identificación de factores de riesgo

### 📊 **Dashboard**
- KPIs principales (Total clientes, Retenidos, Churn, Tasa)
- Gráfico de distribución (Pie chart)
- Churn por antigüedad (Bar chart)
- Distribución de cargo mensual (Histogram)
- Servicios vs Churn (Line chart)

### 📈 **Análisis Masivo**
- Carga de archivos CSV
- Predicciones en lote
- Exportación de resultados
- Visualizaciones agregadas

### ℹ️ **Acerca de**
- Información del sistema
- Métricas del modelo
- Tecnologías utilizadas

## 🚀 Instalación

### Opción 1: Local

```powershell
# 1. Navegar a la carpeta
cd streamlit_app

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar la app
streamlit run app.py
```

### Opción 2: Con entorno virtual

```powershell
# 1. Crear entorno virtual
python -m venv venv
.\venv\Scripts\activate

# 2. Instalar dependencias
pip install -r requirements.txt

# 3. Ejecutar
streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📁 Estructura del Proyecto

```
streamlit_app/
├── app.py                  # Aplicación principal
├── requirements.txt        # Dependencias
├── README.md              # Esta documentación
├── pages/                 # Páginas adicionales (futuro)
├── utils/                 # Funciones auxiliares (futuro)
└── assets/                # Imágenes y recursos (futuro)
```

## 🎨 Capturas de Pantalla

### Home
![Home](https://via.placeholder.com/800x400?text=Dashboard+Home)

### Predicción Individual
![Prediction](https://via.placeholder.com/800x400?text=Prediction+Form)

### Dashboard
![Dashboard](https://via.placeholder.com/800x400?text=Analytics+Dashboard)

## 🎯 Uso

### 1. Predicción Individual

1. Ve a la página **🎯 Predicción Individual**
2. Completa el formulario con los datos del cliente
3. Haz clic en **🔮 Predecir Churn**
4. Revisa:
   - Probabilidad de churn
   - Nivel de riesgo
   - Recomendaciones
   - Factores de riesgo

### 2. Dashboard

1. Ve a la página **📊 Dashboard**
2. Explora las métricas principales
3. Interactúa con los gráficos (zoom, pan, hover)
4. Analiza tendencias y patrones

### 3. Análisis Masivo

1. Ve a la página **📈 Análisis Masivo**
2. Descarga el template CSV
3. Completa con tus datos
4. Carga el archivo
5. Ejecuta predicciones en lote

## 🔧 Personalización

### Cambiar Tema

Edita `.streamlit/config.toml`:

```toml
[theme]
primaryColor = "#FF4B4B"
backgroundColor = "#FFFFFF"
secondaryBackgroundColor = "#F0F2F6"
textColor = "#262730"
font = "sans serif"
```

### Agregar Logo

```python
st.sidebar.image("assets/logo.png", width=200)
```

### Modificar Métricas

Edita las secciones de métricas en `app.py`:

```python
st.metric(
    label="Tu Métrica",
    value="Valor",
    delta="+X%"
)
```

## 📊 Niveles de Riesgo

| Probabilidad | Nivel      | Color | Icono | Acción                    |
|--------------|------------|-------|-------|---------------------------|
| 0.0 - 0.3    | Bajo       | Verde | 🟢    | Monitoreo regular         |
| 0.3 - 0.5    | Medio      | Amarillo | 🟡 | Atención preventiva       |
| 0.5 - 0.7    | Alto       | Naranja | 🟠  | Intervención necesaria    |
| 0.7 - 1.0    | Crítico    | Rojo  | 🔴    | Acción inmediata          |

## 🐛 Troubleshooting

### Error: "Model not loaded"

**Causa:** Archivos del modelo no encontrados

**Solución:**
```powershell
# Verificar que existen los modelos
dir ..\models\

# Deberías ver:
# - xgboost_optimized.pkl
# - scaler.pkl

# Si no existen, ejecuta los notebooks 03 y 05
```

### Error: "ModuleNotFoundError"

**Solución:**
```powershell
pip install -r requirements.txt
```

### La app no se actualiza

**Solución:**
Presiona `R` en la ventana del navegador o habilita el auto-rerun en Settings.

### Gráficos no se muestran

**Solución:**
```powershell
pip install --upgrade plotly
```

## 🚀 Deployment

### Streamlit Cloud (Gratis)

1. Sube tu código a GitHub
2. Ve a [share.streamlit.io](https://share.streamlit.io)
3. Conecta tu repositorio
4. Selecciona `streamlit_app/app.py`
5. ¡Deploy!

### Heroku

```bash
# 1. Crear Procfile
echo "web: sh setup.sh && streamlit run streamlit_app/app.py" > Procfile

# 2. Crear setup.sh
cat > setup.sh << EOF
mkdir -p ~/.streamlit/
echo "[server]
headless = true
port = \$PORT
enableCORS = false
" > ~/.streamlit/config.toml
EOF

# 3. Deploy
heroku create tu-app-churn
git push heroku main
```

### Docker

```dockerfile
FROM python:3.10-slim

WORKDIR /app

COPY streamlit_app/requirements.txt .
RUN pip install -r requirements.txt

COPY streamlit_app/ ./streamlit_app/
COPY models/ ./models/

EXPOSE 8501

CMD ["streamlit", "run", "streamlit_app/app.py", "--server.port=8501"]
```

```bash
docker build -t churn-streamlit .
docker run -p 8501:8501 churn-streamlit
```

## 📝 Roadmap

- [ ] 📊 Más visualizaciones (heatmaps, scatter plots)
- [ ] 📈 Gráficos de tendencias temporales
- [ ] 🎨 Temas personalizables
- [ ] 📁 Exportación de reportes PDF
- [ ] 🔔 Sistema de alertas
- [ ] 📧 Integración con email
- [ ] 🤖 Chatbot de ayuda
- [ ] 🌐 Multi-idioma

## 🤝 Contribuciones

Las contribuciones son bienvenidas! Por favor:

1. Fork el proyecto
2. Crea una rama (`git checkout -b feature/AmazingFeature`)
3. Commit tus cambios (`git commit -m 'Add AmazingFeature'`)
4. Push a la rama (`git push origin feature/AmazingFeature`)
5. Abre un Pull Request

## 📄 Licencia

MIT License - Ver `LICENSE` para más detalles

## 👤 Autor

**Tu Nombre**
- GitHub: github.com/paulinotamborini
- LinkedIn: [https://www.linkedin.com/in/paulino-tamborini-41a60b272/]
- Email: paulinotamborini044@gmail.com

## 🙏 Agradecimientos

- [Streamlit](https://streamlit.io/) por el framework
- [Plotly](https://plotly.com/) por las visualizaciones
- [XGBoost](https://xgboost.readthedocs.io/) por el modelo

---

⭐ Si te gustó este proyecto, dale una estrella en GitHub!

🔮 **Happy Predicting!**
