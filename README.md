# Sistema de Evaluación de Riesgo Crediticio con Machine Learning

![Python](https://img.shields.io/badge/python-v3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-99.29%25-success.svg)
![Dataset](https://img.shields.io/badge/dataset-26,940%20records-blue.svg)

## Descripción

Sistema completo de Machine Learning para evaluación automatizada de riesgo crediticio, entrenado con **26,940 solicitudes reales** de crédito. Utiliza **XGBoost** con **99.29% de AUC-ROC** para clasificar solicitudes como aprobadas o rechazadas, optimizando la toma de decisiones en originación de créditos.

## 🚀 Demo Rápida

```powershell
# 1. Entrenar el modelo
python src/models/train_model_real.py

# 2. Ejecutar dashboard
streamlit run dashboard_final.py --server.port 8508
```

**Dashboard disponible en**: http://localhost:8508

## 🏆 Rendimiento del Modelo

**Modelo XGBoost entrenado con 26,940 registros:**

- **AUC-ROC**: 99.29%
- **Accuracy**: 97.62%
- **Precision**: 97.16%
- **Recall**: 98.69%
- **F1-Score**: 97.92%

## ✨ Características Principales

- 🎯 **Modelo XGBoost** de alta precisión (99.29% AUC-ROC)
- 📊 **Dataset real** con 26,940 solicitudes de crédito
- 🔧 **9 variables parametrizables** en dashboard interactivo
- 📈 **Visualización en tiempo real** de evaluaciones
- ⚙️ **Sistema flexible** con activación/desactivación de variables
- 🧪 **Validación cruzada** y tests automatizados
- 🌐 **API REST** para integración

## Estructura del Proyecto

```
credit-risk/
├── data/
│   ├── raw/                 # Datos originales
│   └── processed/           # Datos procesados
├── src/
│   ├── data/               # Scripts de carga y limpieza
│   ├── features/           # Ingeniería de características
│   ├── models/             # Modelos de ML
│   └── visualization/      # Visualizaciones
├── notebooks/              # Jupyter notebooks
├── api/                    # API REST
├── dashboard/              # Dashboard Streamlit
├── models/                 # Modelos entrenados
├── tests/                  # Tests unitarios
├── docs/                   # Documentación
└── config/                 # Configuraciones
```

## Instalación

### Prerrequisitos

- Python 3.9 o superior
- pip
- Git

### Configuración del Entorno

1. **Clonar el repositorio:**
```bash
git clone <repository-url>
cd credit-risk
```

2. **Crear entorno virtual:**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Instalar dependencias:**
```bash
pip install -r requirements.txt
```

4. **Configurar variables de entorno:**
```bash
cp config/.env.example config/.env
# Editar config/.env con tus configuraciones
```

## Uso Rápido

### 1. Análisis Exploratorio
```bash
jupyter notebook notebooks/01_exploratory_data_analysis.ipynb
```

### 2. Entrenamiento de Modelo
```bash
python src/models/train_model.py
```

### 3. Evaluación de Modelo
```bash
python src/models/evaluate_model.py
```

### 4. API REST
```bash
python api/app.py
```

### 5. Dashboard
```bash
streamlit run dashboard/app.py
```

## Modelos Implementados

| Modelo | Descripción | Uso |
|--------|-------------|-----|
| **Logistic Regression** | Modelo lineal interpretable | Baseline y explicabilidad |
| **Random Forest** | Ensemble de árboles | Balance entre precisión e interpretabilidad |
| **XGBoost** | Gradient boosting optimizado | Alto rendimiento |
| **LightGBM** | Gradient boosting eficiente | Datos grandes y velocidad |

## Métricas de Evaluación

- **Accuracy**: Precisión general
- **Precision/Recall**: Para clases desbalanceadas
- **F1-Score**: Balance entre precision y recall
- **AUC-ROC**: Capacidad de discriminación
- **Confusion Matrix**: Análisis detallado de errores
- **Feature Importance**: Interpretabilidad del modelo

## API Endpoints

### Predicción Individual
```http
POST /predict
Content-Type: application/json

{
  "age": 35,
  "income": 50000,
  "credit_score": 720,
  "debt_ratio": 0.3,
  "employment_years": 5
}
```

### Predicción por Lotes
```http
POST /predict_batch
Content-Type: application/json

{
  "data": [
    {"age": 35, "income": 50000, ...},
    {"age": 28, "income": 35000, ...}
  ]
}
```

### Métricas del Modelo
```http
GET /model/metrics
```

## Dashboard

El dashboard interactivo incluye:

- 📊 **Visualización de datos**: Distribuciones y correlaciones
- 🎯 **Resultados de predicción**: Probabilidades y decisiones
- 📈 **Métricas del modelo**: Rendimiento en tiempo real
- 🔍 **Análisis de características**: Importancia de variables
- 📋 **Simulador**: Herramienta para probar diferentes escenarios

## Testing

```bash
# Ejecutar todos los tests
pytest tests/

# Tests con cobertura
pytest tests/ --cov=src/

# Tests específicos
pytest tests/test_models.py -v
```

## Docker

### Construir imagen
```bash
docker build -t credit-risk-system .
```

### Ejecutar contenedor
```bash
docker run -p 8000:8000 -p 8501:8501 credit-risk-system
```

### Docker Compose
```bash
docker-compose up -d
```

## Contribución

1. Fork del proyecto
2. Crear rama para nueva funcionalidad (`git checkout -b feature/nueva-funcionalidad`)
3. Commit de cambios (`git commit -am 'Agregar nueva funcionalidad'`)
4. Push a la rama (`git push origin feature/nueva-funcionalidad`)
5. Crear Pull Request

## Trabajo de Grado

Este proyecto forma parte del trabajo de grado **"Sistema de Recomendación para Evaluación de Riesgos en Créditos: Un Enfoque Basado en Aprendizaje Automático"** para el programa de Administración.

### Objetivos del Proyecto

- ✅ Identificar y recopilar datos necesarios para el dataset
- ✅ Aplicar técnicas de preprocesamiento y limpieza
- ✅ Implementar modelos de ML para recomendación crediticia
- ✅ Evaluar desempeño con métricas especializadas

### Resultados Esperados

- Reducción en tasas de morosidad
- Mejora en velocidad de aprobación
- Mayor precisión en evaluación de riesgo
- Sistema escalable y automatizado

## Licencia

Este proyecto está bajo la Licencia MIT - ver el archivo [LICENSE](LICENSE) para detalles.

## Contacto

- **Autor**: [Tu Nombre]
- **Email**: [tu.email@ejemplo.com]
- **LinkedIn**: [Tu perfil de LinkedIn]
- **Universidad**: [Nombre de tu Universidad]

## Agradecimientos

- Profesores y asesores del programa
- Comunidad de ciencia de datos
- Librerías de código abierto utilizadas

---

⭐ Si este proyecto te resulta útil, ¡considera darle una estrella!

## Ejecutar
streamlit run dashboard/app.py --server.port 8502

streamlit run dashboard_final.py --server.port 8506