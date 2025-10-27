# Documentación del Sistema de Evaluación de Riesgo Crediticio

## Arquitectura del Sistema

### Componentes Principales

#### 1. Módulo de Datos (`src/data/`)
- **`load_data.py`**: Carga, limpieza y validación de datos crediticios
- **Funcionalidades**:
  - Generación de datos sintéticos para desarrollo
  - Limpieza y preprocesamiento de datos
  - Manejo de valores faltantes y outliers
  - Validación de tipos de datos

#### 2. Módulo de Características (`src/features/`)
- **`feature_engineering.py`**: Ingeniería de características para ML
- **Funcionalidades**:
  - Creación de ratios financieros
  - Codificación de variables categóricas
  - Escalamiento de variables numéricas
  - Transformación de características

#### 3. Módulo de Modelos (`src/models/`)
- **`train_model.py`**: Entrenamiento y evaluación de modelos
- **Modelos Implementados**:
  - Logistic Regression
  - Random Forest
  - XGBoost (opcional)
  - LightGBM (opcional)

#### 4. API REST (`api/`)
- **`app.py`**: Servidor Flask para predicciones
- **Endpoints**:
  - `/predict`: Predicción individual
  - `/predict_batch`: Predicción por lotes
  - `/model/info`: Información del modelo
  - `/health`: Estado de salud

#### 5. Dashboard (`dashboard/`)
- **`app.py`**: Interfaz web interactiva con Streamlit
- **Funcionalidades**:
  - Visualización de datos
  - Predictor individual
  - Análisis por lotes
  - Métricas del modelo

## Flujo de Trabajo

### 1. Preparación de Datos
```python
from src.data.load_data import DataLoader

loader = DataLoader()
df = loader.create_sample_data(n_samples=2000)
df_clean = loader.clean_data(df)
```

### 2. Ingeniería de Características
```python
from src.features.feature_engineering import FeatureEngineer

feature_engineer = FeatureEngineer()
df_features = feature_engineer.create_features(df_clean)
df_encoded = feature_engineer.encode_categorical_features(df_features)
X_train, X_test, y_train, y_test = feature_engineer.prepare_features(df_encoded)
```

### 3. Entrenamiento de Modelos
```python
from src.models.train_model import CreditRiskModel

model = CreditRiskModel()
model.train_models(X_train, y_train)
evaluation = model.evaluate_models(X_test, y_test)
```

### 4. Servir el Modelo
```python
# API REST
python api/app.py

# Dashboard
streamlit run dashboard/app.py
```

## Configuración

### Variables de Entorno (`.env`)
```bash
# API Configuration
API_HOST=0.0.0.0
API_PORT=8000

# Dashboard Configuration
DASHBOARD_HOST=0.0.0.0
DASHBOARD_PORT=8501

# Model Configuration
MODEL_RANDOM_STATE=42
MODEL_TEST_SIZE=0.2
```

### Configuración Principal (`config/config.py`)
- Rutas de archivos y directorios
- Parámetros de modelos
- Configuración de API y dashboard
- Umbrales de riesgo

## Métricas y Evaluación

### Métricas Implementadas
- **Accuracy**: Precisión general del modelo
- **Precision**: Precisión para la clase positiva (default)
- **Recall**: Sensibilidad para detectar defaults
- **F1-Score**: Balance entre precision y recall
- **AUC-ROC**: Área bajo la curva ROC
- **Confusion Matrix**: Matrix de confusión detallada

### Interpretación de Resultados
- **Riesgo Bajo**: Probabilidad < 30% → Aprobar
- **Riesgo Medio**: Probabilidad 30-70% → Revisar
- **Riesgo Alto**: Probabilidad > 70% → Rechazar

## Deployment

### Desarrollo Local
```bash
# Instalar dependencias
pip install -r requirements.txt

# Procesar datos
python src/data/load_data.py

# Entrenar modelo
python src/models/train_model.py

# Ejecutar API
python api/app.py

# Ejecutar dashboard
streamlit run dashboard/app.py
```

### Testing
```bash
# Ejecutar tests
python tests/test_models.py

# O usando pytest
pytest tests/ -v
```

### Estructura de Directorios de Salida
```
credit-risk/
├── data/
│   ├── processed/
│   │   ├── credit_data_processed.csv
│   │   └── credit_data_features.csv
├── models/
│   ├── random_forest_model.pkl
│   └── logistic_regression_model.pkl
├── plots/
│   ├── model_comparison.png
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   └── feature_importance.png
└── logs/
    └── credit_risk.log
```

## Extensibilidad

### Agregar Nuevos Modelos
1. Importar el modelo en `src/models/train_model.py`
2. Agregarlo al diccionario `models` en `initialize_models()`
3. Configurar hiperparámetros según necesidades

### Nuevas Características
1. Implementar lógica en `src/features/feature_engineering.py`
2. Agregar a los métodos `_create_*_features()`
3. Actualizar tests en `tests/test_models.py`

### Nuevos Endpoints de API
1. Agregar ruta en `api/app.py`
2. Implementar lógica de procesamiento
3. Documentar en README.md

## Consideraciones de Producción

### Seguridad
- Validación de entrada de datos
- Autenticación y autorización
- Logging de transacciones
- Encriptación de datos sensibles

### Escalabilidad
- Uso de bases de datos para persistencia
- Implementación de cache
- Balanceador de carga
- Monitoreo de performance

### Mantenimiento
- Reentrenamiento periódico del modelo
- Monitoreo de drift de datos
- Alertas automáticas de degradación
- Versionado de modelos

## Troubleshooting

### Problemas Comunes

#### Error: "Modelo no cargado"
- Verificar que existe un archivo `.pkl` en `/models/`
- Ejecutar `python src/models/train_model.py` primero

#### Error: "Datos no encontrados"
- Ejecutar `python src/data/load_data.py` para generar datos
- Verificar rutas en `config/config.py`

#### Error: "Puerto en uso"
- Cambiar puerto en configuración
- Terminar procesos existentes

#### Performance lento
- Reducir tamaño de datos de entrenamiento
- Usar modelos más simples para desarrollo
- Optimizar hiperparámetros