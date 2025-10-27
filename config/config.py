"""
Configuración principal del proyecto de evaluación de riesgo crediticio.
"""

import os
from pathlib import Path

# Rutas del proyecto
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Crear directorios si no existen
LOGS_DIR.mkdir(exist_ok=True)

# Configuración de modelos
MODEL_CONFIG = {
    'random_state': 42,
    'test_size': 0.2,
    'validation_size': 0.2,
    'cv_folds': 5,
    'scoring_metric': 'roc_auc'
}

# Configuración de datos
DATA_CONFIG = {
    'target_column': 'default',
    'id_column': 'customer_id',
    'categorical_columns': [
        'employment_status', 'education_level', 'marital_status',
        'housing_status', 'loan_purpose'
    ],
    'numerical_columns': [
        'age', 'income', 'credit_score', 'debt_ratio',
        'employment_years', 'loan_amount', 'loan_term'
    ]
}

# Umbrales de riesgo
RISK_THRESHOLDS = {
    'low_risk': 0.3,
    'medium_risk': 0.7,
    'high_risk': 1.0
}

# Configuración de API
API_CONFIG = {
    'host': '0.0.0.0',
    'port': 8000,
    'debug': False,
    'workers': 4
}

# Configuración de Dashboard
DASHBOARD_CONFIG = {
    'host': '0.0.0.0',
    'port': 8501,
    'title': 'Sistema de Evaluación de Riesgo Crediticio',
    'layout': 'wide'
}

# Logging
LOGGING_CONFIG = {
    'version': 1,
    'disable_existing_loggers': False,
    'formatters': {
        'standard': {
            'format': '%(asctime)s [%(levelname)s] %(name)s: %(message)s'
        },
    },
    'handlers': {
        'default': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.StreamHandler',
        },
        'file': {
            'level': 'INFO',
            'formatter': 'standard',
            'class': 'logging.FileHandler',
            'filename': LOGS_DIR / 'credit_risk.log',
            'mode': 'a',
        },
    },
    'loggers': {
        '': {
            'handlers': ['default', 'file'],
            'level': 'INFO',
            'propagate': False
        }
    }
}