"""
Optimización de hiperparámetros para mejorar el modelo
Este script busca los mejores parámetros para Random Forest
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import joblib
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Cargar datos
logger.info("Cargando datos procesados...")
df = pd.read_csv('data/processed/real_credit_data_processed.csv')

# Preparar características (mismas 17 que usamos)
feature_columns = [
    'age', 'income', 'socioeconomic_level', 'dependents', 
    'gender', 'housing_status', 'has_disability',
    'invoice_value', 'invoice_to_income_ratio',
    'income_per_capita', 'stability_score', 'financial_burden',
    'age_risk', 'payment_capacity', 'socio_housing_score',
    'log_income', 'log_invoice'
]

X = df[feature_columns]
y = df['default']

logger.info(f"Datos: {X.shape[0]} registros, {X.shape[1]} características")

# Definir grid de hiperparámetros
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [10, 15, 20, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2'],
    'class_weight': ['balanced', 'balanced_subsample']
}

logger.info("Iniciando búsqueda de hiperparámetros...")
logger.info(f"Combinaciones a probar: {np.prod([len(v) for v in param_grid.values()])}")

# Grid Search con validación cruzada
rf = RandomForestClassifier(random_state=42, n_jobs=-1)
grid_search = GridSearchCV(
    estimator=rf,
    param_grid=param_grid,
    cv=5,
    scoring='roc_auc',
    n_jobs=-1,
    verbose=2
)

grid_search.fit(X, y)

logger.info(f"\n{'='*70}")
logger.info("RESULTADOS DE OPTIMIZACIÓN")
logger.info(f"{'='*70}")
logger.info(f"\n🏆 Mejor AUC-ROC: {grid_search.best_score_:.4f}")
logger.info(f"\n📊 Mejores hiperparámetros:")
for param, value in grid_search.best_params_.items():
    logger.info(f"   {param}: {value}")

# Guardar modelo optimizado
best_model = grid_search.best_estimator_
joblib.dump(best_model, 'models/real_random_forest_optimized_model.pkl')
logger.info(f"\n✅ Modelo optimizado guardado: models/real_random_forest_optimized_model.pkl")

# Comparar con modelo anterior
logger.info(f"\n📈 COMPARACIÓN:")
logger.info(f"   Modelo anterior: 0.6132")
logger.info(f"   Modelo optimizado: {grid_search.best_score_:.4f}")
logger.info(f"   Mejora: {(grid_search.best_score_ - 0.6132)*100:.2f} puntos porcentuales")
