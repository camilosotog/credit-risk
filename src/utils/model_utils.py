"""
Utilidades para el manejo correcto de datos en modelos de Machine Learning.
Evita warnings sobre nombres de características.
"""

import pandas as pd
import numpy as np
from typing import Any, List, Optional
import logging

logger = logging.getLogger(__name__)


def safe_predict(model, data, feature_names: Optional[List[str]] = None):
    """
    Realizar predicción segura manteniendo nombres de características.
    
    Args:
        model: Modelo de ML entrenado
        data: Datos para predicción (DataFrame o array)
        feature_names: Nombres de características (opcional)
    
    Returns:
        Predicciones del modelo
    """
    try:
        # Si es array, convertir a DataFrame con nombres
        if isinstance(data, np.ndarray) and feature_names:
            data = pd.DataFrame(data, columns=feature_names)
        
        # Si es DataFrame, asegurar orden de columnas
        elif isinstance(data, pd.DataFrame) and feature_names:
            data = data.reindex(columns=feature_names, fill_value=0)
        
        return model.predict(data)
        
    except Exception as e:
        logger.error(f"Error en predicción segura: {str(e)}")
        raise


def safe_predict_proba(model, data, feature_names: Optional[List[str]] = None):
    """
    Realizar predicción de probabilidades segura manteniendo nombres de características.
    
    Args:
        model: Modelo de ML entrenado
        data: Datos para predicción (DataFrame o array)
        feature_names: Nombres de características (opcional)
    
    Returns:
        Probabilidades predichas por el modelo
    """
    try:
        # Si es array, convertir a DataFrame con nombres
        if isinstance(data, np.ndarray) and feature_names:
            data = pd.DataFrame(data, columns=feature_names)
        
        # Si es DataFrame, asegurar orden de columnas
        elif isinstance(data, pd.DataFrame) and feature_names:
            data = data.reindex(columns=feature_names, fill_value=0)
        
        return model.predict_proba(data)
        
    except Exception as e:
        logger.error(f"Error en predicción de probabilidades segura: {str(e)}")
        raise


def safe_transform(scaler, data, feature_names: Optional[List[str]] = None, keep_dataframe: bool = True):
    """
    Aplicar transformación manteniendo nombres de características.
    
    Args:
        scaler: Transformador (StandardScaler, etc.)
        data: Datos para transformar
        feature_names: Nombres de características
        keep_dataframe: Si mantener como DataFrame
    
    Returns:
        Datos transformados
    """
    try:
        # Aplicar transformación
        if hasattr(scaler, 'transform'):
            transformed_data = scaler.transform(data)
        else:
            transformed_data = data
        
        # Mantener como DataFrame si se solicita
        if keep_dataframe and feature_names:
            if isinstance(data, pd.DataFrame):
                return pd.DataFrame(
                    transformed_data, 
                    columns=feature_names, 
                    index=data.index
                )
            else:
                return pd.DataFrame(transformed_data, columns=feature_names)
        
        return transformed_data
        
    except Exception as e:
        logger.error(f"Error en transformación segura: {str(e)}")
        raise


class ModelPredictor:
    """
    Clase auxiliar para hacer predicciones de manera consistente.
    """
    
    def __init__(self, model, scaler=None, feature_names=None, label_encoders=None):
        self.model = model
        self.scaler = scaler
        self.feature_names = feature_names or []
        self.label_encoders = label_encoders or {}
    
    def preprocess(self, data):
        """Preprocesar datos de entrada."""
        
        # Convertir a DataFrame si no lo es
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame([data] if isinstance(data, dict) else data)
        
        # Aplicar codificación de variables categóricas
        for col, encoder in self.label_encoders.items():
            if col in data.columns:
                try:
                    data[col] = encoder.transform(data[col].astype(str))
                except:
                    data[col] = 0
        
        # Asegurar que tenemos todas las características
        for feature in self.feature_names:
            if feature not in data.columns:
                data[feature] = 0
        
        # Reordenar columnas
        data = data.reindex(columns=self.feature_names, fill_value=0)
        
        # Aplicar escalado manteniendo DataFrame
        if self.scaler:
            data = safe_transform(
                self.scaler, 
                data, 
                self.feature_names, 
                keep_dataframe=True
            )
        
        return data
    
    def predict(self, data):
        """Predicción segura."""
        processed_data = self.preprocess(data)
        return safe_predict(self.model, processed_data, self.feature_names)
    
    def predict_proba(self, data):
        """Predicción de probabilidades segura."""
        processed_data = self.preprocess(data)
        return safe_predict_proba(self.model, processed_data, self.feature_names)


def load_model_safely(model_path):
    """
    Cargar modelo y crear predictor seguro.
    
    Args:
        model_path: Ruta al archivo del modelo
    
    Returns:
        ModelPredictor configurado
    """
    import joblib
    
    try:
        model_data = joblib.load(model_path)
        
        predictor = ModelPredictor(
            model=model_data['model'],
            scaler=model_data.get('scaler'),
            feature_names=model_data.get('feature_names', []),
            label_encoders=model_data.get('label_encoders', {})
        )
        
        logger.info(f"Modelo cargado correctamente: {model_path}")
        return predictor
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        raise


# Función de conveniencia para uso directo
def predict_with_model(model_path, input_data):
    """
    Función de conveniencia para hacer predicciones directas.
    
    Args:
        model_path: Ruta al modelo
        input_data: Datos de entrada
    
    Returns:
        Predicción y probabilidad
    """
    predictor = load_model_safely(model_path)
    
    prediction_proba = predictor.predict_proba(input_data)
    prediction_class = predictor.predict(input_data)
    
    return {
        'prediction_class': int(prediction_class[0]),
        'default_probability': float(prediction_proba[0, 1]),
        'no_default_probability': float(prediction_proba[0, 0])
    }