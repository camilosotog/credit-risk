"""
Script para ingeniería de características para el modelo de riesgo crediticio.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.model_selection import train_test_split
import logging
from typing import Tuple, List, Dict, Any
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.config import DATA_CONFIG, MODEL_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """Clase para ingeniería de características."""
    
    def __init__(self):
        self.data_config = DATA_CONFIG
        self.model_config = MODEL_CONFIG
        self.scalers = {}
        self.encoders = {}
        self.feature_names = []
        
    def create_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Crear nuevas características a partir de los datos existentes.
        
        Args:
            df: DataFrame con datos originales
            
        Returns:
            DataFrame con características adicionales
        """
        logger.info("Creando nuevas características...")
        
        df_features = df.copy()
        
        # Características financieras
        df_features = self._create_financial_features(df_features)
        
        # Características demográficas
        df_features = self._create_demographic_features(df_features)
        
        # Características de comportamiento crediticio
        df_features = self._create_credit_behavior_features(df_features)
        
        # Ratios e interacciones
        df_features = self._create_ratio_features(df_features)
        
        logger.info(f"Características creadas. Shape final: {df_features.shape}")
        return df_features
    
    def _create_financial_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear características financieras."""
        logger.info("Creando características financieras...")
        
        # Ratio de préstamo a ingresos
        df['loan_to_income_ratio'] = df['loan_amount'] / (df['income'] + 1)
        
        # Capacidad de pago mensual
        monthly_income = df['income'] / 12
        df['monthly_income'] = monthly_income
        
        # Pago mensual estimado (asumiendo interés del 10% anual)
        monthly_rate = 0.10 / 12
        df['estimated_monthly_payment'] = (
            df['loan_amount'] * monthly_rate * (1 + monthly_rate) ** df['loan_term']
        ) / ((1 + monthly_rate) ** df['loan_term'] - 1)
        
        # Ratio de pago mensual a ingresos
        df['payment_to_income_ratio'] = df['estimated_monthly_payment'] / (monthly_income + 1)
        
        # Ingreso disponible después del pago
        df['disposable_income'] = monthly_income - df['estimated_monthly_payment']
        
        # Categorías de ingresos
        df['income_category'] = pd.cut(
            df['income'], 
            bins=[0, 30000, 50000, 75000, 100000, float('inf')],
            labels=['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High']
        )
        
        return df
    
    def _create_demographic_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear características demográficas."""
        logger.info("Creando características demográficas...")
        
        # Grupos de edad
        df['age_group'] = pd.cut(
            df['age'],
            bins=[0, 25, 35, 45, 55, 65, float('inf')],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
        
        # Estabilidad laboral
        df['employment_stability'] = pd.cut(
            df['employment_years'],
            bins=[0, 1, 3, 5, 10, float('inf')],
            labels=['New', 'Short', 'Medium', 'Stable', 'Very Stable']
        )
        
        # Puntuación de estabilidad (combinando edad y empleo)
        df['stability_score'] = (
            (df['age'] - 18) / 62 * 0.4 +  # Normalizar edad
            np.minimum(df['employment_years'] / 20, 1) * 0.6  # Normalizar años de empleo
        )
        
        return df
    
    def _create_credit_behavior_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear características de comportamiento crediticio."""
        logger.info("Creando características de comportamiento crediticio...")
        
        # Categorías de puntaje crediticio
        df['credit_score_category'] = pd.cut(
            df['credit_score'],
            bins=[0, 580, 670, 740, 800, 850],
            labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent']
        )
        
        # Distancia del puntaje crediticio promedio
        avg_credit_score = df['credit_score'].mean()
        df['credit_score_deviation'] = df['credit_score'] - avg_credit_score
        
        # Categorías de ratio de deuda
        df['debt_ratio_category'] = pd.cut(
            df['debt_ratio'],
            bins=[0, 0.2, 0.4, 0.6, 0.8, 1.0],
            labels=['Very Low', 'Low', 'Medium', 'High', 'Very High']
        )
        
        # Puntuación de riesgo preliminar
        df['preliminary_risk_score'] = (
            (850 - df['credit_score']) / 550 * 0.4 +  # Puntaje crediticio invertido
            df['debt_ratio'] * 0.3 +  # Ratio de deuda
            np.minimum(df['loan_to_income_ratio'], 1) * 0.3  # Ratio préstamo/ingresos
        )
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Crear características de ratio e interacciones."""
        logger.info("Creando características de ratio...")
        
        # Interacción edad-ingresos
        df['age_income_interaction'] = df['age'] * np.log(df['income'] + 1)
        
        # Interacción puntaje crediticio-ingresos
        df['credit_income_interaction'] = df['credit_score'] * np.log(df['income'] + 1)
        
        # Ratio de experiencia laboral a edad
        df['employment_age_ratio'] = df['employment_years'] / (df['age'] - 18 + 1)
        
        # Puntuación compuesta de capacidad financiera
        df['financial_capacity_score'] = (
            np.log(df['income'] + 1) / 12 * 0.3 +  # Ingresos normalizados
            (df['credit_score'] / 850) * 0.4 +  # Puntaje crediticio normalizado
            (1 - df['debt_ratio']) * 0.3  # Inverso del ratio de deuda
        )
        
        return df
    
    def encode_categorical_features(self, df: pd.DataFrame, 
                                   fit: bool = True) -> pd.DataFrame:
        """
        Codificar características categóricas.
        
        Args:
            df: DataFrame con características
            fit: Si debe ajustar los encoders
            
        Returns:
            DataFrame con características codificadas
        """
        logger.info("Codificando características categóricas...")
        
        df_encoded = df.copy()
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        # Excluir la columna objetivo y ID si existen
        exclude_cols = []
        if self.data_config.get('target_column') in categorical_cols:
            exclude_cols.append(self.data_config['target_column'])
        if self.data_config.get('id_column') in categorical_cols:
            exclude_cols.append(self.data_config['id_column'])
        
        categorical_cols = [col for col in categorical_cols if col not in exclude_cols]
        
        for col in categorical_cols:
            if col not in df_encoded.columns:
                continue
                
            if fit:
                # Usar LabelEncoder para variables binarias u ordinales
                if df_encoded[col].nunique() <= 2:
                    le = LabelEncoder()
                    df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
                    self.encoders[col] = le
                else:
                    # Usar One-Hot Encoding para variables categóricas múltiples
                    dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                    df_encoded = pd.concat([df_encoded.drop(col, axis=1), dummies], axis=1)
                    self.encoders[col] = list(dummies.columns)
            else:
                # Aplicar encoders ya entrenados
                if col in self.encoders:
                    if isinstance(self.encoders[col], LabelEncoder):
                        # Manejar categorías no vistas
                        unique_vals = df_encoded[col].unique()
                        le = self.encoders[col]
                        for val in unique_vals:
                            if val not in le.classes_:
                                le.classes_ = np.append(le.classes_, val)
                        df_encoded[col] = le.transform(df_encoded[col].astype(str))
                    else:
                        # One-hot encoding
                        dummies = pd.get_dummies(df_encoded[col], prefix=col, drop_first=True)
                        df_encoded = df_encoded.drop(col, axis=1)
                        for dummy_col in self.encoders[col]:
                            if dummy_col in dummies.columns:
                                df_encoded[dummy_col] = dummies[dummy_col]
                            else:
                                df_encoded[dummy_col] = 0
        
        return df_encoded
    
    def scale_numerical_features(self, df: pd.DataFrame, 
                                fit: bool = True) -> pd.DataFrame:
        """
        Escalar características numéricas.
        
        Args:
            df: DataFrame con características
            fit: Si debe ajustar los scalers
            
        Returns:
            DataFrame con características escaladas
        """
        logger.info("Escalando características numéricas...")
        
        df_scaled = df.copy()
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        
        # Excluir columnas de identificación y objetivo
        exclude_cols = []
        if self.data_config.get('target_column') in numerical_cols:
            exclude_cols.append(self.data_config['target_column'])
        if self.data_config.get('id_column') in numerical_cols:
            exclude_cols.append(self.data_config['id_column'])
        
        numerical_cols = [col for col in numerical_cols if col not in exclude_cols]
        
        if fit:
            scaler = StandardScaler()
            df_scaled[numerical_cols] = scaler.fit_transform(df_scaled[numerical_cols])
            self.scalers['numerical'] = scaler
        else:
            if 'numerical' in self.scalers:
                scaler = self.scalers['numerical']
                # Solo escalar columnas que existen tanto en los datos como en el scaler
                cols_to_scale = [col for col in numerical_cols if col in scaler.feature_names_in_]
                if cols_to_scale:
                    df_scaled[cols_to_scale] = scaler.transform(df_scaled[cols_to_scale])
        
        return df_scaled
    
    def prepare_features(self, df: pd.DataFrame, 
                        target_col: str = None,
                        test_size: float = None) -> Tuple[pd.DataFrame, pd.DataFrame, 
                                                        pd.Series, pd.Series]:
        """
        Preparar características para entrenamiento.
        
        Args:
            df: DataFrame con datos
            target_col: Nombre de la columna objetivo
            test_size: Proporción de datos para test
            
        Returns:
            X_train, X_test, y_train, y_test
        """
        logger.info("Preparando características para entrenamiento...")
        
        # Usar configuración por defecto si no se especifica
        if target_col is None:
            target_col = self.data_config.get('target_column', 'default')
        if test_size is None:
            test_size = self.model_config.get('test_size', 0.2)
        
        # Separar características y objetivo
        if target_col not in df.columns:
            raise ValueError(f"Columna objetivo '{target_col}' no encontrada en los datos")
        
        X = df.drop(columns=[target_col])
        y = df[target_col]
        
        # Eliminar columna de ID si existe
        id_col = self.data_config.get('id_column')
        if id_col and id_col in X.columns:
            X = X.drop(columns=[id_col])
        
        # Dividir en train/test
        random_state = self.model_config.get('random_state', 42)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, stratify=y
        )
        
        logger.info(f"División completada:")
        logger.info(f"Train: {X_train.shape[0]} muestras")
        logger.info(f"Test: {X_test.shape[0]} muestras")
        logger.info(f"Características: {X_train.shape[1]}")
        
        # Guardar nombres de características
        self.feature_names = list(X_train.columns)
        
        return X_train, X_test, y_train, y_test
    
    def get_feature_info(self) -> Dict[str, Any]:
        """
        Obtener información sobre las características procesadas.
        
        Returns:
            Diccionario con información de características
        """
        return {
            'feature_names': self.feature_names,
            'n_features': len(self.feature_names),
            'encoders': list(self.encoders.keys()),
            'scalers': list(self.scalers.keys())
        }


def main():
    """Función principal para ejecutar ingeniería de características."""
    from src.data.load_data import DataLoader
    
    # Cargar datos
    loader = DataLoader()
    df = pd.read_csv(loader.processed_data_dir / 'credit_data_processed.csv')
    
    # Ingeniería de características
    feature_engineer = FeatureEngineer()
    
    # Crear nuevas características
    df_features = feature_engineer.create_features(df)
    
    # Codificar características categóricas
    df_encoded = feature_engineer.encode_categorical_features(df_features, fit=True)
    
    # Escalar características numéricas
    df_scaled = feature_engineer.scale_numerical_features(df_encoded, fit=True)
    
    # Preparar para entrenamiento
    X_train, X_test, y_train, y_test = feature_engineer.prepare_features(df_scaled)
    
    # Mostrar información
    feature_info = feature_engineer.get_feature_info()
    logger.info(f"Información de características: {feature_info}")
    
    # Guardar datos procesados
    logger.info("Guardando datos con características...")
    df_scaled.to_csv(loader.processed_data_dir / 'credit_data_features.csv', index=False)
    
    logger.info("Ingeniería de características completada!")


if __name__ == "__main__":
    main()