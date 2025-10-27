"""
Script para cargar y procesar datos crediticios.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Tuple, Optional
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, DATA_CONFIG

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DataLoader:
    """Clase para cargar y procesar datos crediticios."""
    
    def __init__(self):
        self.raw_data_dir = RAW_DATA_DIR
        self.processed_data_dir = PROCESSED_DATA_DIR
        self.data_config = DATA_CONFIG
        
    def load_raw_data(self, filename: str) -> pd.DataFrame:
        """
        Cargar datos crudos desde archivo.
        
        Args:
            filename: Nombre del archivo de datos
            
        Returns:
            DataFrame con los datos cargados
        """
        file_path = self.raw_data_dir / filename
        
        try:
            if filename.endswith('.csv'):
                df = pd.read_csv(file_path)
            elif filename.endswith(('.xlsx', '.xls')):
                df = pd.read_excel(file_path)
            else:
                raise ValueError(f"Formato de archivo no soportado: {filename}")
                
            logger.info(f"Datos cargados exitosamente: {df.shape[0]} filas, {df.shape[1]} columnas")
            return df
            
        except FileNotFoundError:
            logger.error(f"Archivo no encontrado: {file_path}")
            raise
        except Exception as e:
            logger.error(f"Error al cargar datos: {str(e)}")
            raise
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Limpiar y preprocesar datos.
        
        Args:
            df: DataFrame con datos crudos
            
        Returns:
            DataFrame con datos limpios
        """
        logger.info("Iniciando limpieza de datos...")
        
        # Crear copia para no modificar el original
        df_clean = df.copy()
        
        # Eliminar duplicados
        initial_rows = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        logger.info(f"Duplicados eliminados: {initial_rows - len(df_clean)}")
        
        # Manejar valores faltantes
        df_clean = self._handle_missing_values(df_clean)
        
        # Validar tipos de datos
        df_clean = self._validate_data_types(df_clean)
        
        # Detectar y manejar outliers
        df_clean = self._handle_outliers(df_clean)
        
        logger.info(f"Limpieza completada. Datos finales: {df_clean.shape[0]} filas")
        return df_clean
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Manejar valores faltantes."""
        logger.info("Manejando valores faltantes...")
        
        # Imputar valores numéricos con la mediana
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        for col in numerical_cols:
            if df[col].isnull().sum() > 0:
                median_value = df[col].median()
                df[col].fillna(median_value, inplace=True)
                logger.info(f"Imputados {col} con mediana: {median_value}")
        
        # Imputar valores categóricos con la moda
        categorical_cols = df.select_dtypes(include=['object']).columns
        for col in categorical_cols:
            if df[col].isnull().sum() > 0:
                mode_value = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
                df[col].fillna(mode_value, inplace=True)
                logger.info(f"Imputados {col} con moda: {mode_value}")
        
        return df
    
    def _validate_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Validar y corregir tipos de datos."""
        logger.info("Validando tipos de datos...")
        
        # Convertir columnas categóricas conocidas
        if 'categorical_columns' in self.data_config:
            for col in self.data_config['categorical_columns']:
                if col in df.columns:
                    df[col] = df[col].astype('category')
        
        # Asegurar que columnas numéricas sean numéricas
        if 'numerical_columns' in self.data_config:
            for col in self.data_config['numerical_columns']:
                if col in df.columns:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
        
        return df
    
    def _handle_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        """Detectar y manejar outliers usando IQR."""
        logger.info("Detectando outliers...")
        
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        outlier_counts = {}
        
        for col in numerical_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            outlier_counts[col] = len(outliers)
            
            # Cap outliers instead of removing them
            df[col] = np.where(df[col] < lower_bound, lower_bound, df[col])
            df[col] = np.where(df[col] > upper_bound, upper_bound, df[col])
        
        logger.info(f"Outliers detectados y manejados: {outlier_counts}")
        return df
    
    def create_sample_data(self, n_samples: int = 1000) -> pd.DataFrame:
        """
        Crear datos de muestra para pruebas.
        
        Args:
            n_samples: Número de muestras a generar
            
        Returns:
            DataFrame con datos sintéticos
        """
        logger.info(f"Generando {n_samples} muestras de datos sintéticos...")
        
        np.random.seed(42)
        
        data = {
            'customer_id': range(1, n_samples + 1),
            'age': np.random.normal(40, 12, n_samples).astype(int),
            'income': np.random.lognormal(10.5, 0.5, n_samples),
            'credit_score': np.random.normal(650, 100, n_samples).astype(int),
            'debt_ratio': np.random.beta(2, 5, n_samples),
            'employment_years': np.random.exponential(5, n_samples),
            'loan_amount': np.random.lognormal(9, 0.8, n_samples),
            'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
            'employment_status': np.random.choice(['Employed', 'Self-employed', 'Unemployed'], n_samples, p=[0.7, 0.25, 0.05]),
            'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], n_samples, p=[0.3, 0.4, 0.25, 0.05]),
            'marital_status': np.random.choice(['Single', 'Married', 'Divorced'], n_samples, p=[0.4, 0.5, 0.1]),
            'housing_status': np.random.choice(['Own', 'Rent', 'Mortgage'], n_samples, p=[0.3, 0.4, 0.3]),
            'loan_purpose': np.random.choice(['Personal', 'Auto', 'Home', 'Business'], n_samples, p=[0.4, 0.3, 0.2, 0.1])
        }
        
        df = pd.DataFrame(data)
        
        # Clip values to reasonable ranges
        df['age'] = np.clip(df['age'], 18, 80)
        df['credit_score'] = np.clip(df['credit_score'], 300, 850)
        df['debt_ratio'] = np.clip(df['debt_ratio'], 0, 1)
        df['employment_years'] = np.clip(df['employment_years'], 0, 40)
        
        # Create target variable (default) based on features
        default_prob = (
            0.1 +  # base probability
            (700 - df['credit_score']) / 1000 * 0.3 +  # credit score effect
            df['debt_ratio'] * 0.4 +  # debt ratio effect
            np.where(df['employment_status'] == 'Unemployed', 0.3, 0) +  # unemployment effect
            np.maximum(0, (df['loan_amount'] / df['income'] - 0.3)) * 0.2  # loan to income effect
        )
        
        df['default'] = np.random.binomial(1, np.clip(default_prob, 0, 1), n_samples)
        
        logger.info(f"Datos sintéticos generados. Tasa de default: {df['default'].mean():.2%}")
        return df
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> None:
        """
        Guardar datos procesados.
        
        Args:
            df: DataFrame a guardar
            filename: Nombre del archivo de salida
        """
        # Crear directorio si no existe
        self.processed_data_dir.mkdir(parents=True, exist_ok=True)
        
        file_path = self.processed_data_dir / filename
        
        try:
            if filename.endswith('.csv'):
                df.to_csv(file_path, index=False)
            elif filename.endswith(('.xlsx', '.xls')):
                df.to_excel(file_path, index=False)
            else:
                raise ValueError(f"Formato de archivo no soportado: {filename}")
                
            logger.info(f"Datos guardados exitosamente en: {file_path}")
            
        except Exception as e:
            logger.error(f"Error al guardar datos: {str(e)}")
            raise


def main():
    """Función principal para ejecutar el procesamiento de datos."""
    loader = DataLoader()
    
    # Generar datos de muestra si no existen datos reales
    logger.info("Generando datos de muestra...")
    df = loader.create_sample_data(n_samples=2000)
    
    # Limpiar datos
    df_clean = loader.clean_data(df)
    
    # Guardar datos procesados
    loader.save_processed_data(df_clean, 'credit_data_processed.csv')
    
    # Mostrar estadísticas básicas
    logger.info("Estadísticas de los datos procesados:")
    logger.info(f"Shape: {df_clean.shape}")
    logger.info(f"Tasa de default: {df_clean['default'].mean():.2%}")
    logger.info("Tipos de datos:")
    logger.info(df_clean.dtypes)


if __name__ == "__main__":
    main()