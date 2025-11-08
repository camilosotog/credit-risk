"""
Script para procesar el dataset real DataCreditos_baland.csv
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.config import RAW_DATA_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def analyze_real_dataset():
    """Analizar el dataset real de créditos."""
    
    # Cargar los datos - AHORA USANDO docs/DataCreditos.csv
    data_path = Path('docs/DataCreditos.csv')
    
    if not data_path.exists():
        logger.error(f"Dataset no encontrado en: {data_path}")
        return None
    
    logger.info(f"Cargando dataset desde: {data_path}")
    df = pd.read_csv(data_path)
    
    # FILTRAR SOLO VIABILIDADES 1 (APROBADO) Y 4 (RECHAZADO)
    logger.info(f"Total de registros antes de filtrar: {len(df):,}")
    df = df[df['Viabilidad'].isin([1, 4])].copy()
    logger.info(f"✅ Registros filtrados (Viabilidad 1 y 4): {len(df):,}")
    
    # Análisis básico
    logger.info("="*60)
    logger.info("ANÁLISIS DEL DATASET REAL")
    logger.info("="*60)
    
    logger.info(f"Dimensiones: {df.shape[0]} filas x {df.shape[1]} columnas")
    logger.info(f"Memoria utilizada: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Columnas
    logger.info("\\nCOLUMNAS DISPONIBLES:")
    for i, col in enumerate(df.columns, 1):
        logger.info(f"{i:2d}. {col}")
    
    # Tipos de datos
    logger.info("\\nTIPOS DE DATOS:")
    logger.info(df.dtypes)
    
    # Estadísticas básicas
    logger.info("\\nESTADÍSTICAS BÁSICAS:")
    logger.info(df.describe())
    
    # Análisis de la variable objetivo (Viabilidad)
    logger.info("\\nANÁLISIS DE VARIABLE OBJETIVO (Viabilidad):")
    viabilidad_counts = df['Viabilidad'].value_counts().sort_index()
    logger.info("Distribución:")
    for value, count in viabilidad_counts.items():
        percentage = count / len(df) * 100
        logger.info(f"  Viabilidad {value}: {count:,} ({percentage:.1f}%)")
    
    # Valores faltantes
    logger.info("\\nVALORES FALTANTES:")
    missing = df.isnull().sum()
    missing_percent = (missing / len(df)) * 100
    
    for col in df.columns:
        if missing[col] > 0:
            logger.info(f"  {col}: {missing[col]} ({missing_percent[col]:.1f}%)")
    
    if missing.sum() == 0:
        logger.info("  ✅ No hay valores faltantes")
    
    # Análisis de variables categóricas
    logger.info("\\nVARIABLES CATEGÓRICAS:")
    categorical_cols = df.select_dtypes(include=['object']).columns
    for col in categorical_cols:
        unique_values = df[col].nunique()
        logger.info(f"  {col}: {unique_values} valores únicos")
        if unique_values <= 10:  # Mostrar valores si son pocos
            logger.info(f"    Valores: {df[col].unique().tolist()}")
    
    return df


def map_columns_to_standard(df):
    """Mapear columnas del dataset real a nombres estándar."""
    
    logger.info("\\nMAPEANDO COLUMNAS A FORMATO ESTÁNDAR...")
    
    # Mapeo de columnas
    column_mapping = {
        'Edad': 'age',
        'IngresoPrincipalMensual': 'income',
        'Genero': 'gender',
        'Estrato': 'socioeconomic_level',
        'PersonasACargo': 'dependents',
        'UstedEsArrendadorOPropietario': 'housing_status',
        'CiudadDeResidencia': 'city',
        'LineaDeCredito': 'credit_line',
        'ValorFactura': 'invoice_value',
        'UstedTieneAlgunaDiscapacidad': 'has_disability',
        'Viabilidad': 'target',  # Variable objetivo
        'ValorCupoAprobado': 'approved_limit'
    }
    
    # Renombrar columnas
    df_mapped = df.rename(columns=column_mapping)
    
    logger.info("Columnas mapeadas:")
    for old_name, new_name in column_mapping.items():
        logger.info(f"  {old_name} -> {new_name}")
    
    return df_mapped


def preprocess_real_data(df):
    """Preprocesar el dataset real."""
    
    logger.info("\\nPREPROCESANDO DATOS...")
    
    df_processed = df.copy()
    
    # Los valores ya están en escala real (no logarítmica)
    logger.info("Verificando rangos de valores monetarios...")
    
    if 'invoice_value' in df_processed.columns:
        logger.info(f"  invoice_value: Rango [${df_processed['invoice_value'].min():,.0f}, ${df_processed['invoice_value'].max():,.0f}]")
    
    if 'approved_limit' in df_processed.columns:
        logger.info(f"  approved_limit: Rango [${df_processed['approved_limit'].min():,.0f}, ${df_processed['approved_limit'].max():,.0f}]")
    
    # Convertir tipos de datos
    logger.info("Convirtiendo tipos de datos...")
    
    # Variables numéricas
    numeric_columns = ['age', 'income', 'invoice_value', 'approved_limit']
    for col in numeric_columns:
        if col in df_processed.columns:
            df_processed[col] = pd.to_numeric(df_processed[col], errors='coerce')
    
    # Crear características adicionales ANTES de convertir a categórica
    logger.info("Creando características adicionales...")
    
    # === CARACTERÍSTICAS BÁSICAS ===
    # Ratio de factura a ingresos
    if 'invoice_value' in df_processed.columns and 'income' in df_processed.columns:
        df_processed['invoice_to_income_ratio'] = df_processed['invoice_value'] / (df_processed['income'] + 1)
    
    # Ratio de límite aprobado a ingresos
    if 'approved_limit' in df_processed.columns and 'income' in df_processed.columns:
        df_processed['limit_to_income_ratio'] = df_processed['approved_limit'] / (df_processed['income'] + 1)
    
    # === NUEVAS CARACTERÍSTICAS AVANZADAS ===
    logger.info("Creando características avanzadas para mejorar precisión...")
    
    # 1. Ingreso per cápita (ingreso por persona en el hogar)
    if 'income' in df_processed.columns and 'dependents' in df_processed.columns:
        df_processed['income_per_capita'] = df_processed['income'] / (df_processed['dependents'] + 1)
    
    # 2. Score de estabilidad (edad * estrato)
    if 'age' in df_processed.columns and 'socioeconomic_level' in df_processed.columns:
        df_processed['stability_score'] = df_processed['age'] * df_processed['socioeconomic_level']
    
    # 3. Carga financiera (factura relativa al ingreso per cápita)
    if 'invoice_value' in df_processed.columns and 'income_per_capita' in df_processed.columns:
        df_processed['financial_burden'] = df_processed['invoice_value'] / (df_processed['income_per_capita'] + 1)
    
    # 4. Indicador de riesgo de edad (jóvenes y muy mayores son más riesgosos)
    if 'age' in df_processed.columns:
        df_processed['age_risk'] = df_processed['age'].apply(
            lambda x: 1 if x < 25 or x > 65 else 0
        )
    
    # 5. Capacidad de pago (ingresos menos factura)
    if 'income' in df_processed.columns and 'invoice_value' in df_processed.columns:
        df_processed['payment_capacity'] = df_processed['income'] - df_processed['invoice_value']
        df_processed['payment_capacity'] = df_processed['payment_capacity'].clip(lower=0)
    
    # 6. Score combinado estrato-vivienda
    if 'socioeconomic_level' in df_processed.columns and 'housing_status' in df_processed.columns:
        df_processed['socio_housing_score'] = df_processed['socioeconomic_level'] * (df_processed['housing_status'] + 1)
    
    # 7. Logaritmo del ingreso (para normalizar distribución)
    if 'income' in df_processed.columns:
        df_processed['log_income'] = np.log1p(df_processed['income'])
    
    # 8. Logaritmo del valor factura
    if 'invoice_value' in df_processed.columns:
        df_processed['log_invoice'] = np.log1p(df_processed['invoice_value'])
    
    # Categorías de edad
    if 'age' in df_processed.columns:
        df_processed['age_group'] = pd.cut(
            df_processed['age'],
            bins=[0, 25, 35, 45, 55, 65, 100],
            labels=['18-25', '26-35', '36-45', '46-55', '56-65', '65+']
        )
    
    # Categorías de ingresos
    if 'income' in df_processed.columns:
        df_processed['income_category'] = pd.qcut(
            df_processed['income'],
            q=5,
            labels=['Muy Bajo', 'Bajo', 'Medio', 'Alto', 'Muy Alto'],
            duplicates='drop'
        )
    
    logger.info(f"✅ Características avanzadas creadas: 8 nuevas variables")
    
    # AHORA sí convertir variables categóricas (después de crear features)
    categorical_columns = ['gender', 'housing_status', 'has_disability']
    for col in categorical_columns:
        if col in df_processed.columns:
            df_processed[col] = df_processed[col].astype('category')
    
    # Crear variable objetivo binaria
    if 'target' in df_processed.columns:
        # Mapear Viabilidad a variable binaria
        # Viabilidad 1 = APROBADO (bajo riesgo) -> default=0
        # Viabilidad 4 = RECHAZADO (alto riesgo) -> default=1
        df_processed['default'] = df_processed['target'].apply(
            lambda x: 1 if x == 4 else 0
        )
        logger.info(f"✅ Variable objetivo creada:")
        logger.info(f"   Viabilidad 1 (Aprobado) -> default=0: {(df_processed['default']==0).sum():,} casos")
        logger.info(f"   Viabilidad 4 (Rechazado) -> default=1: {(df_processed['default']==1).sum():,} casos")
    
    logger.info(f"Datos procesados: {df_processed.shape}")
    logger.info(f"Nuevas características creadas: {len(df_processed.columns) - len(df.columns)}")
    
    return df_processed


def save_processed_real_data(df):
    """Guardar datos procesados."""
    
    # Crear directorio si no existe
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Guardar datos procesados
    output_path = PROCESSED_DATA_DIR / 'real_credit_data_processed.csv'
    df.to_csv(output_path, index=False)
    
    logger.info(f"\\nDatos procesados guardados en: {output_path}")
    
    return output_path


def create_data_visualizations(df):
    """Crear visualizaciones básicas del dataset."""
    
    logger.info("\\nCREANDO VISUALIZACIONES...")
    
    # Crear directorio para gráficos
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # 1. Distribución de la variable objetivo
    if 'target' in df.columns:
        plt.figure(figsize=(10, 6))
        df['target'].value_counts().plot(kind='bar')
        plt.title('Distribución de Viabilidad')
        plt.xlabel('Viabilidad')
        plt.ylabel('Cantidad')
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(plots_dir / 'target_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 2. Distribución de edades
    if 'age' in df.columns:
        plt.figure(figsize=(10, 6))
        df['age'].hist(bins=30, alpha=0.7)
        plt.title('Distribución de Edades')
        plt.xlabel('Edad')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'age_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 3. Distribución de ingresos
    if 'income' in df.columns:
        plt.figure(figsize=(10, 6))
        df['income'].hist(bins=50, alpha=0.7)
        plt.title('Distribución de Ingresos')
        plt.xlabel('Ingreso Mensual')
        plt.ylabel('Frecuencia')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(plots_dir / 'income_distribution.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    # 4. Matriz de correlación
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 10))
        correlation_matrix = df[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', center=0, square=True)
        plt.title('Matriz de Correlación')
        plt.tight_layout()
        plt.savefig(plots_dir / 'correlation_matrix.png', dpi=300, bbox_inches='tight')
        plt.close()
    
    logger.info(f"Visualizaciones guardadas en: {plots_dir}")


def main():
    """Función principal."""
    
    logger.info("INICIANDO PROCESAMIENTO DEL DATASET REAL")
    logger.info("="*60)
    
    # 1. Analizar dataset
    df = analyze_real_dataset()
    
    if df is None:
        logger.error("No se pudo cargar el dataset")
        return
    
    # 2. Mapear columnas
    df_mapped = map_columns_to_standard(df)
    
    # 3. Preprocesar datos
    df_processed = preprocess_real_data(df_mapped)
    
    # 4. Guardar datos procesados
    output_path = save_processed_real_data(df_processed)
    
    # 5. Crear visualizaciones
    create_data_visualizations(df_processed)
    
    # 6. Resumen final
    logger.info("="*60)
    logger.info("PROCESAMIENTO COMPLETADO")
    logger.info("="*60)
    logger.info(f"✅ Dataset original: {df.shape}")
    logger.info(f"✅ Dataset procesado: {df_processed.shape}")
    logger.info(f"✅ Archivo guardado: {output_path}")
    logger.info(f"✅ Variable objetivo: {'default' if 'default' in df_processed.columns else 'target'}")
    if 'default' in df_processed.columns:
        default_rate = df_processed['default'].mean()
        logger.info(f"✅ Tasa de default: {default_rate:.2%}")
    
    logger.info("\\n🚀 SIGUIENTE PASO: Ejecutar entrenamiento de modelos")
    logger.info("   python src/models/train_model_real.py")


if __name__ == "__main__":
    main()