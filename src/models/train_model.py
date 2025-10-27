"""
Script para entrenar modelos de Machine Learning para evaluación de riesgo crediticio.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
from sklearn.model_selection import cross_val_score, GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
from typing import Dict, Any, Tuple, List
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.config import MODELS_DIR, MODEL_CONFIG, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class CreditRiskModel:
    """Clase para entrenar y evaluar modelos de riesgo crediticio."""
    
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        
    def initialize_models(self) -> Dict[str, Any]:
        """
        Inicializar modelos de Machine Learning.
        
        Returns:
            Diccionario con modelos inicializados
        """
        logger.info("Inicializando modelos...")
        
        random_state = MODEL_CONFIG.get('random_state', 42)
        
        models = {
            'logistic_regression': LogisticRegression(
                random_state=random_state,
                max_iter=1000,
                solver='liblinear'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            )
        }
        
        # Intentar importar XGBoost si está disponible
        try:
            import xgboost as xgb
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss'
            )
            logger.info("XGBoost agregado a los modelos")
        except ImportError:
            logger.warning("XGBoost no disponible, continuando sin él")
        
        # Intentar importar LightGBM si está disponible
        try:
            import lightgbm as lgb
            models['lightgbm'] = lgb.LGBMClassifier(
                n_estimators=100,
                random_state=random_state,
                max_depth=6,
                learning_rate=0.1,
                verbose=-1
            )
            logger.info("LightGBM agregado a los modelos")
        except ImportError:
            logger.warning("LightGBM no disponible, continuando sin él")
        
        self.models = models
        logger.info(f"Modelos inicializados: {list(models.keys())}")
        return models
    
    def train_models(self, X_train: pd.DataFrame, y_train: pd.Series,
                    X_val: pd.DataFrame = None, y_val: pd.Series = None) -> Dict[str, Any]:
        """
        Entrenar todos los modelos.
        
        Args:
            X_train: Características de entrenamiento
            y_train: Variables objetivo de entrenamiento
            X_val: Características de validación (opcional)
            y_val: Variables objetivo de validación (opcional)
            
        Returns:
            Diccionario con modelos entrenados
        """
        logger.info("Iniciando entrenamiento de modelos...")
        
        if not self.models:
            self.initialize_models()
        
        self.feature_names = list(X_train.columns)
        trained_models = {}
        
        for name, model in self.models.items():
            logger.info(f"Entrenando {name}...")
            
            try:
                # Entrenamiento básico
                model.fit(X_train, y_train)
                
                # Validación cruzada
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=MODEL_CONFIG.get('cv_folds', 5),
                    scoring=MODEL_CONFIG.get('scoring_metric', 'roc_auc')
                )
                
                trained_models[name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"{name} - CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"Error entrenando {name}: {str(e)}")
                continue
        
        self.models = {name: info['model'] for name, info in trained_models.items()}
        self.model_scores = {
            name: {'cv_mean': info['cv_mean'], 'cv_std': info['cv_std']}
            for name, info in trained_models.items()
        }
        
        logger.info("Entrenamiento completado")
        return self.models
    
    def evaluate_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> Dict[str, Dict[str, float]]:
        """
        Evaluar modelos en conjunto de prueba.
        
        Args:
            X_test: Características de prueba
            y_test: Variables objetivo de prueba
            
        Returns:
            Diccionario con métricas de evaluación
        """
        logger.info("Evaluando modelos...")
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            logger.info(f"Evaluando {name}...")
            
            try:
                # Predicciones
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Métricas
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'f1_score': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                # Agregar métricas de CV
                if name in self.model_scores:
                    metrics.update(self.model_scores[name])
                
                evaluation_results[name] = metrics
                
                logger.info(f"{name} - Accuracy: {metrics['accuracy']:.4f}, "
                           f"F1: {metrics['f1_score']:.4f}, AUC: {metrics['roc_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"Error evaluando {name}: {str(e)}")
                continue
        
        # Seleccionar mejor modelo basado en AUC
        if evaluation_results:
            best_model_name = max(evaluation_results.keys(), 
                                key=lambda x: evaluation_results[x]['roc_auc'])
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
            
            logger.info(f"Mejor modelo: {best_model_name} "
                       f"(AUC: {evaluation_results[best_model_name]['roc_auc']:.4f})")
        
        return evaluation_results
    
    def plot_model_comparison(self, evaluation_results: Dict[str, Dict[str, float]],
                            save_path: str = None) -> None:
        """
        Crear gráfico de comparación de modelos.
        
        Args:
            evaluation_results: Resultados de evaluación
            save_path: Ruta para guardar el gráfico
        """
        logger.info("Creando gráfico de comparación...")
        
        # Preparar datos para el gráfico
        models = list(evaluation_results.keys())
        metrics = ['accuracy', 'f1_score', 'roc_auc']
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        for i, metric in enumerate(metrics):
            values = [evaluation_results[model][metric] for model in models]
            
            bars = axes[i].bar(models, values, alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            
            # Agregar valores en las barras
            for bar, value in zip(bars, values):
                axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'{value:.3f}', ha='center', va='bottom')
            
            # Rotar etiquetas si hay muchos modelos
            if len(models) > 3:
                axes[i].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico guardado en: {save_path}")
        
        plt.show()
    
    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series,
                            model_name: str = None, save_path: str = None) -> None:
        """
        Crear matriz de confusión para un modelo.
        
        Args:
            X_test: Características de prueba
            y_test: Variables objetivo de prueba
            model_name: Nombre del modelo (usa el mejor si no se especifica)
            save_path: Ruta para guardar el gráfico
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        if model is None:
            logger.error("No hay modelo disponible para crear matriz de confusión")
            return
        
        logger.info(f"Creando matriz de confusión para {model_name}...")
        
        # Predicciones
        y_pred = model.predict(X_test)
        
        # Matriz de confusión
        cm = confusion_matrix(y_test, y_pred)
        
        # Crear gráfico
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['No Default', 'Default'],
                   yticklabels=['No Default', 'Default'])
        plt.title(f'Matriz de Confusión - {model_name}')
        plt.ylabel('Valores Reales')
        plt.xlabel('Predicciones')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Matriz de confusión guardada en: {save_path}")
        
        plt.show()
    
    def plot_roc_curves(self, X_test: pd.DataFrame, y_test: pd.Series,
                       save_path: str = None) -> None:
        """
        Crear curvas ROC para todos los modelos.
        
        Args:
            X_test: Características de prueba
            y_test: Variables objetivo de prueba
            save_path: Ruta para guardar el gráfico
        """
        logger.info("Creando curvas ROC...")
        
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            try:
                # Predicciones de probabilidad
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Curva ROC
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc = roc_auc_score(y_test, y_pred_proba)
                
                # Plotear
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
                
            except Exception as e:
                logger.error(f"Error creando ROC para {name}: {str(e)}")
                continue
        
        # Línea diagonal de referencia
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curvas ROC - Comparación de Modelos')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Curvas ROC guardadas en: {save_path}")
        
        plt.show()
    
    def get_feature_importance(self, model_name: str = None, top_n: int = 15) -> pd.DataFrame:
        """
        Obtener importancia de características.
        
        Args:
            model_name: Nombre del modelo (usa el mejor si no se especifica)
            top_n: Número de características principales a mostrar
            
        Returns:
            DataFrame con importancia de características
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        if model is None:
            logger.error("No hay modelo disponible para obtener importancia")
            return pd.DataFrame()
        
        logger.info(f"Obteniendo importancia de características para {model_name}...")
        
        try:
            # Obtener importancia según el tipo de modelo
            if hasattr(model, 'feature_importances_'):
                importance = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importance = np.abs(model.coef_[0])
            else:
                logger.warning(f"No se puede obtener importancia para {model_name}")
                return pd.DataFrame()
            
            # Crear DataFrame
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(top_n)
            
            return feature_importance_df
            
        except Exception as e:
            logger.error(f"Error obteniendo importancia: {str(e)}")
            return pd.DataFrame()
    
    def plot_feature_importance(self, model_name: str = None, top_n: int = 15,
                              save_path: str = None) -> None:
        """
        Crear gráfico de importancia de características.
        
        Args:
            model_name: Nombre del modelo (usa el mejor si no se especifica)
            top_n: Número de características principales a mostrar
            save_path: Ruta para guardar el gráfico
        """
        feature_importance_df = self.get_feature_importance(model_name, top_n)
        
        if feature_importance_df.empty:
            return
        
        if model_name is None:
            model_name = self.best_model_name
        
        logger.info(f"Creando gráfico de importancia para {model_name}...")
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance_df, x='importance', y='feature')
        plt.title(f'Importancia de Características - {model_name}')
        plt.xlabel('Importancia')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Gráfico de importancia guardado en: {save_path}")
        
        plt.show()
    
    def save_model(self, model_name: str = None, filename: str = None) -> str:
        """
        Guardar modelo entrenado.
        
        Args:
            model_name: Nombre del modelo (usa el mejor si no se especifica)
            filename: Nombre del archivo (genera automáticamente si no se especifica)
            
        Returns:
            Ruta del archivo guardado
        """
        if model_name is None:
            model_name = self.best_model_name
            model = self.best_model
        else:
            model = self.models[model_name]
        
        if model is None:
            logger.error("No hay modelo para guardar")
            return ""
        
        # Crear directorio si no existe
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f'{model_name}_model.pkl'
        
        filepath = MODELS_DIR / filename
        
        try:
            # Guardar modelo y metadatos
            model_data = {
                'model': model,
                'model_name': model_name,
                'feature_names': self.feature_names,
                'model_scores': self.model_scores.get(model_name, {})
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"Modelo guardado en: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"Error guardando modelo: {str(e)}")
            return ""
    
    def load_model(self, filepath: str) -> bool:
        """
        Cargar modelo guardado.
        
        Args:
            filepath: Ruta del archivo del modelo
            
        Returns:
            True si se cargó exitosamente
        """
        try:
            model_data = joblib.load(filepath)
            
            self.best_model = model_data['model']
            self.best_model_name = model_data['model_name']
            self.feature_names = model_data.get('feature_names', [])
            
            # Agregar a la colección de modelos
            self.models[self.best_model_name] = self.best_model
            
            logger.info(f"Modelo cargado exitosamente desde: {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Error cargando modelo: {str(e)}")
            return False
    
    def predict(self, X: pd.DataFrame, return_proba: bool = False) -> np.ndarray:
        """
        Realizar predicciones con el mejor modelo.
        
        Args:
            X: Características para predicción
            return_proba: Si devolver probabilidades
            
        Returns:
            Predicciones o probabilidades
        """
        if self.best_model is None:
            logger.error("No hay modelo disponible para predicción")
            return np.array([])
        
        try:
            if return_proba:
                return self.best_model.predict_proba(X)[:, 1]
            else:
                return self.best_model.predict(X)
                
        except Exception as e:
            logger.error(f"Error en predicción: {str(e)}")
            return np.array([])


def main():
    """Función principal para entrenar modelos."""
    from src.data.load_data import DataLoader
    from src.features.feature_engineering import FeatureEngineer
    
    # Cargar y preparar datos
    loader = DataLoader()
    
    # Verificar si existen datos procesados
    data_path = PROCESSED_DATA_DIR / 'credit_data_processed.csv'
    if not data_path.exists():
        logger.info("Generando datos de muestra...")
        df = loader.create_sample_data(n_samples=2000)
        df_clean = loader.clean_data(df)
        loader.save_processed_data(df_clean, 'credit_data_processed.csv')
    else:
        df_clean = pd.read_csv(data_path)
    
    # Ingeniería de características
    feature_engineer = FeatureEngineer()
    df_features = feature_engineer.create_features(df_clean)
    df_encoded = feature_engineer.encode_categorical_features(df_features, fit=True)
    df_scaled = feature_engineer.scale_numerical_features(df_encoded, fit=True)
    
    # Preparar datos para entrenamiento
    X_train, X_test, y_train, y_test = feature_engineer.prepare_features(df_scaled)
    
    # Entrenar modelos
    credit_model = CreditRiskModel()
    credit_model.train_models(X_train, y_train)
    
    # Evaluar modelos
    evaluation_results = credit_model.evaluate_models(X_test, y_test)
    
    # Crear visualizaciones
    logger.info("Creando visualizaciones...")
    
    # Crear directorio para gráficos
    plots_dir = Path("plots")
    plots_dir.mkdir(exist_ok=True)
    
    # Gráfico de comparación
    credit_model.plot_model_comparison(
        evaluation_results,
        save_path=plots_dir / "model_comparison.png"
    )
    
    # Matriz de confusión
    credit_model.plot_confusion_matrix(
        X_test, y_test,
        save_path=plots_dir / "confusion_matrix.png"
    )
    
    # Curvas ROC
    credit_model.plot_roc_curves(
        X_test, y_test,
        save_path=plots_dir / "roc_curves.png"
    )
    
    # Importancia de características
    credit_model.plot_feature_importance(
        save_path=plots_dir / "feature_importance.png"
    )
    
    # Guardar mejor modelo
    model_path = credit_model.save_model()
    logger.info(f"Mejor modelo guardado en: {model_path}")
    
    # Mostrar resumen final
    logger.info("\n=== RESUMEN FINAL ===")
    logger.info(f"Mejor modelo: {credit_model.best_model_name}")
    logger.info("Resultados de evaluación:")
    for model, metrics in evaluation_results.items():
        logger.info(f"{model}:")
        for metric, value in metrics.items():
            logger.info(f"  {metric}: {value:.4f}")


if __name__ == "__main__":
    main()