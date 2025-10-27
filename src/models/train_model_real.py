"""
Script para entrenar modelos con el dataset real de créditos.
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    roc_curve, precision_recall_curve, f1_score, accuracy_score
)
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import warnings
import sys
import os

# Suprimir warnings
warnings.filterwarnings('ignore')

# Agregar el directorio raíz al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from config.config import MODELS_DIR, PROCESSED_DATA_DIR

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RealCreditRiskModel:
    """Clase para entrenar modelos con datos reales de crédito."""
    
    def __init__(self):
        self.models = {}
        self.model_scores = {}
        self.best_model = None
        self.best_model_name = None
        self.feature_names = []
        self.label_encoders = {}
        self.scaler = None
        
    def load_real_data(self):
        """Cargar datos reales procesados."""
        
        data_path = PROCESSED_DATA_DIR / 'real_credit_data_processed.csv'
        
        if not data_path.exists():
            logger.error(f"Datos procesados no encontrados: {data_path}")
            logger.info("Ejecuta primero: python src/data/process_real_data.py")
            return None
        
        logger.info(f"Cargando datos desde: {data_path}")
        df = pd.read_csv(data_path)
        
        logger.info(f"Datos cargados: {df.shape[0]} filas, {df.shape[1]} columnas")
        return df
    
    def prepare_features(self, df):
        """Preparar características para entrenamiento."""
        
        logger.info("Preparando características...")
        
        # Crear copia de trabajo
        df_work = df.copy()
        
        # Seleccionar características principales
        feature_columns = [
            'age', 'income', 'socioeconomic_level', 'dependents',
            'gender', 'housing_status', 'has_disability',
            'invoice_value', 'approved_limit',
            'invoice_to_income_ratio', 'limit_to_income_ratio'
        ]
        
        # Filtrar columnas que existen
        available_features = [col for col in feature_columns if col in df_work.columns]
        logger.info(f"Características disponibles: {len(available_features)}")
        
        # Preparar conjunto de características
        X = df_work[available_features].copy()
        y = df_work['default'].copy()
        
        # Codificar variables categóricas
        categorical_columns = X.select_dtypes(include=['object', 'category']).columns
        
        for col in categorical_columns:
            if col in X.columns:
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Codificada variable: {col}")
        
        # Manejar valores faltantes
        X = X.fillna(X.median())
        
        # Escalar características numéricas
        self.scaler = StandardScaler()
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        self.feature_names = list(X.columns)
        
        logger.info(f"Características preparadas: {X_scaled.shape}")
        logger.info(f"Distribución objetivo - No Default: {(y==0).sum()}, Default: {(y==1).sum()}")
        
        return X_scaled, y
    
    def initialize_models(self):
        """Inicializar modelos para entrenamiento."""
        
        logger.info("Inicializando modelos...")
        
        models = {
            'logistic_regression': LogisticRegression(
                random_state=42,
                max_iter=1000,
                solver='liblinear'
            ),
            'random_forest': RandomForestClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2
            ),
            'gradient_boosting': GradientBoostingClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1
            ),
            'svm': SVC(
                random_state=42,
                probability=True,
                kernel='rbf',
                C=1.0
            )
        }
        
        # Intentar agregar XGBoost si está disponible
        try:
            import xgboost as xgb
            models['xgboost'] = xgb.XGBClassifier(
                n_estimators=100,
                random_state=42,
                max_depth=6,
                learning_rate=0.1,
                eval_metric='logloss'
            )
            logger.info("XGBoost agregado")
        except ImportError:
            logger.warning("XGBoost no disponible")
        
        self.models = models
        logger.info(f"Modelos inicializados: {list(models.keys())}")
        
        return models
    
    def train_models(self, X_train, y_train):
        """Entrenar todos los modelos."""
        
        logger.info("Iniciando entrenamiento de modelos...")
        logger.info(f"Datos de entrenamiento: {X_train.shape}")
        
        if not self.models:
            self.initialize_models()
        
        trained_models = {}
        
        for name, model in self.models.items():
            logger.info(f"\\nEntrenando {name}...")
            
            try:
                # Entrenar modelo
                model.fit(X_train, y_train)
                
                # Validación cruzada
                cv_scores = cross_val_score(
                    model, X_train, y_train, 
                    cv=5, scoring='roc_auc'
                )
                
                trained_models[name] = {
                    'model': model,
                    'cv_mean': cv_scores.mean(),
                    'cv_std': cv_scores.std()
                }
                
                logger.info(f"  ✅ CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
                
            except Exception as e:
                logger.error(f"  ❌ Error en {name}: {str(e)}")
                continue
        
        # Actualizar modelos entrenados
        self.models = {name: info['model'] for name, info in trained_models.items()}
        self.model_scores = {
            name: {'cv_mean': info['cv_mean'], 'cv_std': info['cv_std']}
            for name, info in trained_models.items()
        }
        
        logger.info(f"\\n✅ Entrenamiento completado: {len(self.models)} modelos")
        return self.models
    
    def evaluate_models(self, X_test, y_test):
        """Evaluar modelos en conjunto de prueba."""
        
        logger.info("\\nEvaluando modelos...")
        
        evaluation_results = {}
        
        for name, model in self.models.items():
            logger.info(f"\\nEvaluando {name}...")
            
            try:
                # Predicciones
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Métricas
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': classification_report(y_test, y_pred, output_dict=True)['1']['precision'],
                    'recall': classification_report(y_test, y_pred, output_dict=True)['1']['recall'],
                    'f1_score': f1_score(y_test, y_pred),
                    'roc_auc': roc_auc_score(y_test, y_pred_proba)
                }
                
                # Agregar métricas de CV
                if name in self.model_scores:
                    metrics.update(self.model_scores[name])
                
                evaluation_results[name] = metrics
                
                logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
                logger.info(f"  Precision: {metrics['precision']:.4f}")
                logger.info(f"  Recall: {metrics['recall']:.4f}")
                logger.info(f"  F1-Score: {metrics['f1_score']:.4f}")
                logger.info(f"  AUC-ROC: {metrics['roc_auc']:.4f}")
                
            except Exception as e:
                logger.error(f"  ❌ Error evaluando {name}: {str(e)}")
                continue
        
        # Seleccionar mejor modelo
        if evaluation_results:
            best_model_name = max(evaluation_results.keys(), 
                                key=lambda x: evaluation_results[x]['roc_auc'])
            self.best_model_name = best_model_name
            self.best_model = self.models[best_model_name]
            
            logger.info(f"\\n🏆 MEJOR MODELO: {best_model_name}")
            logger.info(f"   AUC-ROC: {evaluation_results[best_model_name]['roc_auc']:.4f}")
        
        return evaluation_results
    
    def create_visualizations(self, X_test, y_test, evaluation_results):
        """Crear visualizaciones de resultados."""
        
        logger.info("\\nCreando visualizaciones...")
        
        # Crear directorio
        plots_dir = Path("plots")
        plots_dir.mkdir(exist_ok=True)
        
        # 1. Comparación de modelos
        if evaluation_results:
            metrics_df = pd.DataFrame(evaluation_results).T
            
            plt.figure(figsize=(12, 8))
            metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1_score', 'roc_auc']
            
            x_pos = np.arange(len(evaluation_results))
            width = 0.15
            
            for i, metric in enumerate(metrics_to_plot):
                values = [evaluation_results[model][metric] for model in evaluation_results.keys()]
                plt.bar(x_pos + i*width, values, width, label=metric.replace('_', ' ').title())
            
            plt.xlabel('Modelos')
            plt.ylabel('Score')
            plt.title('Comparación de Métricas por Modelo')
            plt.xticks(x_pos + width*2, list(evaluation_results.keys()), rotation=45)
            plt.legend()
            plt.tight_layout()
            plt.savefig(plots_dir / 'real_model_comparison.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 2. Curvas ROC
        plt.figure(figsize=(10, 8))
        
        for name, model in self.models.items():
            try:
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
                auc = roc_auc_score(y_test, y_pred_proba)
                plt.plot(fpr, tpr, label=f'{name} (AUC = {auc:.3f})')
            except:
                continue
        
        plt.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        plt.xlabel('Tasa de Falsos Positivos')
        plt.ylabel('Tasa de Verdaderos Positivos')
        plt.title('Curvas ROC - Modelos con Datos Reales')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.savefig(plots_dir / 'real_roc_curves.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Matriz de confusión del mejor modelo
        if self.best_model:
            y_pred = self.best_model.predict(X_test)
            cm = confusion_matrix(y_test, y_pred)
            
            plt.figure(figsize=(8, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                       xticklabels=['No Default', 'Default'],
                       yticklabels=['No Default', 'Default'])
            plt.title(f'Matriz de Confusión - {self.best_model_name}')
            plt.ylabel('Valores Reales')
            plt.xlabel('Predicciones')
            plt.savefig(plots_dir / 'real_confusion_matrix.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        # 4. Importancia de características
        if self.best_model and hasattr(self.best_model, 'feature_importances_'):
            importance = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': self.feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False).head(10)
            
            plt.figure(figsize=(10, 6))
            sns.barplot(data=feature_importance_df, x='importance', y='feature')
            plt.title(f'Importancia de Características - {self.best_model_name}')
            plt.xlabel('Importancia')
            plt.tight_layout()
            plt.savefig(plots_dir / 'real_feature_importance.png', dpi=300, bbox_inches='tight')
            plt.close()
        
        logger.info(f"Visualizaciones guardadas en: {plots_dir}")
    
    def save_model(self, filename=None):
        """Guardar el mejor modelo."""
        
        if not self.best_model:
            logger.error("No hay modelo para guardar")
            return ""
        
        # Crear directorio
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        
        if filename is None:
            filename = f'real_{self.best_model_name}_model.pkl'
        
        filepath = MODELS_DIR / filename
        
        try:
            model_data = {
                'model': self.best_model,
                'model_name': self.best_model_name,
                'feature_names': self.feature_names,
                'label_encoders': self.label_encoders,
                'scaler': self.scaler,
                'model_scores': self.model_scores.get(self.best_model_name, {})
            }
            
            joblib.dump(model_data, filepath)
            logger.info(f"✅ Modelo guardado: {filepath}")
            return str(filepath)
            
        except Exception as e:
            logger.error(f"❌ Error guardando modelo: {str(e)}")
            return ""


def main():
    """Función principal."""
    
    logger.info("="*70)
    logger.info("ENTRENAMIENTO DE MODELOS CON DATOS REALES")
    logger.info("="*70)
    
    # Inicializar modelo
    credit_model = RealCreditRiskModel()
    
    # 1. Cargar datos reales
    df = credit_model.load_real_data()
    if df is None:
        return
    
    # 2. Preparar características
    X, y = credit_model.prepare_features(df)
    
    # 3. Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    logger.info(f"\\nDivisión de datos:")
    logger.info(f"  Entrenamiento: {X_train.shape[0]} muestras")
    logger.info(f"  Prueba: {X_test.shape[0]} muestras")
    
    # 4. Entrenar modelos
    credit_model.train_models(X_train, y_train)
    
    # 5. Evaluar modelos
    evaluation_results = credit_model.evaluate_models(X_test, y_test)
    
    # 6. Crear visualizaciones
    credit_model.create_visualizations(X_test, y_test, evaluation_results)
    
    # 7. Guardar mejor modelo
    model_path = credit_model.save_model()
    
    # 8. Resumen final
    logger.info("\\n" + "="*70)
    logger.info("RESUMEN FINAL")
    logger.info("="*70)
    
    if evaluation_results:
        logger.info(f"\\n📊 RESULTADOS DE MODELOS:")
        for model_name, metrics in evaluation_results.items():
            logger.info(f"\\n  {model_name.upper()}:")
            logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
            logger.info(f"    Precision: {metrics['precision']:.4f}")
            logger.info(f"    Recall:    {metrics['recall']:.4f}")
            logger.info(f"    F1-Score:  {metrics['f1_score']:.4f}")
            logger.info(f"    AUC-ROC:   {metrics['roc_auc']:.4f}")
        
        if credit_model.best_model_name:
            best_metrics = evaluation_results[credit_model.best_model_name]
            logger.info(f"\\n🏆 MEJOR MODELO: {credit_model.best_model_name.upper()}")
            logger.info(f"   AUC-ROC: {best_metrics['roc_auc']:.4f}")
            logger.info(f"   Accuracy: {best_metrics['accuracy']:.4f}")
    
    logger.info(f"\\n✅ Características utilizadas: {len(credit_model.feature_names)}")
    logger.info(f"✅ Registros procesados: {len(df):,}")
    logger.info(f"✅ Modelo guardado: {model_path}")
    
    logger.info("\\n🎉 ENTRENAMIENTO COMPLETADO!")
    logger.info("\\n🚀 SIGUIENTE PASO:")
    logger.info("   - Ejecutar API: python api/app.py")
    logger.info("   - Ejecutar Dashboard: streamlit run dashboard/app.py")


if __name__ == "__main__":
    main()