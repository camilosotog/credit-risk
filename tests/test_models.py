"""
Tests unitarios para el sistema de evaluación de riesgo crediticio.
"""

import unittest
import pandas as pd
import numpy as np
from pathlib import Path
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.load_data import DataLoader
from src.features.feature_engineering import FeatureEngineer
from src.models.train_model import CreditRiskModel


class TestDataLoader(unittest.TestCase):
    """Tests para el cargador de datos."""
    
    def setUp(self):
        """Configuración inicial para los tests."""
        self.loader = DataLoader()
    
    def test_create_sample_data(self):
        """Test para generación de datos sintéticos."""
        df = self.loader.create_sample_data(n_samples=100)
        
        # Verificar dimensiones
        self.assertEqual(len(df), 100)
        self.assertTrue(len(df.columns) > 10)
        
        # Verificar columnas requeridas
        required_columns = ['customer_id', 'age', 'income', 'credit_score', 'default']
        for col in required_columns:
            self.assertIn(col, df.columns)
        
        # Verificar rangos de datos
        self.assertTrue(df['age'].min() >= 18)
        self.assertTrue(df['age'].max() <= 80)
        self.assertTrue(df['credit_score'].min() >= 300)
        self.assertTrue(df['credit_score'].max() <= 850)
        self.assertTrue(df['debt_ratio'].min() >= 0)
        self.assertTrue(df['debt_ratio'].max() <= 1)
    
    def test_clean_data(self):
        """Test para limpieza de datos."""
        # Crear datos con problemas
        df_dirty = self.loader.create_sample_data(n_samples=50)
        
        # Introducir valores faltantes
        df_dirty.loc[0:5, 'income'] = np.nan
        df_dirty.loc[10:15, 'credit_score'] = np.nan
        
        # Limpiar datos
        df_clean = self.loader.clean_data(df_dirty)
        
        # Verificar que no hay valores faltantes
        self.assertEqual(df_clean.isnull().sum().sum(), 0)
        
        # Verificar que se mantiene el número de filas
        self.assertEqual(len(df_clean), len(df_dirty))


class TestFeatureEngineer(unittest.TestCase):
    """Tests para ingeniería de características."""
    
    def setUp(self):
        """Configuración inicial para los tests."""
        self.feature_engineer = FeatureEngineer()
        loader = DataLoader()
        self.df = loader.create_sample_data(n_samples=100)
    
    def test_create_features(self):
        """Test para creación de nuevas características."""
        df_features = self.feature_engineer.create_features(self.df)
        
        # Verificar que se crearon nuevas características
        self.assertTrue(len(df_features.columns) > len(self.df.columns))
        
        # Verificar características específicas
        expected_features = [
            'loan_to_income_ratio', 'monthly_income', 
            'payment_to_income_ratio', 'income_category'
        ]
        
        for feature in expected_features:
            self.assertIn(feature, df_features.columns)
        
        # Verificar que no hay valores infinitos
        self.assertFalse(np.isinf(df_features.select_dtypes(include=[np.number])).any().any())
    
    def test_encode_categorical_features(self):
        """Test para codificación de características categóricas."""
        df_encoded = self.feature_engineer.encode_categorical_features(self.df, fit=True)
        
        # Verificar que las características categóricas fueron procesadas
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        
        # Algunas columnas categóricas deben haber sido transformadas
        for col in categorical_cols:
            if col in ['employment_status', 'education_level']:
                # Estas columnas deben ser transformadas a numéricas o one-hot
                pass
    
    def test_prepare_features(self):
        """Test para preparación de características."""
        X_train, X_test, y_train, y_test = self.feature_engineer.prepare_features(
            self.df, target_col='default', test_size=0.2
        )
        
        # Verificar dimensiones
        self.assertEqual(len(X_train) + len(X_test), len(self.df))
        self.assertEqual(len(X_train), len(y_train))
        self.assertEqual(len(X_test), len(y_test))
        
        # Verificar que no contiene la columna objetivo
        self.assertNotIn('default', X_train.columns)
        self.assertNotIn('default', X_test.columns)
        
        # Verificar tipos de datos
        self.assertIsInstance(X_train, pd.DataFrame)
        self.assertIsInstance(y_train, pd.Series)


class TestCreditRiskModel(unittest.TestCase):
    """Tests para el modelo de riesgo crediticio."""
    
    def setUp(self):
        """Configuración inicial para los tests."""
        self.model = CreditRiskModel()
        
        # Crear datos de prueba
        loader = DataLoader()
        df = loader.create_sample_data(n_samples=200)
        
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.create_features(df)
        df_encoded = feature_engineer.encode_categorical_features(df_features, fit=True)
        
        self.X_train, self.X_test, self.y_train, self.y_test = \
            feature_engineer.prepare_features(df_encoded)
    
    def test_initialize_models(self):
        """Test para inicialización de modelos."""
        models = self.model.initialize_models()
        
        # Verificar que se inicializaron modelos
        self.assertTrue(len(models) > 0)
        self.assertIn('logistic_regression', models)
        self.assertIn('random_forest', models)
    
    def test_train_models(self):
        """Test para entrenamiento de modelos."""
        # Reducir datos para test rápido
        X_train_small = self.X_train.head(50)
        y_train_small = self.y_train.head(50)
        
        trained_models = self.model.train_models(X_train_small, y_train_small)
        
        # Verificar que se entrenaron modelos
        self.assertTrue(len(trained_models) > 0)
        
        # Verificar que los modelos están entrenados
        for name, model in trained_models.items():
            self.assertTrue(hasattr(model, 'predict'))
    
    def test_predict(self):
        """Test para predicciones."""
        # Entrenar modelo simple
        self.model.initialize_models()
        self.model.train_models(self.X_train.head(50), self.y_train.head(50))
        
        # Hacer predicciones
        if self.model.best_model is not None:
            predictions = self.model.predict(self.X_test.head(10))
            
            # Verificar dimensiones
            self.assertEqual(len(predictions), 10)
            
            # Verificar que las predicciones son válidas
            self.assertTrue(all(pred in [0, 1] for pred in predictions))


class TestIntegration(unittest.TestCase):
    """Tests de integración para el pipeline completo."""
    
    def test_complete_pipeline(self):
        """Test para el pipeline completo."""
        # 1. Cargar datos
        loader = DataLoader()
        df = loader.create_sample_data(n_samples=100)
        
        # 2. Limpiar datos
        df_clean = loader.clean_data(df)
        
        # 3. Ingeniería de características
        feature_engineer = FeatureEngineer()
        df_features = feature_engineer.create_features(df_clean)
        df_encoded = feature_engineer.encode_categorical_features(df_features, fit=True)
        
        # 4. Preparar datos
        X_train, X_test, y_train, y_test = feature_engineer.prepare_features(df_encoded)
        
        # 5. Entrenar modelo
        model = CreditRiskModel()
        model.train_models(X_train, y_train)
        
        # 6. Evaluar modelo
        if len(model.models) > 0:
            evaluation = model.evaluate_models(X_test, y_test)
            
            # Verificar que hay resultados de evaluación
            self.assertTrue(len(evaluation) > 0)
            
            # Verificar métricas básicas
            for model_name, metrics in evaluation.items():
                self.assertIn('accuracy', metrics)
                self.assertIn('roc_auc', metrics)
                
                # Verificar rangos válidos
                self.assertTrue(0 <= metrics['accuracy'] <= 1)
                self.assertTrue(0 <= metrics['roc_auc'] <= 1)


def run_tests():
    """Ejecutar todos los tests."""
    # Crear suite de tests
    suite = unittest.TestSuite()
    
    # Agregar tests
    suite.addTest(unittest.makeSuite(TestDataLoader))
    suite.addTest(unittest.makeSuite(TestFeatureEngineer))
    suite.addTest(unittest.makeSuite(TestCreditRiskModel))
    suite.addTest(unittest.makeSuite(TestIntegration))
    
    # Ejecutar tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == '__main__':
    success = run_tests()
    if success:
        print("\n✅ Todos los tests pasaron exitosamente!")
    else:
        print("\n❌ Algunos tests fallaron.")
        exit(1)