"""
Script de prueba para verificar el funcionamiento del sistema completo
con el modelo real entrenado.
"""

import requests
import json
import pandas as pd
import joblib
from pathlib import Path
import logging
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))

from src.utils.model_utils import load_model_safely, predict_with_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_api_connection():
    """Probar conexión con la API."""
    
    logger.info("🔍 Probando conexión con API...")
    
    try:
        response = requests.get("http://localhost:8000/health", timeout=5)
        if response.status_code == 200:
            logger.info("✅ API funcionando correctamente")
            return True
        else:
            logger.error(f"❌ API respondió con código: {response.status_code}")
            return False
    except Exception as e:
        logger.error(f"❌ Error conectando con API: {str(e)}")
        logger.info("💡 Asegúrate de que la API esté ejecutándose: python api/app.py")
        return False


def test_model_prediction():
    """Probar predicciones del modelo."""
    
    logger.info("🔍 Probando modelo local...")
    
    # Cargar modelo
    model_path = Path("models/real_random_forest_model.pkl")
    
    if not model_path.exists():
        logger.error("❌ Modelo no encontrado")
        return False
    
    try:
        # Usar utilidad segura para cargar modelo
        predictor = load_model_safely(model_path)
        
        logger.info(f"✅ Modelo cargado correctamente")
        logger.info(f"✅ Características: {len(predictor.feature_names)}")
        
        # Crear datos de prueba
        test_data = {
            'age': 35,
            'income': 2500000,
            'socioeconomic_level': 3,
            'dependents': 2,
            'gender': 1,  # Codificado
            'housing_status': 1,  # Codificado  
            'has_disability': 0,
            'invoice_value': 1500000,
            'approved_limit': 3000000,
            'invoice_to_income_ratio': 0.6,
            'limit_to_income_ratio': 1.2
        }
        
        # Hacer predicción segura (sin warnings)
        prob_array = predictor.predict_proba(test_data)
        prob = prob_array[0, 1]
        pred_class = int(prob > 0.5)
        
        logger.info(f"✅ Predicción exitosa:")
        logger.info(f"   Probabilidad de default: {prob:.4f}")
        logger.info(f"   Clase predicha: {'Default' if pred_class else 'No Default'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error en predicción: {str(e)}")
        return False


def test_api_prediction():
    """Probar predicción via API."""
    
    logger.info("🔍 Probando predicción por API...")
    
    # Datos de prueba
    test_customer = {
        "age": 28,
        "income": 3000000,
        "socioeconomic_level": 4,
        "dependents": 1,
        "gender": "F",
        "housing_status": "Propia",
        "has_disability": "No",
        "invoice_value": 2000000,
        "approved_limit": 4000000
    }
    
    try:
        response = requests.post(
            "http://localhost:8000/predict",
            json=test_customer,
            timeout=10
        )
        
        if response.status_code == 200:
            result = response.json()
            pred = result.get('prediction', {})
            
            logger.info("✅ API prediction exitosa:")
            logger.info(f"   Probabilidad: {pred.get('default_probability', 'N/A'):.4f}")
            logger.info(f"   Riesgo: {pred.get('risk_level', 'N/A')}")
            logger.info(f"   Recomendación: {pred.get('recommendation', 'N/A')}")
            
            return True
        else:
            logger.error(f"❌ API error: {response.status_code}")
            logger.error(f"   Respuesta: {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error en API prediction: {str(e)}")
        return False


def test_real_data_sample():
    """Probar con muestra de datos reales."""
    
    logger.info("🔍 Probando con datos reales...")
    
    # Cargar datos procesados
    data_path = Path("data/processed/real_credit_data_processed.csv")
    
    if not data_path.exists():
        logger.error("❌ Datos procesados no encontrados")
        return False
    
    try:
        df = pd.read_csv(data_path)
        logger.info(f"✅ Datos cargados: {len(df)} registros")
        
        # Tomar muestra aleatoria
        sample = df.sample(n=5, random_state=42)
        
        logger.info("\\n📊 Muestra de datos reales:")
        for idx, row in sample.iterrows():
            logger.info(f"\\n  Registro {idx}:")
            logger.info(f"    Edad: {row.get('age', 'N/A')}")
            logger.info(f"    Ingresos: ${row.get('income', 'N/A'):,.0f}")
            logger.info(f"    Valor factura: ${row.get('invoice_value', 'N/A'):,.0f}")
            logger.info(f"    Default real: {'Sí' if row.get('default', 0) else 'No'}")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Error con datos reales: {str(e)}")
        return False


def main():
    """Función principal de pruebas."""
    
    logger.info("="*70)
    logger.info("🧪 PRUEBAS DEL SISTEMA DE RIESGO CREDITICIO")
    logger.info("="*70)
    
    results = []
    
    # Prueba 1: Modelo local
    logger.info("\\n" + "="*50)
    logger.info("PRUEBA 1: MODELO LOCAL")
    logger.info("="*50)
    results.append(("Modelo Local", test_model_prediction()))
    
    # Prueba 2: Conexión API
    logger.info("\\n" + "="*50)
    logger.info("PRUEBA 2: CONEXIÓN API")
    logger.info("="*50)
    results.append(("Conexión API", test_api_connection()))
    
    # Prueba 3: Predicción API
    logger.info("\\n" + "="*50)
    logger.info("PRUEBA 3: PREDICCIÓN API")
    logger.info("="*50)
    results.append(("Predicción API", test_api_prediction()))
    
    # Prueba 4: Datos reales
    logger.info("\\n" + "="*50)
    logger.info("PRUEBA 4: DATOS REALES")
    logger.info("="*50)
    results.append(("Datos Reales", test_real_data_sample()))
    
    # Resumen
    logger.info("\\n" + "="*70)
    logger.info("📋 RESUMEN DE PRUEBAS")
    logger.info("="*70)
    
    passed = 0
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {test_name:.<30} {status}")
        if result:
            passed += 1
    
    logger.info(f"\\n🎯 RESULTADO: {passed}/{len(results)} pruebas exitosas")
    
    if passed == len(results):
        logger.info("\\n🎉 ¡TODAS LAS PRUEBAS EXITOSAS!")
        logger.info("\\n🚀 EL SISTEMA ESTÁ LISTO PARA USAR:")
        logger.info("   📊 Dashboard: http://localhost:8502")
        logger.info("   🔗 API: http://localhost:8000")
        logger.info("   📈 Modelo: Random Forest (99.35% AUC-ROC)")
    else:
        logger.info("\\n⚠️ Algunas pruebas fallaron. Revisa los logs arriba.")


if __name__ == "__main__":
    main()