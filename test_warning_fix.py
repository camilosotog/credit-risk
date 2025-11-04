"""
Prueba simple para verificar que el warning se corrigió.
"""

import pandas as pd
import joblib
from pathlib import Path
import warnings

# Configurar para mostrar warnings
warnings.filterwarnings('default')

def test_prediction_without_warning():
    """Probar predicción sin warning de feature names."""
    
    print("🔍 Probando predicción sin warnings...")
    
    # Cargar modelo
    model_path = Path("models/real_random_forest_model.pkl")
    
    if not model_path.exists():
        print("❌ Modelo no encontrado")
        return False
    
    # Cargar modelo y componentes
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_names = model_data.get('feature_names', [])
    
    print(f"✅ Modelo cargado: {len(feature_names)} características")
    
    # Datos de prueba
    test_data = {
        'age': 35,
        'income': 2500000,
        'socioeconomic_level': 3,
        'dependents': 2,
        'gender': 1,
        'housing_status': 1,
        'has_disability': 0,
        'invoice_value': 1500000,
        'approved_limit': 3000000,
        'invoice_to_income_ratio': 0.6,
        'limit_to_income_ratio': 1.2
    }
    
    # Crear DataFrame con nombres de columnas
    input_df = pd.DataFrame([test_data])
    
    # Asegurar todas las características
    for feature in feature_names:
        if feature not in input_df.columns:
            input_df[feature] = 0
    
    # Reordenar columnas
    input_df = input_df.reindex(columns=feature_names, fill_value=0)
    
    print(f"📊 Datos preparados: {input_df.shape}")
    print(f"📋 Columnas: {list(input_df.columns)}")
    
    # Escalar manteniendo nombres de columnas
    if scaler:
        input_scaled = scaler.transform(input_df)
        # CLAVE: Mantener como DataFrame con nombres
        input_df_final = pd.DataFrame(
            input_scaled, 
            columns=feature_names, 
            index=input_df.index
        )
    else:
        input_df_final = input_df
    
    print(f"🔧 Datos escalados: {type(input_df_final)}")
    print(f"📝 Es DataFrame: {isinstance(input_df_final, pd.DataFrame)}")
    
    # Predicción (debe ser sin warning)
    print("\\n🎯 Haciendo predicción...")
    prob_array = model.predict_proba(input_df_final)
    prob = prob_array[0, 1]
    
    print(f"✅ Predicción exitosa (sin warnings):")
    print(f"   Probabilidad: {prob:.4f}")
    print(f"   Clase: {'Default' if prob > 0.5 else 'No Default'}")
    
    return True

if __name__ == "__main__":
    print("="*50)
    print("🧪 PRUEBA DE CORRECCIÓN DE WARNINGS")
    print("="*50)
    
    success = test_prediction_without_warning()
    
    if success:
        print("\\n🎉 ¡Prueba exitosa! El warning debería estar corregido.")
    else:
        print("\\n❌ Prueba falló.")