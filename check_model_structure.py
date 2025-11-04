"""
Script para verificar la estructura del modelo guardado
"""
import joblib

# Cargar el modelo para ver su estructura
try:
    model_data = joblib.load('models/real_random_forest_model.pkl')
    print("🔍 ESTRUCTURA DEL MODELO:")
    print("=" * 50)
    
    if isinstance(model_data, dict):
        print("📦 El modelo es un diccionario con claves:")
        for key in model_data.keys():
            print(f"   - {key}: {type(model_data[key])}")
    else:
        print(f"📦 El modelo es de tipo: {type(model_data)}")
        print(f"📦 Contenido: {model_data}")
        
except Exception as e:
    print(f"❌ Error: {e}")