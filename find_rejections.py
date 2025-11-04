"""
Encontrar el perfil exacto que genera rechazo vs aprobación.
"""

import pandas as pd
import joblib
from pathlib import Path

def find_rejection_vs_approval():
    """Encontrar qué perfil genera rechazo."""
    
    print("🔍 BUSCANDO PERFILES DE RECHAZO VS APROBACIÓN")
    print("="*60)
    
    # Cargar modelo
    model_path = Path("models/real_random_forest_model.pkl")
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_names = model_data.get('feature_names', [])
    
    def test_profile(name, data):
        """Probar un perfil específico."""
        input_df = pd.DataFrame([data])
        
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        
        if scaler:
            input_scaled = scaler.transform(input_df)
            input_df_final = pd.DataFrame(input_scaled, columns=feature_names)
        else:
            input_df_final = input_df
        
        prob_array = model.predict_proba(input_df_final)
        default_prob = prob_array[0, 1]
        decision = "✅ APROBADO" if default_prob < 0.5 else "❌ RECHAZADO"
        
        print(f"\\n{name}")
        print(f"  📊 Default: {default_prob:.1%} | {decision}")
        print(f"  💰 Ingresos: ${data['income']:,}")
        print(f"  🏠 Estrato: {data['socioeconomic_level']}")
        print(f"  👥 Dependientes: {data['dependents']}")
        print(f"  🏡 Vivienda: {'Propia' if data['housing_status'] == 1 else 'Arrendada'}")
        
        return default_prob > 0.5
    
    # Perfiles extremos
    profiles = [
        ("💚 PERFIL PREMIUM", {
            'age': 40,
            'income': 8000000,
            'socioeconomic_level': 6,
            'dependents': 0,
            'gender': 1,
            'housing_status': 1,  # Propia
            'has_disability': 0,
            'invoice_value': 1000000,
            'approved_limit': 10000000,
            'invoice_to_income_ratio': 0.125,  # Muy bajo
            'limit_to_income_ratio': 1.25
        }),
        
        ("🟡 PERFIL BUENO", {
            'age': 35,
            'income': 3000000,
            'socioeconomic_level': 4,
            'dependents': 1,
            'gender': 1,
            'housing_status': 1,
            'has_disability': 0,
            'invoice_value': 1500000,
            'approved_limit': 4000000,
            'invoice_to_income_ratio': 0.5,
            'limit_to_income_ratio': 1.33
        }),
        
        ("🟠 PERFIL RIESGOSO", {
            'age': 22,
            'income': 800000,
            'socioeconomic_level': 1,
            'dependents': 5,
            'gender': 0,
            'housing_status': 0,  # Arrendada
            'has_disability': 1,
            'invoice_value': 700000,
            'approved_limit': 1000000,
            'invoice_to_income_ratio': 0.875,  # Muy alto
            'limit_to_income_ratio': 1.25
        }),
        
        ("🔴 PERFIL EXTREMO", {
            'age': 19,
            'income': 500000,
            'socioeconomic_level': 1,
            'dependents': 6,
            'gender': 0,
            'housing_status': 0,
            'has_disability': 1,
            'invoice_value': 450000,
            'approved_limit': 600000,
            'invoice_to_income_ratio': 0.9,
            'limit_to_income_ratio': 1.2
        }),
        
        ("💀 PERFIL IMPOSIBLE", {
            'age': 18,
            'income': 200000,
            'socioeconomic_level': 1,
            'dependents': 10,
            'gender': 0,
            'housing_status': 0,
            'has_disability': 1,
            'invoice_value': 180000,
            'approved_limit': 250000,
            'invoice_to_income_ratio': 0.9,
            'limit_to_income_ratio': 1.25
        })
    ]
    
    rejections = 0
    for name, data in profiles:
        is_rejected = test_profile(name, data)
        if is_rejected:
            rejections += 1
    
    print(f"\\n📊 RESULTADO: {rejections}/{len(profiles)} perfiles rechazados")
    
    if rejections == 0:
        print("\\n🤔 EL MODELO ES MUY PERMISIVO")
        print("Todos los perfiles fueron aprobados. Esto puede indicar:")
        print("- El modelo está sobreajustado hacia aprobación")
        print("- Los datos de entrenamiento tenían pocos rechazos")
        print("- El umbral de decisión podría necesitar ajuste")

if __name__ == "__main__":
    find_rejection_vs_approval()