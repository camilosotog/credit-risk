"""
Probar perfiles que deberían ser aprobados por el modelo.
"""

import pandas as pd
import joblib
from pathlib import Path

def test_approval_profiles():
    """Probar diferentes perfiles para aprobación."""
    
    print("🎯 PROBANDO PERFILES PARA APROBACIÓN")
    print("="*50)
    
    # Cargar modelo
    model_path = Path("models/real_random_forest_model.pkl")
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_names = model_data.get('feature_names', [])
    label_encoders = model_data.get('label_encoders', {})
    
    # Perfiles de prueba
    profiles = {
        "👨‍💼 Ejecutivo Senior": {
            'age': 42,
            'income': 5000000,
            'socioeconomic_level': 5,
            'dependents': 2,
            'gender': 1,  # Codificado
            'housing_status': 1,  # Codificado (Propia)
            'has_disability': 0,
            'invoice_value': 2000000,
            'approved_limit': 8000000,
            'invoice_to_income_ratio': 0.4,  # 2M/5M
            'limit_to_income_ratio': 1.6     # 8M/5M
        },
        
        "👩‍⚕️ Profesional Independiente": {
            'age': 38,
            'income': 3500000,
            'socioeconomic_level': 4,
            'dependents': 1,
            'gender': 0,  # Codificado (F)
            'housing_status': 1,  # Codificado (Propia)
            'has_disability': 0,
            'invoice_value': 1200000,
            'approved_limit': 5000000,
            'invoice_to_income_ratio': 0.34,  # 1.2M/3.5M
            'limit_to_income_ratio': 1.43     # 5M/3.5M
        },
        
        "🚀 Empresario Joven": {
            'age': 35,
            'income': 4500000,
            'socioeconomic_level': 5,
            'dependents': 0,
            'gender': 1,  # Codificado (M)
            'housing_status': 1,  # Codificado (Propia)
            'has_disability': 0,
            'invoice_value': 1800000,
            'approved_limit': 7000000,
            'invoice_to_income_ratio': 0.4,   # 1.8M/4.5M
            'limit_to_income_ratio': 1.56     # 7M/4.5M
        },
        
        "⚠️ Perfil de Riesgo (comparación)": {
            'age': 24,
            'income': 1200000,
            'socioeconomic_level': 2,
            'dependents': 4,
            'gender': 1,
            'housing_status': 0,  # Arrendada
            'has_disability': 0,
            'invoice_value': 1000000,
            'approved_limit': 1500000,
            'invoice_to_income_ratio': 0.83,  # 1M/1.2M
            'limit_to_income_ratio': 1.25     # 1.5M/1.2M
        }
    }
    
    results = []
    
    for profile_name, data in profiles.items():
        print(f"\\n🧪 Evaluando: {profile_name}")
        print("-" * 40)
        
        # Preparar datos
        input_df = pd.DataFrame([data])
        
        # Asegurar todas las características
        for feature in feature_names:
            if feature not in input_df.columns:
                input_df[feature] = 0
        
        input_df = input_df.reindex(columns=feature_names, fill_value=0)
        
        # Escalar
        if scaler:
            input_scaled = scaler.transform(input_df)
            input_df_final = pd.DataFrame(input_scaled, columns=feature_names)
        else:
            input_df_final = input_df
        
        # Predecir
        prob_array = model.predict_proba(input_df_final)
        default_prob = prob_array[0, 1]
        approval_prob = prob_array[0, 0]
        
        # Determinar decisión
        decision = "✅ APROBADO" if default_prob < 0.5 else "❌ RECHAZADO"
        risk_level = "BAJO" if default_prob < 0.3 else "MEDIO" if default_prob < 0.7 else "ALTO"
        
        print(f"  📊 Probabilidad de default: {default_prob:.1%}")
        print(f"  📈 Probabilidad de no default: {approval_prob:.1%}")
        print(f"  🎯 Decisión: {decision}")
        print(f"  ⚡ Nivel de riesgo: {risk_level}")
        
        # Mostrar factores clave
        print(f"  💰 Ingresos: ${data['income']:,}")
        print(f"  🏠 Estrato: {data['socioeconomic_level']}")
        print(f"  📊 Ratio factura/ingresos: {data['invoice_to_income_ratio']:.2f}")
        
        results.append({
            'perfil': profile_name,
            'default_prob': default_prob,
            'decision': decision,
            'risk_level': risk_level
        })
    
    # Resumen
    print("\\n" + "="*60)
    print("📋 RESUMEN DE RESULTADOS")
    print("="*60)
    
    for result in results:
        icon = "✅" if "APROBADO" in result['decision'] else "❌"
        print(f"{icon} {result['perfil']:<30} | {result['default_prob']:.1%} | {result['decision']}")
    
    return results

if __name__ == "__main__":
    test_approval_profiles()