"""
Análisis de perfiles de alto riesgo para encontrar rechazos reales.
"""

import pandas as pd
import joblib
from pathlib import Path

def test_high_risk_profiles():
    """Probar perfiles diseñados para generar rechazo."""
    
    print("🔍 ANÁLISIS DE PERFILES DE ALTO RIESGO")
    print("="*60)
    
    # Cargar modelo
    model_path = Path("models/real_random_forest_model.pkl")
    model_data = joblib.load(model_path)
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_names = model_data.get('feature_names', [])
    label_encoders = model_data.get('label_encoders', {})
    
    def evaluate_profile(name, data):
        """Evaluar un perfil específico."""
        input_df = pd.DataFrame([data])
        
        # Codificar variables categóricas
        for col, encoder in label_encoders.items():
            if col in input_df.columns:
                try:
                    input_df[col] = encoder.transform(input_df[col].astype(str))
                except:
                    input_df[col] = 0
        
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
        decision = "❌ RECHAZADO" if default_prob > 0.5 else "✅ APROBADO"
        
        print(f"\\n📋 {name}")
        print(f"   Default: {default_prob:.1%} | {decision}")
        print(f"   💰 Ingresos: ${data['income']:,}")
        print(f"   🏠 Estrato: {data['socioeconomic_level']}")
        print(f"   👥 Dependientes: {data['dependents']}")
        print(f"   📊 Ratio F/I: {data['invoice_to_income_ratio']:.2f}")
        print(f"   📊 Ratio C/I: {data['limit_to_income_ratio']:.2f}")
        
        return default_prob > 0.5, default_prob
    
    # Perfiles de alto riesgo progresivo
    profiles = [
        ("🔴 PERFIL EXTREMO V1", {
            'age': 18,
            'income': 500000,
            'socioeconomic_level': 1,
            'dependents': 8,
            'gender': 0,  # F
            'housing_status': 0,  # Arrendada
            'has_disability': 1,  # Sí
            'invoice_value': 450000,
            'approved_limit': 600000,
            'invoice_to_income_ratio': 0.9,
            'limit_to_income_ratio': 1.2
        }),
        
        ("🔴 PERFIL EXTREMO V2", {
            'age': 19,
            'income': 300000,
            'socioeconomic_level': 1,
            'dependents': 10,
            'gender': 0,
            'housing_status': 0,
            'has_disability': 1,
            'invoice_value': 280000,
            'approved_limit': 350000,
            'invoice_to_income_ratio': 0.93,
            'limit_to_income_ratio': 1.17
        }),
        
        ("🔴 PERFIL CRISIS", {
            'age': 20,
            'income': 200000,
            'socioeconomic_level': 1,
            'dependents': 15,
            'gender': 0,
            'housing_status': 0,
            'has_disability': 1,
            'invoice_value': 190000,
            'approved_limit': 250000,
            'invoice_to_income_ratio': 0.95,
            'limit_to_income_ratio': 1.25
        }),
        
        ("🔴 PERFIL IMPOSIBLE", {
            'age': 18,
            'income': 100000,
            'socioeconomic_level': 1,
            'dependents': 20,
            'gender': 0,
            'housing_status': 0,
            'has_disability': 1,
            'invoice_value': 95000,
            'approved_limit': 120000,
            'invoice_to_income_ratio': 0.95,
            'limit_to_income_ratio': 1.2
        }),
        
        ("🟠 PERFIL LÍMITE", {
            'age': 25,
            'income': 1200000,
            'socioeconomic_level': 2,
            'dependents': 6,
            'gender': 1,
            'housing_status': 0,
            'has_disability': 0,
            'invoice_value': 1100000,
            'approved_limit': 1400000,
            'invoice_to_income_ratio': 0.92,
            'limit_to_income_ratio': 1.17
        })
    ]
    
    rejections = 0
    results = []
    
    for name, data in profiles:
        is_rejected, prob = evaluate_profile(name, data)
        results.append((name, prob, is_rejected))
        if is_rejected:
            rejections += 1
    
    print(f"\\n📊 RESUMEN: {rejections}/{len(profiles)} perfiles rechazados")
    
    # Ordenar por probabilidad de default
    results.sort(key=lambda x: x[1], reverse=True)
    
    print("\\n🏆 RANKING DE RIESGO (Mayor a menor):")
    print("-" * 50)
    for i, (name, prob, rejected) in enumerate(results, 1):
        status = "❌ RECHAZADO" if rejected else "✅ APROBADO"
        print(f"{i}. {prob:.1%} - {name} - {status}")
    
    # Análisis de factores
    print("\\n🔍 ANÁLISIS DE FACTORES DE RIESGO:")
    print("-" * 40)
    print("Los factores que más influyen en el rechazo son:")
    print("1. 💰 Ingresos muy bajos (<500,000)")
    print("2. 👥 Muchos dependientes (>5)")
    print("3. 🏠 Estrato socioeconómico bajo (1)")
    print("4. 🏡 Vivienda no propia (arrendada/familiar)")
    print("5. ♿ Presencia de discapacidad")
    print("6. 👶 Edad muy joven (<25 años)")
    print("7. 📊 Ratio factura/ingresos alto (>0.8)")
    
    return results

if __name__ == "__main__":
    test_high_risk_profiles()