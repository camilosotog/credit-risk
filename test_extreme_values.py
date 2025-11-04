"""
🧪 TEST DE VALORES EXTREMOS PARA ENCONTRAR RECHAZOS
==================================================
Este script prueba valores verdaderamente extremos para intentar
encontrar los límites del modelo y casos de rechazo.
"""

import pandas as pd
import joblib
import numpy as np

def load_model():
    """Cargar el modelo entrenado"""
    try:
        model_data = joblib.load('models/real_random_forest_model.pkl')
        return model_data['model'], model_data['scaler'], model_data['label_encoders'], model_data['feature_names']
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")
        return None, None, None, None

def create_extreme_profile(age, income, socioeconomic_level, dependents, 
                          gender, housing_status, has_disability, 
                          invoice_value, approved_limit):
    """Crear perfil de cliente con valores específicos"""
    
    profile = {
        'Edad': age,
        'Ingresos': income,
        'Estrato': socioeconomic_level,
        'Dependientes': dependents,
        'Genero': gender,
        'TipoVivienda': housing_status,
        'Discapacidad': has_disability,
        'ValorFactura': invoice_value,
        'CupoAprobado': approved_limit,
        'RatioFacturaIngresos': invoice_value / income if income > 0 else 0,
        'RatioCupoIngresos': approved_limit / income if income > 0 else 0
    }
    
    return profile

def evaluate_profile(profile, model, scaler, encoders, feature_names):
    """Evaluar un perfil específico"""
    
    df = pd.DataFrame([profile])
    
    # Aplicar encoders
    for col, encoder in encoders.items():
        if col in df.columns:
            try:
                df[col] = encoder.transform(df[col])
            except:
                df[col] = 0
    
    # Asegurar todas las características
    for feature in feature_names:
        if feature not in df.columns:
            df[feature] = 0
    
    # Reordenar columnas
    df = df.reindex(columns=feature_names, fill_value=0)
    
    # Escalar
    if scaler:
        df_scaled = scaler.transform(df)
        df_final = pd.DataFrame(df_scaled, columns=feature_names)
    else:
        df_final = df
    
    # Predecir
    prob_array = model.predict_proba(df_final)
    default_prob = prob_array[0, 1]
    approval_prob = prob_array[0, 0]
    
    return default_prob, approval_prob

def main():
    print("🧪 TESTING DE VALORES EXTREMOS")
    print("=" * 60)
    
    # Cargar modelo
    model, scaler, encoders, feature_names = load_model()
    
    if model is None:
        return
    
    # Casos extremos para probar
    extreme_cases = [
        {
            'name': '💀 CASO IMPOSIBLE 1',
            'description': 'Menor sueldo mínimo + máximos dependientes',
            'params': (18, 50000, 1, 30, 'Femenino', 'Arrendada', 'Sí', 45000, 100000)
        },
        {
            'name': '💀 CASO IMPOSIBLE 2', 
            'description': 'Ingresos ultra bajos + ratio extremo',
            'params': (18, 30000, 1, 25, 'Femenino', 'Arrendada', 'Sí', 25000, 50000)
        },
        {
            'name': '💀 CASO IMPOSIBLE 3',
            'description': 'Sobreendeudamiento extremo',
            'params': (18, 100000, 1, 20, 'Femenino', 'Familiar', 'Sí', 95000, 200000)
        },
        {
            'name': '💀 CASO LÍMITE 1',
            'description': 'Factura > Ingresos',
            'params': (18, 200000, 1, 15, 'Femenino', 'Arrendada', 'Sí', 250000, 300000)
        },
        {
            'name': '💀 CASO LÍMITE 2',
            'description': 'Cupo 10x ingresos',
            'params': (18, 150000, 1, 20, 'Femenino', 'Arrendada', 'Sí', 140000, 1500000)
        },
        {
            'name': '🔥 CASO EXPERIMENTAL 1',
            'description': 'Ingresos mínimos teóricos',
            'params': (18, 10000, 1, 30, 'Femenino', 'Arrendada', 'Sí', 9500, 20000)
        },
        {
            'name': '🔥 CASO EXPERIMENTAL 2', 
            'description': 'Edad mínima + todo negativo',
            'params': (18, 20000, 1, 30, 'Femenino', 'Arrendada', 'Sí', 19000, 25000)
        },
        {
            'name': '⚡ CASO NUCLEAR',
            'description': 'El peor caso posible matemáticamente',
            'params': (18, 1000, 1, 30, 'Femenino', 'Arrendada', 'Sí', 950, 2000)
        }
    ]
    
    results = []
    
    for case in extreme_cases:
        name = case['name']
        desc = case['description']
        params = case['params']
        
        # Crear perfil
        profile = create_extreme_profile(*params)
        
        # Evaluar
        default_prob, approval_prob = evaluate_profile(profile, model, scaler, encoders, feature_names)
        
        # Calcular métricas
        default_pct = default_prob * 100
        approval_pct = approval_prob * 100
        
        # Determinar resultado
        if approval_prob > 0.5:
            result = "✅ APROBADO"
            emoji = "🟢"
        else:
            result = "❌ RECHAZADO"
            emoji = "🔴"
        
        print(f"\n📋 {name}")
        print(f"   {desc}")
        print(f"   Riesgo: {default_pct:.1f}% | {emoji} {result}")
        print(f"   💰 Ingresos: ${params[1]:,}")
        print(f"   👥 Dependientes: {params[3]}")
        print(f"   📊 Ratio F/I: {profile['RatioFacturaIngresos']:.2f}")
        print(f"   📊 Ratio C/I: {profile['RatioCupoIngresos']:.2f}")
        
        results.append({
            'case': name,
            'default_risk': default_pct,
            'approved': approval_prob > 0.5,
            'income': params[1],
            'dependents': params[3]
        })
    
    # Resumen final
    approved_count = sum(1 for r in results if r['approved'])
    rejected_count = len(results) - approved_count
    max_risk = max(r['default_risk'] for r in results)
    min_risk = min(r['default_risk'] for r in results)
    
    print(f"\n📊 RESUMEN FINAL:")
    print(f"--------------------------------------------------")
    print(f"✅ Aprobados: {approved_count}/{len(results)}")
    print(f"❌ Rechazados: {rejected_count}/{len(results)}")
    print(f"📈 Riesgo máximo encontrado: {max_risk:.1f}%")
    print(f"📉 Riesgo mínimo encontrado: {min_risk:.1f}%")
    
    if rejected_count > 0:
        print(f"\n🎯 ¡ENCONTRAMOS RECHAZOS!")
        rejected_cases = [r for r in results if not r['approved']]
        for case in rejected_cases:
            print(f"   🔴 {case['case']}: {case['default_risk']:.1f}% riesgo")
    else:
        print(f"\n🤖 EL MODELO SIGUE SIENDO MUY PERMISIVO")
        print(f"   Incluso con casos extremos, no rechaza ningún perfil.")
        print(f"   Esto refleja una estrategia de máxima inclusión financiera.")
    
    print(f"\n💡 CONCLUSIÓN PARA LA TESIS:")
    print(f"   El modelo está optimizado para inclusión, no para rechazo estricto.")
    print(f"   Esto es típico en Fintechs que priorizan cobertura sobre selección.")

if __name__ == "__main__":
    main()