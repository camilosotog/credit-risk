"""
Prueba del modelo SIN la variable Cupo Aprobado
"""

import pandas as pd
import joblib
from pathlib import Path
import numpy as np

print("=" * 70)
print("PRUEBA: ¿QUÉ PASA SI DESHABILITAMOS CUPO APROBADO?")
print("=" * 70)

# Cargar modelo
model_path = Path("models/real_xgboost_model.pkl")
if model_path.exists():
    model_data = joblib.load(model_path)
    print("\n✅ Modelo XGBoost cargado")
    
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_names = model_data.get('feature_names', [])
    label_encoders = model_data.get('label_encoders', {})
    
    # Caso de prueba: Perfil de BAJO RIESGO
    print("\n" + "=" * 70)
    print("📋 CASO DE PRUEBA: Perfil de BAJO RIESGO")
    print("=" * 70)
    
    test_case = {
        'Edad': 35,
        'Ingresos': 3500000,
        'Estrato': 4,
        'Dependientes': 1,
        'Genero': 1,
        'TipoVivienda': 1,
        'Discapacidad': 2,
        'ValorFactura': 200000,
        'CupoAprobado': 2860000  # Con valor real
    }
    
    print(f"\nEdad: {test_case['Edad']} años")
    print(f"Ingresos: ${test_case['Ingresos']:,}")
    print(f"Estrato: {test_case['Estrato']}")
    print(f"Valor Factura: ${test_case['ValorFactura']:,}")
    print(f"Cupo Aprobado: ${test_case['CupoAprobado']:,}")
    
    # Mapear columnas
    column_mapping = {
        'Edad': 'age',
        'Ingresos': 'income',
        'Genero': 'gender',
        'Estrato': 'socioeconomic_level',
        'Dependientes': 'dependents',
        'TipoVivienda': 'housing_status',
        'Discapacidad': 'has_disability',
        'ValorFactura': 'invoice_value',
        'CupoAprobado': 'approved_limit'
    }
    
    # Función para hacer predicción
    def predict(data_dict, use_approved_limit=True):
        df = pd.DataFrame([data_dict])
        df = df.rename(columns=column_mapping)
        
        # Crear características adicionales
        df['invoice_to_income_ratio'] = df['invoice_value'] / (df['income'] + 1)
        df['limit_to_income_ratio'] = df['approved_limit'] / (df['income'] + 1)
        
        # Si no usamos approved_limit, poner valor neutral
        if not use_approved_limit:
            df['approved_limit'] = 1200000  # Valor neutro (mediana del dataset)
            df['limit_to_income_ratio'] = df['approved_limit'] / (df['income'] + 1)
        
        # Codificar variables categóricas
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except:
                    df[col] = 0
        
        # Asegurar todas las características
        for feature in feature_names:
            if feature not in df.columns:
                df[feature] = 0
        
        df = df.reindex(columns=feature_names, fill_value=0)
        
        # Escalar
        if scaler:
            df_scaled = scaler.transform(df)
            df_final = pd.DataFrame(df_scaled, columns=feature_names)
        else:
            df_final = df
        
        # Predecir
        prob_array = model.predict_proba(df_final)
        return prob_array[0, 1] * 100  # Probabilidad de rechazo
    
    # Predicción CON Cupo Aprobado
    print("\n" + "-" * 70)
    print("🟢 PREDICCIÓN CON CUPO APROBADO HABILITADO:")
    print("-" * 70)
    prob_con = predict(test_case, use_approved_limit=True)
    print(f"Probabilidad de RECHAZO: {prob_con:.2f}%")
    print(f"Probabilidad de APROBACIÓN: {100-prob_con:.2f}%")
    if prob_con < 50:
        print("✅ DECISIÓN: APROBADO")
    else:
        print("❌ DECISIÓN: RECHAZADO")
    
    # Predicción SIN Cupo Aprobado
    print("\n" + "-" * 70)
    print("🔴 PREDICCIÓN SIN CUPO APROBADO (deshabilitado):")
    print("-" * 70)
    prob_sin = predict(test_case, use_approved_limit=False)
    print(f"Probabilidad de RECHAZO: {prob_sin:.2f}%")
    print(f"Probabilidad de APROBACIÓN: {100-prob_sin:.2f}%")
    if prob_sin < 50:
        print("✅ DECISIÓN: APROBADO")
    else:
        print("❌ DECISIÓN: RECHAZADO")
    
    # Análisis del cambio
    print("\n" + "=" * 70)
    print("📊 ANÁLISIS DEL IMPACTO")
    print("=" * 70)
    
    diferencia = abs(prob_sin - prob_con)
    print(f"\n🔍 Diferencia en probabilidad de rechazo: {diferencia:.2f} puntos porcentuales")
    
    if diferencia < 5:
        print("\n✅ IMPACTO BAJO: Las otras variables compensan razonablemente")
    elif diferencia < 20:
        print("\n⚠️ IMPACTO MODERADO: Hay cambio notable pero el modelo sigue funcionando")
    else:
        print("\n❌ IMPACTO ALTO: La predicción cambia significativamente")
    
    print("\n💡 INTERPRETACIÓN:")
    print(f"   - CON Cupo Aprobado: El modelo usa la variable más importante (91.73%)")
    print(f"   - SIN Cupo Aprobado: El modelo debe basarse en las otras variables (8.27%)")
    print(f"   - Esto puede hacer el modelo menos confiable y más incierto")
    
    # Prueba con más casos
    print("\n" + "=" * 70)
    print("🧪 PRUEBAS ADICIONALES")
    print("=" * 70)
    
    test_cases = [
        {
            'nombre': 'Perfil EXCELENTE',
            'data': {
                'Edad': 45,
                'Ingresos': 8000000,
                'Estrato': 6,
                'Dependientes': 2,
                'Genero': 1,
                'TipoVivienda': 1,
                'Discapacidad': 2,
                'ValorFactura': 150000,
                'CupoAprobado': 10000000
            }
        },
        {
            'nombre': 'Perfil MEDIO',
            'data': {
                'Edad': 30,
                'Ingresos': 2500000,
                'Estrato': 3,
                'Dependientes': 1,
                'Genero': 2,
                'TipoVivienda': 2,
                'Discapacidad': 2,
                'ValorFactura': 300000,
                'CupoAprobado': 2000000
            }
        },
        {
            'nombre': 'Perfil RIESGOSO',
            'data': {
                'Edad': 20,
                'Ingresos': 1200000,
                'Estrato': 1,
                'Dependientes': 0,
                'Genero': 2,
                'TipoVivienda': 2,
                'Discapacidad': 2,
                'ValorFactura': 500000,
                'CupoAprobado': 0
            }
        }
    ]
    
    for caso in test_cases:
        print(f"\n📋 {caso['nombre']}")
        print(f"   Ingresos: ${caso['data']['Ingresos']:,}, Estrato: {caso['data']['Estrato']}")
        
        prob_con = predict(caso['data'], use_approved_limit=True)
        prob_sin = predict(caso['data'], use_approved_limit=False)
        
        print(f"   CON Cupo:  Rechazo {prob_con:5.2f}% → {'✅ APROBADO' if prob_con < 50 else '❌ RECHAZADO'}")
        print(f"   SIN Cupo:  Rechazo {prob_sin:5.2f}% → {'✅ APROBADO' if prob_sin < 50 else '❌ RECHAZADO'}")
        print(f"   Diferencia: {abs(prob_sin - prob_con):.2f} puntos")
    
    print("\n" + "=" * 70)
    print("🎯 CONCLUSIÓN FINAL")
    print("=" * 70)
    print("""
Al deshabilitar el Cupo Aprobado:
    
1. El modelo pierde su variable MÁS IMPORTANTE (91.73% de influencia)
2. Debe basarse en variables con mucho menor peso combinado (8.27%)
3. Las predicciones se vuelven menos confiables y más inciertas
4. Puede cambiar radicalmente las decisiones en algunos casos

💡 RECOMENDACIÓN:
   Si deseas un modelo que NO dependa del Cupo Aprobado, deberías
   RE-ENTRENAR el modelo EXCLUYENDO esta variable desde el inicio,
   para que aprenda a dar más peso a las otras características.
    """)
    
else:
    print("\n❌ Modelo no encontrado")

print("=" * 70)
