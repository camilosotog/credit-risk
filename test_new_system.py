"""
Script de prueba del sistema completo con el nuevo dataset DataCreditos.csv
"""

import pandas as pd
import joblib
from pathlib import Path
import numpy as np

print("=" * 70)
print("PRUEBA DEL SISTEMA DE RIESGO CREDITICIO")
print("=" * 70)

# 1. Verificar dataset procesado
print("\n1️⃣ Verificando dataset procesado...")
data_path = Path("data/processed/real_credit_data_processed.csv")
if data_path.exists():
    df = pd.read_csv(data_path)
    print(f"   ✅ Dataset cargado: {len(df):,} registros")
    print(f"   ✅ Columnas: {len(df.columns)}")
    
    # Verificar variable objetivo
    if 'default' in df.columns:
        default_counts = df['default'].value_counts()
        print(f"\n   📊 Distribución de variable objetivo:")
        print(f"      No Default (0): {default_counts.get(0, 0):,} casos ({default_counts.get(0, 0)/len(df)*100:.1f}%)")
        print(f"      Default (1):    {default_counts.get(1, 0):,} casos ({default_counts.get(1, 0)/len(df)*100:.1f}%)")
else:
    print("   ❌ Dataset no encontrado")

# 2. Verificar modelo
print("\n2️⃣ Verificando modelo XGBoost...")
model_path = Path("models/real_xgboost_model.pkl")
if model_path.exists():
    model_data = joblib.load(model_path)
    print("   ✅ Modelo XGBoost cargado exitosamente")
    print(f"   ✅ Características del modelo: {len(model_data.get('feature_names', []))}")
else:
    print("   ⚠️ Modelo XGBoost no encontrado, buscando Random Forest...")
    model_path = Path("models/real_random_forest_model.pkl")
    if model_path.exists():
        model_data = joblib.load(model_path)
        print("   ✅ Modelo Random Forest cargado")
    else:
        print("   ❌ No se encontró ningún modelo")
        model_data = None

# 3. Hacer predicción de prueba
if model_data:
    print("\n3️⃣ Realizando predicción de prueba...")
    
    # Caso 1: Perfil de bajo riesgo (debería aprobar)
    test_case_1 = {
        'Edad': 35,
        'Ingresos': 3500000,
        'Estrato': 4,
        'Dependientes': 1,
        'Genero': 1,
        'TipoVivienda': 1,
        'Discapacidad': 2,
        'ValorFactura': 200000,
        'CupoAprobado': 2860000
    }
    
    print("\n   📋 Caso 1: Perfil de BAJO RIESGO")
    print(f"      Edad: {test_case_1['Edad']} años")
    print(f"      Ingresos: ${test_case_1['Ingresos']:,}")
    print(f"      Estrato: {test_case_1['Estrato']}")
    print(f"      Valor Factura: ${test_case_1['ValorFactura']:,}")
    print(f"      Cupo Aprobado: ${test_case_1['CupoAprobado']:,}")
    
    # Preparar datos
    df_test = pd.DataFrame([test_case_1])
    
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
    
    df_test = df_test.rename(columns=column_mapping)
    
    # Crear características adicionales
    df_test['invoice_to_income_ratio'] = df_test['invoice_value'] / (df_test['income'] + 1)
    df_test['limit_to_income_ratio'] = df_test['approved_limit'] / (df_test['income'] + 1)
    
    # Obtener características del modelo
    feature_names = model_data.get('feature_names', [])
    scaler = model_data.get('scaler')
    label_encoders = model_data.get('label_encoders', {})
    
    # Codificar variables categóricas
    for col, encoder in label_encoders.items():
        if col in df_test.columns:
            try:
                df_test[col] = encoder.transform(df_test[col].astype(str))
            except:
                df_test[col] = 0
    
    # Asegurar todas las características
    for feature in feature_names:
        if feature not in df_test.columns:
            df_test[feature] = 0
    
    # Reordenar columnas
    df_test = df_test.reindex(columns=feature_names, fill_value=0)
    
    # Escalar
    if scaler:
        df_scaled = scaler.transform(df_test)
        df_final = pd.DataFrame(df_scaled, columns=feature_names)
    else:
        df_final = df_test
    
    # Predecir
    model = model_data['model']
    prob_array = model.predict_proba(df_final)
    default_prob = prob_array[0, 1] * 100
    
    print(f"\n   🎯 Resultado de predicción:")
    print(f"      Probabilidad de RECHAZO: {default_prob:.2f}%")
    print(f"      Probabilidad de APROBACIÓN: {100-default_prob:.2f}%")
    
    if default_prob < 50:
        print("      ✅ DECISIÓN: APROBADO (Bajo riesgo)")
    else:
        print("      ❌ DECISIÓN: RECHAZADO (Alto riesgo)")
    
    # Caso 2: Perfil de alto riesgo
    test_case_2 = {
        'Edad': 18,
        'Ingresos': 1400000,
        'Estrato': 1,
        'Dependientes': 0,
        'Genero': 2,
        'TipoVivienda': 2,
        'Discapacidad': 2,
        'ValorFactura': 300000,
        'CupoAprobado': 0
    }
    
    print("\n   📋 Caso 2: Perfil de ALTO RIESGO")
    print(f"      Edad: {test_case_2['Edad']} años")
    print(f"      Ingresos: ${test_case_2['Ingresos']:,}")
    print(f"      Estrato: {test_case_2['Estrato']}")
    print(f"      Valor Factura: ${test_case_2['ValorFactura']:,}")
    print(f"      Cupo Aprobado: ${test_case_2['CupoAprobado']:,}")
    
    # Preparar y predecir caso 2
    df_test2 = pd.DataFrame([test_case_2])
    df_test2 = df_test2.rename(columns=column_mapping)
    df_test2['invoice_to_income_ratio'] = df_test2['invoice_value'] / (df_test2['income'] + 1)
    df_test2['limit_to_income_ratio'] = df_test2['approved_limit'] / (df_test2['income'] + 1)
    
    for col, encoder in label_encoders.items():
        if col in df_test2.columns:
            try:
                df_test2[col] = encoder.transform(df_test2[col].astype(str))
            except:
                df_test2[col] = 0
    
    for feature in feature_names:
        if feature not in df_test2.columns:
            df_test2[feature] = 0
    
    df_test2 = df_test2.reindex(columns=feature_names, fill_value=0)
    
    if scaler:
        df_scaled2 = scaler.transform(df_test2)
        df_final2 = pd.DataFrame(df_scaled2, columns=feature_names)
    else:
        df_final2 = df_test2
    
    prob_array2 = model.predict_proba(df_final2)
    default_prob2 = prob_array2[0, 1] * 100
    
    print(f"\n   🎯 Resultado de predicción:")
    print(f"      Probabilidad de RECHAZO: {default_prob2:.2f}%")
    print(f"      Probabilidad de APROBACIÓN: {100-default_prob2:.2f}%")
    
    if default_prob2 < 50:
        print("      ✅ DECISIÓN: APROBADO (Bajo riesgo)")
    else:
        print("      ❌ DECISIÓN: RECHAZADO (Alto riesgo)")

print("\n" + "=" * 70)
print("✅ PRUEBA COMPLETADA")
print("=" * 70)
print("\n🚀 Para usar el dashboard:")
print("   streamlit run dashboard_final.py --server.port 8508")
print("\n🌐 Dashboard disponible en:")
print("   http://localhost:8508")
print("\n" + "=" * 70)
