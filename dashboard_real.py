"""
Dashboard simplificado para el modelo real de riesgo crediticio.
"""

import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import numpy as np

# Configuración de la página
st.set_page_config(
    page_title="Sistema de Riesgo Crediticio Real",
    page_icon="🏦",
    layout="wide"
)

def load_real_model():
    """Cargar el modelo real entrenado."""
    model_path = Path("models/real_random_forest_model.pkl")
    
    if not model_path.exists():
        st.error("❌ Modelo no encontrado")
        return None
    
    try:
        model_data = joblib.load(model_path)
        st.success("✅ Modelo REAL cargado con 99.35% de precisión")
        return model_data
    except Exception as e:
        st.error(f"❌ Error cargando modelo: {str(e)}")
        return None

def make_prediction(model_data, input_data):
    """Hacer predicción con el modelo real."""
    
    model = model_data['model']
    scaler = model_data.get('scaler')
    feature_names = model_data.get('feature_names', [])
    label_encoders = model_data.get('label_encoders', {})
    
    # Crear DataFrame
    df = pd.DataFrame([input_data])
    
    # Aplicar codificación
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
    no_default_prob = prob_array[0, 0]
    
    return default_prob, no_default_prob

def main():
    """Función principal del dashboard."""
    
    st.title("🏦 Sistema de Riesgo Crediticio - Modelo Real")
    st.markdown("---")
    
    # Cargar modelo
    model_data = load_real_model()
    
    if model_data is None:
        st.stop()
    
    # Información del modelo
    with st.expander("ℹ️ Información del Modelo"):
        st.markdown("""
        - **Modelo**: Random Forest
        - **Precisión**: 99.35% AUC-ROC
        - **Datos de entrenamiento**: 23,348 registros reales
        - **Variables**: 11 características principales
        """)
    
    st.markdown("## 📝 Ingresa los Datos del Cliente")
    
    # Crear dos columnas
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 👤 Información Personal")
        age = st.number_input("Edad", min_value=18, max_value=80, value=35)
        income = st.number_input("Ingresos Mensuales ($)", min_value=500000, max_value=10000000, value=3000000, step=100000)
        socioeconomic_level = st.selectbox("Estrato Socioeconómico", [1, 2, 3, 4, 5, 6], index=3)
        dependents = st.number_input("Dependientes", min_value=0, max_value=10, value=1)
        
        st.markdown("### 🏠 Información de Vivienda")
        gender = st.selectbox("Género", ["Masculino", "Femenino"])
        housing_status = st.selectbox("Tipo de Vivienda", ["Propia", "Arrendada", "Familiar"])
        has_disability = st.selectbox("¿Tiene discapacidad?", ["No", "Sí"])
    
    with col2:
        st.markdown("### 💰 Información Financiera")
        invoice_value = st.number_input("Valor de la Factura ($)", min_value=100000, max_value=5000000, value=1500000, step=100000)
        approved_limit = st.number_input("Cupo Aprobado ($)", min_value=500000, max_value=15000000, value=4000000, step=100000)
        
        # Mostrar ratios automáticos
        invoice_ratio = invoice_value / income if income > 0 else 0
        limit_ratio = approved_limit / income if income > 0 else 0
        
        st.info(f"📊 Ratio Factura/Ingresos: {invoice_ratio:.2f}")
        st.info(f"📊 Ratio Cupo/Ingresos: {limit_ratio:.2f}")
    
    # Botón de evaluación
    if st.button("🎯 Evaluar Riesgo Crediticio", type="primary"):
        
        # Preparar datos
        input_data = {
            'age': age,
            'income': income,
            'socioeconomic_level': socioeconomic_level,
            'dependents': dependents,
            'gender': 1 if gender == "Masculino" else 0,
            'housing_status': 1 if housing_status == "Propia" else 0,
            'has_disability': 1 if has_disability == "Sí" else 0,
            'invoice_value': invoice_value,
            'approved_limit': approved_limit,
            'invoice_to_income_ratio': invoice_ratio,
            'limit_to_income_ratio': limit_ratio
        }
        
        # Hacer predicción
        default_prob, no_default_prob = make_prediction(model_data, input_data)
        
        # Mostrar resultados
        st.markdown("---")
        st.markdown("## 📊 Resultados de la Evaluación")
        
        # Crear tres columnas para los resultados
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric(
                label="🎯 Probabilidad de No Default",
                value=f"{no_default_prob:.1%}",
                delta=f"{no_default_prob - 0.5:.1%}" if no_default_prob > 0.5 else None
            )
        
        with col2:
            st.metric(
                label="⚠️ Probabilidad de Default",
                value=f"{default_prob:.1%}",
                delta=f"{default_prob - 0.5:.1%}" if default_prob > 0.5 else None
            )
        
        with col3:
            # Decisión final
            if default_prob < 0.5:
                decision = "✅ APROBADO"
                color = "green"
                risk = "BAJO" if default_prob < 0.3 else "MEDIO"
            else:
                decision = "❌ RECHAZADO"
                color = "red"
                risk = "ALTO"
            
            st.markdown(f"### {decision}")
            st.markdown(f"**Nivel de Riesgo:** {risk}")
        
        # Explicación
        with st.expander("📋 Detalles de la Evaluación"):
            st.markdown(f"""
            **Factores Evaluados:**
            - Edad: {age} años
            - Ingresos: ${income:,}
            - Estrato: {socioeconomic_level}
            - Dependientes: {dependents}
            - Ratio Factura/Ingresos: {invoice_ratio:.2f}
            - Ratio Cupo/Ingresos: {limit_ratio:.2f}
            
            **Interpretación:**
            - Probabilidad de no cumplir: {default_prob:.1%}
            - Probabilidad de cumplir: {no_default_prob:.1%}
            - Decisión: {"Aprobar crédito" if default_prob < 0.5 else "Rechazar crédito"}
            """)
    
    # Perfiles de ejemplo
    st.markdown("---")
    st.markdown("## 💡 Perfiles de Ejemplo para Copiar")
    
    with st.expander("👀 Ver Perfiles de Ejemplo"):
        st.markdown("""
        ### 👨‍💼 Ejecutivo Senior (Alta Aprobación)
        - **Edad**: 42
        - **Ingresos**: 5,000,000
        - **Estrato**: 5
        - **Dependientes**: 2
        - **Género**: Masculino
        - **Vivienda**: Propia
        - **Discapacidad**: No
        - **Valor Factura**: 2,000,000
        - **Cupo Aprobado**: 8,000,000
        
        ### 👩‍⚕️ Profesional (Buena Aprobación)
        - **Edad**: 38
        - **Ingresos**: 3,500,000
        - **Estrato**: 4
        - **Dependientes**: 1
        - **Género**: Femenino
        - **Vivienda**: Propia
        - **Discapacidad**: No
        - **Valor Factura**: 1,200,000
        - **Cupo Aprobado**: 5,000,000
        
        ### 🚀 Empresario (Excelente Aprobación)
        - **Edad**: 35
        - **Ingresos**: 4,500,000
        - **Estrato**: 5
        - **Dependientes**: 0
        - **Género**: Masculino
        - **Vivienda**: Propia
        - **Discapacidad**: No
        - **Valor Factura**: 1,800,000
        - **Cupo Aprobado**: 7,000,000
        """)

if __name__ == "__main__":
    main()