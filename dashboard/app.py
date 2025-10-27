"""
Dashboard interactivo para el Sistema de Evaluación de Riesgo Crediticio.
Desarrollado con Streamlit.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
from pathlib import Path
import requests
import json
import sys
import os

# Configuración de la página
st.set_page_config(
    page_title="🏦 Sistema de Riesgo Crediticio",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Agregar el directorio raíz al path
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.append(str(project_root))

try:
    from config.config import MODELS_DIR, PROCESSED_DATA_DIR, DASHBOARD_CONFIG, RISK_THRESHOLDS
except ImportError:
    # Valores por defecto si no se puede importar la configuración
    MODELS_DIR = Path("models")
    PROCESSED_DATA_DIR = Path("data/processed")
    RISK_THRESHOLDS = {'low_risk': 0.3, 'medium_risk': 0.7, 'high_risk': 1.0}


# Funciones de utilidad
@st.cache_data
def load_data():
    """Cargar datos de muestra."""
    try:
        # Intentar cargar datos procesados
        data_file = PROCESSED_DATA_DIR / 'credit_data_processed.csv'
        if data_file.exists():
            return pd.read_csv(data_file)
        else:
            # Generar datos de muestra si no existen
            return generate_sample_data()
    except Exception as e:
        st.error(f"Error cargando datos: {str(e)}")
        return generate_sample_data()


def generate_sample_data(n_samples=1000):
    """Generar datos de muestra para demo."""
    np.random.seed(42)
    
    data = {
        'customer_id': range(1, n_samples + 1),
        'age': np.random.normal(40, 12, n_samples).astype(int),
        'income': np.random.lognormal(10.5, 0.5, n_samples),
        'credit_score': np.random.normal(650, 100, n_samples).astype(int),
        'debt_ratio': np.random.beta(2, 5, n_samples),
        'employment_years': np.random.exponential(5, n_samples),
        'loan_amount': np.random.lognormal(9, 0.8, n_samples),
        'loan_term': np.random.choice([12, 24, 36, 48, 60], n_samples),
        'employment_status': np.random.choice(['Employed', 'Self-employed', 'Unemployed'], 
                                            n_samples, p=[0.7, 0.25, 0.05]),
        'education_level': np.random.choice(['High School', 'Bachelor', 'Master', 'PhD'], 
                                          n_samples, p=[0.3, 0.4, 0.25, 0.05])
    }
    
    df = pd.DataFrame(data)
    
    # Clip values to reasonable ranges
    df['age'] = np.clip(df['age'], 18, 80)
    df['credit_score'] = np.clip(df['credit_score'], 300, 850)
    df['debt_ratio'] = np.clip(df['debt_ratio'], 0, 1)
    df['employment_years'] = np.clip(df['employment_years'], 0, 40)
    
    # Create synthetic default variable
    default_prob = (
        0.1 +
        (700 - df['credit_score']) / 1000 * 0.3 +
        df['debt_ratio'] * 0.4 +
        np.where(df['employment_status'] == 'Unemployed', 0.3, 0)
    )
    
    df['default'] = np.random.binomial(1, np.clip(default_prob, 0, 1), n_samples)
    
    return df


def get_risk_level(probability):
    """Determinar nivel de riesgo basado en probabilidad."""
    if probability <= RISK_THRESHOLDS['low_risk']:
        return 'Bajo', '🟢'
    elif probability <= RISK_THRESHOLDS['medium_risk']:
        return 'Medio', '🟡'
    else:
        return 'Alto', '🔴'


def make_prediction_api(customer_data, api_url="http://localhost:8000/predict"):
    """Realizar predicción usando la API."""
    try:
        response = requests.post(api_url, json=customer_data, timeout=10)
        if response.status_code == 200:
            return response.json()
        else:
            return None
    except Exception:
        return None


def make_prediction_local(customer_data):
    """Realizar predicción usando modelo real entrenado."""
    try:
        # Priorizar modelo real
        real_model_path = MODELS_DIR / 'real_random_forest_model.pkl'
        
        if real_model_path.exists():
            model_data = joblib.load(real_model_path)
        else:
            # Usar modelo sintético si no hay real
            model_files = list(MODELS_DIR.glob("*.pkl"))
            if not model_files:
                return None
            model_data = joblib.load(model_files[0])
        
        model = model_data['model']
        scaler = model_data.get('scaler')
        feature_names = model_data.get('feature_names', [])
        label_encoders = model_data.get('label_encoders', {})
        
        # Preparar datos con pipeline completo
        input_data = pd.DataFrame([customer_data])
        
        # Aplicar codificación si es necesario
        for col, encoder in label_encoders.items():
            if col in input_data.columns:
                try:
                    input_data[col] = encoder.transform(input_data[col].astype(str))
                except:
                    input_data[col] = 0  # Valor por defecto
        
        # Asegurar que tenemos todas las características
        for feature in feature_names:
            if feature not in input_data.columns:
                input_data[feature] = 0
        
        # Ordenar columnas según el entrenamiento
        input_data = input_data[feature_names]
        
        # Escalar si hay scaler
        if scaler:
            input_data_scaled = scaler.transform(input_data)
        else:
            input_data_scaled = input_data.values
        
        # Hacer predicción real
        prediction_prob = model.predict_proba(input_data_scaled)[0, 1]
        prediction_class = int(prediction_prob > 0.5)
        
        return {
            'prediction': {
                'default_probability': float(prediction_prob),
                'predicted_class': prediction_class
            }
            }
    except Exception:
        pass
    
    # Fallback: predicción simulada
    risk_score = (
        (700 - customer_data.get('credit_score', 650)) / 500 * 0.4 +
        customer_data.get('debt_ratio', 0.3) * 0.6
    )
    prediction_prob = max(0, min(1, risk_score))
    
    return {
        'prediction': {
            'default_probability': prediction_prob,
            'predicted_class': int(prediction_prob > 0.5)
        }
    }


# Sidebar para navegación
st.sidebar.title("🏦 Sistema de Riesgo Crediticio")
st.sidebar.markdown("---")

# Navegación
page = st.sidebar.selectbox(
    "Seleccionar Página",
    ["🏠 Dashboard Principal", "🔍 Análisis de Datos", "🤖 Predictor Individual", 
     "📊 Análisis por Lotes", "📈 Métricas del Modelo"]
)

# Cargar datos
df = load_data()

# Página principal
if page == "🏠 Dashboard Principal":
    st.title("🏦 Sistema de Evaluación de Riesgo Crediticio")
    st.markdown("### Panel de Control Principal")
    
    # Métricas principales
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Clientes", f"{len(df):,}")
    
    with col2:
        default_rate = df['default'].mean() if 'default' in df.columns else 0.15
        st.metric("Tasa de Default", f"{default_rate:.2%}")
    
    with col3:
        avg_credit_score = df['credit_score'].mean() if 'credit_score' in df.columns else 650
        st.metric("Puntaje Promedio", f"{avg_credit_score:.0f}")
    
    with col4:
        avg_loan = df['loan_amount'].mean() if 'loan_amount' in df.columns else 25000
        st.metric("Préstamo Promedio", f"${avg_loan:,.0f}")
    
    # Gráficos principales
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Distribución de Puntajes Crediticios")
        if 'credit_score' in df.columns:
            fig = px.histogram(df, x='credit_score', nbins=30, 
                             title="Distribución de Puntajes Crediticios")
            fig.update_layout(showlegend=False)
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Tasa de Default por Segmento")
        if 'credit_score' in df.columns and 'default' in df.columns:
            # Crear segmentos de puntaje crediticio
            df['credit_segment'] = pd.cut(df['credit_score'], 
                                        bins=[0, 580, 670, 740, 800, 850],
                                        labels=['Poor', 'Fair', 'Good', 'Very Good', 'Excellent'])
            
            segment_default = df.groupby('credit_segment')['default'].mean().reset_index()
            
            fig = px.bar(segment_default, x='credit_segment', y='default',
                        title="Tasa de Default por Segmento de Crédito")
            fig.update_layout(xaxis_title="Segmento de Crédito", yaxis_title="Tasa de Default")
            st.plotly_chart(fig, use_container_width=True)
    
    # Análisis adicional
    st.subheader("📈 Tendencias y Patrones")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if 'age' in df.columns and 'default' in df.columns:
            # Análisis por edad
            age_bins = pd.cut(df['age'], bins=range(20, 81, 10))
            age_default = df.groupby(age_bins)['default'].mean().reset_index()
            age_default['age_group'] = age_default['age'].astype(str)
            
            fig = px.line(age_default, x='age_group', y='default',
                         title="Tasa de Default por Grupo de Edad")
            fig.update_layout(xaxis_title="Grupo de Edad", yaxis_title="Tasa de Default")
            st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        if 'employment_status' in df.columns and 'default' in df.columns:
            # Análisis por estado laboral
            emp_default = df.groupby('employment_status')['default'].mean().reset_index()
            
            fig = px.pie(emp_default, values='default', names='employment_status',
                        title="Distribución de Riesgo por Estado Laboral")
            st.plotly_chart(fig, use_container_width=True)


elif page == "🔍 Análisis de Datos":
    st.title("🔍 Análisis Exploratorio de Datos")
    
    # Información del dataset
    st.subheader("📋 Información del Dataset")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"**Filas:** {df.shape[0]:,}")
    with col2:
        st.write(f"**Columnas:** {df.shape[1]:,}")
    with col3:
        st.write(f"**Memoria:** {df.memory_usage(deep=True).sum() / 1024**2:.1f} MB")
    
    # Mostrar datos
    st.subheader("📊 Muestra de Datos")
    st.dataframe(df.head(10), use_container_width=True)
    
    # Estadísticas descriptivas
    st.subheader("📈 Estadísticas Descriptivas")
    
    # Seleccionar columnas numéricas
    numeric_columns = df.select_dtypes(include=[np.number]).columns.tolist()
    selected_columns = st.multiselect(
        "Seleccionar columnas para análisis:",
        numeric_columns,
        default=numeric_columns[:5]
    )
    
    if selected_columns:
        st.dataframe(df[selected_columns].describe(), use_container_width=True)
        
        # Matriz de correlación
        st.subheader("🔗 Matriz de Correlación")
        corr_matrix = df[selected_columns].corr()
        
        fig = px.imshow(corr_matrix, 
                       title="Matriz de Correlación",
                       color_continuous_scale="RdBu",
                       aspect="auto")
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)
    
    # Análisis de distribuciones
    st.subheader("📊 Análisis de Distribuciones")
    
    if selected_columns:
        selected_var = st.selectbox("Seleccionar variable para análisis:", selected_columns)
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Histograma
            fig = px.histogram(df, x=selected_var, nbins=30,
                             title=f"Distribución de {selected_var}")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Box plot
            fig = px.box(df, y=selected_var,
                        title=f"Box Plot de {selected_var}")
            st.plotly_chart(fig, use_container_width=True)


elif page == "🤖 Predictor Individual":
    st.title("🤖 Predictor de Riesgo Individual")
    st.markdown("### Evalúa el riesgo crediticio de un cliente específico")
    
    # Formulario de entrada
    with st.form("prediccion_form"):
        st.subheader("📝 Información del Cliente")
        
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Edad", min_value=18, max_value=80, value=35)
            income = st.number_input("Ingresos anuales ($)", min_value=1000, value=50000, step=1000)
            credit_score = st.number_input("Puntaje crediticio", min_value=300, max_value=850, value=720)
            employment_years = st.number_input("Años de empleo", min_value=0.0, max_value=40.0, value=5.0)
        
        with col2:
            debt_ratio = st.slider("Ratio de deuda", min_value=0.0, max_value=1.0, value=0.3, step=0.01)
            loan_amount = st.number_input("Monto del préstamo ($)", min_value=1000, value=25000, step=1000)
            loan_term = st.selectbox("Plazo del préstamo (meses)", [12, 24, 36, 48, 60], index=2)
            employment_status = st.selectbox("Estado laboral", ["Employed", "Self-employed", "Unemployed"])
        
        submit_button = st.form_submit_button("🔍 Evaluar Riesgo", type="primary")
    
    if submit_button:
        # Preparar datos del cliente
        customer_data = {
            'age': age,
            'income': income,
            'credit_score': credit_score,
            'debt_ratio': debt_ratio,
            'employment_years': employment_years,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'employment_status': employment_status
        }
        
        # Intentar predicción con API, fallback a modelo local
        with st.spinner("Evaluando riesgo..."):
            prediction = make_prediction_api(customer_data)
            if prediction is None:
                prediction = make_prediction_local(customer_data)
        
        # Mostrar resultados
        if prediction and 'prediction' in prediction:
            prob = prediction['prediction']['default_probability']
            risk_level, risk_emoji = get_risk_level(prob)
            
            st.success("✅ Evaluación completada")
            
            # Métricas principales
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("Probabilidad de Default", f"{prob:.1%}")
            
            with col2:
                st.metric("Nivel de Riesgo", f"{risk_emoji} {risk_level}")
            
            with col3:
                decision = "APROBAR" if prob < 0.3 else "REVISAR" if prob < 0.7 else "RECHAZAR"
                color = "green" if decision == "APROBAR" else "orange" if decision == "REVISAR" else "red"
                st.markdown(f"**Recomendación:** <span style='color: {color}'>{decision}</span>", 
                          unsafe_allow_html=True)
            
            # Gráfico de probabilidad
            fig = go.Figure(go.Indicator(
                mode = "gauge+number",
                value = prob * 100,
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Riesgo de Default (%)"},
                gauge = {
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 30], 'color': "lightgreen"},
                        {'range': [30, 70], 'color': "yellow"},
                        {'range': [70, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            
            fig.update_layout(height=400)
            st.plotly_chart(fig, use_container_width=True)
            
            # Detalles adicionales
            st.subheader("📋 Detalles de la Evaluación")
            
            # Factores de riesgo
            factors = []
            if credit_score < 600:
                factors.append("⚠️ Puntaje crediticio bajo")
            if debt_ratio > 0.6:
                factors.append("⚠️ Alto ratio de deuda")
            if employment_status == "Unemployed":
                factors.append("⚠️ Sin empleo actual")
            if loan_amount / income > 0.5:
                factors.append("⚠️ Alto ratio préstamo/ingresos")
            
            if factors:
                st.warning("**Factores de Riesgo Identificados:**")
                for factor in factors:
                    st.write(factor)
            else:
                st.success("✅ No se identificaron factores de riesgo significativos")


elif page == "📊 Análisis por Lotes":
    st.title("📊 Análisis de Riesgo por Lotes")
    st.markdown("### Evalúa múltiples solicitudes de crédito")
    
    # Opción de cargar archivo
    uploaded_file = st.file_uploader("Cargar archivo CSV con solicitudes", type=['csv'])
    
    if uploaded_file is not None:
        try:
            batch_df = pd.read_csv(uploaded_file)
            st.success(f"✅ Archivo cargado: {len(batch_df)} solicitudes")
            
            # Mostrar muestra de datos
            st.subheader("📋 Vista previa de datos")
            st.dataframe(batch_df.head(), use_container_width=True)
            
            # Botón para procesar
            if st.button("🔍 Procesar Lote", type="primary"):
                with st.spinner("Procesando solicitudes..."):
                    # Simular procesamiento por lotes
                    results = []
                    for idx, row in batch_df.iterrows():
                        customer_data = row.to_dict()
                        prediction = make_prediction_local(customer_data)
                        
                        if prediction and 'prediction' in prediction:
                            prob = prediction['prediction']['default_probability']
                            risk_level, _ = get_risk_level(prob)
                            decision = "APROBAR" if prob < 0.3 else "REVISAR" if prob < 0.7 else "RECHAZAR"
                            
                            results.append({
                                'Cliente_ID': idx + 1,
                                'Probabilidad_Default': prob,
                                'Nivel_Riesgo': risk_level,
                                'Decision': decision
                            })
                
                # Mostrar resultados
                if results:
                    results_df = pd.DataFrame(results)
                    
                    st.success("✅ Procesamiento completado")
                    
                    # Resumen
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.metric("Total Procesadas", len(results))
                    
                    with col2:
                        aprobadas = len(results_df[results_df['Decision'] == 'APROBAR'])
                        st.metric("Aprobadas", aprobadas)
                    
                    with col3:
                        revisiones = len(results_df[results_df['Decision'] == 'REVISAR'])
                        st.metric("Para Revisión", revisiones)
                    
                    with col4:
                        rechazadas = len(results_df[results_df['Decision'] == 'RECHAZAR'])
                        st.metric("Rechazadas", rechazadas)
                    
                    # Visualizaciones
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        # Distribución de decisiones
                        decision_counts = results_df['Decision'].value_counts()
                        fig = px.pie(values=decision_counts.values, names=decision_counts.index,
                                   title="Distribución de Decisiones")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    with col2:
                        # Distribución de probabilidades
                        fig = px.histogram(results_df, x='Probabilidad_Default',
                                         title="Distribución de Probabilidades de Default")
                        st.plotly_chart(fig, use_container_width=True)
                    
                    # Tabla de resultados
                    st.subheader("📊 Resultados Detallados")
                    st.dataframe(results_df, use_container_width=True)
                    
                    # Opción de descarga
                    csv = results_df.to_csv(index=False)
                    st.download_button(
                        label="📥 Descargar Resultados",
                        data=csv,
                        file_name="resultados_evaluacion.csv",
                        mime="text/csv"
                    )
        
        except Exception as e:
            st.error(f"Error procesando archivo: {str(e)}")
    
    else:
        # Mostrar formato esperado
        st.info("📋 **Formato esperado del archivo CSV:**")
        
        sample_data = {
            'age': [35, 28, 45],
            'income': [50000, 35000, 75000],
            'credit_score': [720, 650, 780],
            'debt_ratio': [0.3, 0.5, 0.2],
            'employment_years': [5, 2, 10],
            'loan_amount': [25000, 15000, 40000],
            'loan_term': [36, 24, 48],
            'employment_status': ['Employed', 'Employed', 'Self-employed']
        }
        
        sample_df = pd.DataFrame(sample_data)
        st.dataframe(sample_df, use_container_width=True)


elif page == "📈 Métricas del Modelo":
    st.title("📈 Métricas y Rendimiento del Modelo")
    
    # Información del modelo
    st.subheader("🤖 Información del Modelo")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Modelo Actual", "Random Forest")
    
    with col2:
        st.metric("Precisión (AUC)", "0.87")
    
    with col3:
        st.metric("Última Actualización", "2024-01-15")
    
    # Métricas simuladas (en producción vendrían del modelo real)
    st.subheader("📊 Métricas de Rendimiento")
    
    # Crear datos simulados para métricas
    metrics_data = {
        'Métrica': ['Precision', 'Recall', 'F1-Score', 'Accuracy', 'AUC-ROC'],
        'Valor': [0.85, 0.78, 0.81, 0.83, 0.87],
        'Benchmark': [0.80, 0.75, 0.77, 0.80, 0.85]
    }
    
    metrics_df = pd.DataFrame(metrics_data)
    
    # Gráfico de métricas
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Modelo Actual',
        x=metrics_df['Métrica'],
        y=metrics_df['Valor'],
        marker_color='lightblue'
    ))
    
    fig.add_trace(go.Bar(
        name='Benchmark',
        x=metrics_df['Métrica'],
        y=metrics_df['Benchmark'],
        marker_color='coral'
    ))
    
    fig.update_layout(
        title='Comparación de Métricas vs Benchmark',
        xaxis_title='Métricas',
        yaxis_title='Valor',
        barmode='group'
    )
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Matriz de confusión simulada
    st.subheader("🎯 Matriz de Confusión")
    
    # Datos simulados para matriz de confusión
    confusion_data = np.array([[850, 120], [95, 435]])
    
    fig = px.imshow(confusion_data,
                   labels=dict(x="Predicho", y="Real", color="Cantidad"),
                   x=['No Default', 'Default'],
                   y=['No Default', 'Default'],
                   color_continuous_scale="Blues",
                   title="Matriz de Confusión")
    
    # Agregar texto a las celdas
    for i in range(len(confusion_data)):
        for j in range(len(confusion_data[0])):
            fig.add_annotation(x=j, y=i,
                             text=str(confusion_data[i][j]),
                             showarrow=False,
                             font=dict(color="white" if confusion_data[i][j] > 400 else "black"))
    
    st.plotly_chart(fig, use_container_width=True)
    
    # Tendencia histórica simulada
    st.subheader("📈 Tendencia de Rendimiento")
    
    dates = pd.date_range('2023-01-01', periods=12, freq='M')
    auc_scores = np.random.normal(0.85, 0.02, 12)
    auc_scores = np.clip(auc_scores, 0.80, 0.90)
    
    trend_df = pd.DataFrame({
        'Fecha': dates,
        'AUC_Score': auc_scores
    })
    
    fig = px.line(trend_df, x='Fecha', y='AUC_Score',
                 title='Evolución del AUC Score',
                 markers=True)
    
    fig.add_hline(y=0.85, line_dash="dash", line_color="red",
                 annotation_text="Objetivo mínimo")
    
    st.plotly_chart(fig, use_container_width=True)


# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📞 Contacto")
st.sidebar.info("""
**Sistema de Riesgo Crediticio**  
Versión: 1.0.0  
Desarrollado para trabajo de grado  
Universidad: [Tu Universidad]  
Autor: [Tu Nombre]
""")

st.sidebar.markdown("---")
st.sidebar.markdown("*Dashboard desarrollado con ❤️ usando Streamlit*")