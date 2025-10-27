"""
API REST para el sistema de evaluación de riesgo crediticio.
"""

from flask import Flask, request, jsonify, render_template_string
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
import logging
from typing import Dict, Any, List
import sys
import os

# Agregar el directorio raíz al path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from config.config import MODELS_DIR, API_CONFIG, RISK_THRESHOLDS

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Crear aplicación Flask
app = Flask(__name__)

# Variables globales
model_data = None
feature_engineer = None


def load_model():
    """Cargar el modelo entrenado (priorizar modelo real)."""
    global model_data
    
    try:
        # Priorizar modelo real
        real_model_path = MODELS_DIR / 'real_random_forest_model.pkl'
        
        if real_model_path.exists():
            model_data = joblib.load(real_model_path)
            logger.info(f"✅ Modelo REAL cargado: {model_data.get('model_name', 'Random Forest')}")
            return True
        
        # Buscar otros archivos de modelo
        model_files = list(MODELS_DIR.glob("*.pkl"))
        
        if not model_files:
            logger.error("No se encontraron modelos guardados")
            return False
        
        # Cargar el primer modelo encontrado
        model_path = model_files[0]
        model_data = joblib.load(model_path)
        
        logger.info(f"Modelo cargado: {model_data.get('model_name', 'Modelo')}")
        return True
        
    except Exception as e:
        logger.error(f"Error cargando modelo: {str(e)}")
        return False


def preprocess_input(data: Dict[str, Any]) -> pd.DataFrame:
    """
    Preprocesar datos de entrada para el modelo real.
    
    Args:
        data: Diccionario con datos del cliente
        
    Returns:
        DataFrame preprocesado
    """
    try:
        # Crear DataFrame con los datos de entrada
        df = pd.DataFrame([data])
        
        # Mapear campos de entrada a las características del modelo
        feature_mapping = {
            'age': 'age',
            'income': 'income', 
            'socioeconomic_level': 'socioeconomic_level',
            'dependents': 'dependents',
            'gender': 'gender',
            'housing_status': 'housing_status',
            'has_disability': 'has_disability',
            'invoice_value': 'invoice_value',
            'approved_limit': 'approved_limit'
        }
        
        # Aplicar mapeo básico
        for api_field, model_field in feature_mapping.items():
            if api_field in data and model_field not in df.columns:
                df[model_field] = data[api_field]
        
        # Codificar variables categóricas usando los encoders del modelo
        label_encoders = model_data.get('label_encoders', {})
        for col, encoder in label_encoders.items():
            if col in df.columns:
                try:
                    df[col] = encoder.transform(df[col].astype(str))
                except:
                    df[col] = 0  # Valor por defecto si no existe en encoder
        
        # Crear características derivadas
        if 'invoice_value' in df.columns and 'income' in df.columns:
            df['invoice_to_income_ratio'] = df['invoice_value'] / df['income'].replace(0, 1)
        
        if 'approved_limit' in df.columns and 'income' in df.columns:
            df['limit_to_income_ratio'] = df['approved_limit'] / df['income'].replace(0, 1)
        
        # Obtener características requeridas por el modelo
        required_features = model_data.get('feature_names', [])
        
        # Rellenar características faltantes con valores por defecto
        defaults = {
            'age': 35,
            'income': 2000000,
            'socioeconomic_level': 3,
            'dependents': 1,
            'gender': 0,
            'housing_status': 0,
            'has_disability': 0,
            'invoice_value': 1000000,
            'approved_limit': 2000000,
            'invoice_to_income_ratio': 0.5,
            'limit_to_income_ratio': 1.0
        }
        
        for feature in required_features:
            if feature not in df.columns:
                df[feature] = defaults.get(feature, 0)
        
        # Seleccionar solo las características que el modelo espera
        df = df.reindex(columns=required_features, fill_value=0)
        
        # Aplicar escalado si existe scaler
        scaler = model_data.get('scaler')
        if scaler:
            df_scaled = scaler.transform(df)
            df = pd.DataFrame(df_scaled, columns=required_features)
        
        return df
        
    except Exception as e:
        logger.error(f"Error preprocesando datos: {str(e)}")
        raise


def calculate_risk_level(probability: float) -> str:
    """
    Calcular nivel de riesgo basado en la probabilidad.
    
    Args:
        probability: Probabilidad de default
        
    Returns:
        Nivel de riesgo como string
    """
    if probability <= RISK_THRESHOLDS['low_risk']:
        return 'Bajo'
    elif probability <= RISK_THRESHOLDS['medium_risk']:
        return 'Medio'
    else:
        return 'Alto'


def get_recommendation(probability: float) -> Dict[str, Any]:
    """
    Obtener recomendación basada en la probabilidad.
    
    Args:
        probability: Probabilidad de default
        
    Returns:
        Diccionario con recomendación
    """
    risk_level = calculate_risk_level(probability)
    
    if risk_level == 'Bajo':
        decision = 'APROBAR'
        message = 'Cliente de bajo riesgo. Se recomienda aprobar el crédito.'
        suggested_interest = 'Tasa estándar'
    elif risk_level == 'Medio':
        decision = 'REVISAR'
        message = 'Cliente de riesgo medio. Se recomienda revisión adicional.'
        suggested_interest = 'Tasa estándar + 1-2%'
    else:
        decision = 'RECHAZAR'
        message = 'Cliente de alto riesgo. Se recomienda rechazar el crédito.'
        suggested_interest = 'No aplicable'
    
    return {
        'decision': decision,
        'risk_level': risk_level,
        'message': message,
        'suggested_interest': suggested_interest
    }


@app.route('/')
def home():
    """Página principal de la API."""
    html_template = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Sistema de Evaluación de Riesgo Crediticio</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }
            .container { max-width: 800px; margin: 0 auto; background-color: white; padding: 30px; border-radius: 10px; box-shadow: 0 0 10px rgba(0,0,0,0.1); }
            h1 { color: #2c3e50; text-align: center; }
            .endpoint { background-color: #ecf0f1; padding: 15px; margin: 10px 0; border-radius: 5px; }
            .method { color: #27ae60; font-weight: bold; }
            .url { color: #3498db; font-family: monospace; }
            .description { margin-top: 5px; color: #7f8c8d; }
            .status { text-align: center; padding: 20px; }
            .status.online { color: #27ae60; }
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🏦 Sistema de Evaluación de Riesgo Crediticio</h1>
            
            <div class="status online">
                <h2>✅ API Activa y Funcionando</h2>
                <p>Modelo cargado: {{ model_name }}</p>
            </div>
            
            <h2>📡 Endpoints Disponibles</h2>
            
            <div class="endpoint">
                <span class="method">POST</span> <span class="url">/predict</span>
                <div class="description">Evaluar riesgo crediticio de un cliente individual</div>
            </div>
            
            <div class="endpoint">
                <span class="method">POST</span> <span class="url">/predict_batch</span>
                <div class="description">Evaluar múltiples solicitudes de crédito</div>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/model/info</span>
                <div class="description">Obtener información del modelo actual</div>
            </div>
            
            <div class="endpoint">
                <span class="method">GET</span> <span class="url">/health</span>
                <div class="description">Verificar estado de salud de la API</div>
            </div>
            
            <h2>📋 Ejemplo de Uso</h2>
            <pre style="background-color: #2c3e50; color: white; padding: 15px; border-radius: 5px; overflow-x: auto;">
curl -X POST http://localhost:8000/predict \\
  -H "Content-Type: application/json" \\
  -d '{
    "age": 35,
    "income": 50000,
    "credit_score": 720,
    "debt_ratio": 0.3,
    "employment_years": 5,
    "loan_amount": 25000,
    "loan_term": 36
  }'
            </pre>
        </div>
    </body>
    </html>
    """
    
    model_name = model_data['model_name'] if model_data else 'No cargado'
    return render_template_string(html_template, model_name=model_name)


@app.route('/health', methods=['GET'])
def health_check():
    """Verificar estado de salud de la API."""
    status = {
        'status': 'healthy',
        'model_loaded': model_data is not None,
        'model_name': model_data['model_name'] if model_data else None,
        'api_version': '1.0.0'
    }
    
    return jsonify(status), 200


@app.route('/model/info', methods=['GET'])
def model_info():
    """Obtener información del modelo."""
    if not model_data:
        return jsonify({'error': 'Modelo no cargado'}), 500
    
    info = {
        'model_name': model_data['model_name'],
        'features_count': len(model_data.get('feature_names', [])),
        'feature_names': model_data.get('feature_names', [])[:10],  # Primeras 10
        'model_scores': model_data.get('model_scores', {}),
        'risk_thresholds': RISK_THRESHOLDS
    }
    
    return jsonify(info), 200


@app.route('/predict', methods=['POST'])
def predict_single():
    """
    Evaluar riesgo crediticio para un cliente individual.
    
    Ejemplo de entrada:
    {
        "age": 35,
        "income": 50000,
        "credit_score": 720,
        "debt_ratio": 0.3,
        "employment_years": 5,
        "loan_amount": 25000,
        "loan_term": 36
    }
    """
    try:
        # Verificar que el modelo esté cargado
        if not model_data:
            return jsonify({'error': 'Modelo no disponible'}), 500
        
        # Obtener datos del request
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No se proporcionaron datos'}), 400
        
        # Validar campos requeridos
        # Campos requeridos para el modelo real
        required_fields = ['age', 'income']  # Solo campos mínimos requeridos
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            return jsonify({
                'error': f'Campos faltantes: {", ".join(missing_fields)}'
            }), 400
        
        # Preprocesar datos
        df_processed = preprocess_input(data)
        
        # Realizar predicción
        model = model_data['model']
        prediction_proba = model.predict_proba(df_processed)[0, 1]
        prediction_class = model.predict(df_processed)[0]
        
        # Obtener recomendación
        recommendation = get_recommendation(prediction_proba)
        
        # Preparar respuesta
        response = {
            'customer_data': data,
            'prediction': {
                'default_probability': float(prediction_proba),
                'predicted_class': int(prediction_class),
                'risk_level': recommendation['risk_level']
            },
            'recommendation': recommendation,
            'model_info': {
                'model_name': model_data['model_name'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        logger.info(f"Predicción realizada: {recommendation['decision']} "
                   f"(probabilidad: {prediction_proba:.3f})")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error en predicción: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500


@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """
    Evaluar múltiples solicitudes de crédito.
    
    Ejemplo de entrada:
    {
        "data": [
            {"age": 35, "income": 50000, ...},
            {"age": 28, "income": 35000, ...}
        ]
    }
    """
    try:
        # Verificar que el modelo esté cargado
        if not model_data:
            return jsonify({'error': 'Modelo no disponible'}), 500
        
        # Obtener datos del request
        request_data = request.get_json()
        
        if not request_data or 'data' not in request_data:
            return jsonify({'error': 'Formato incorrecto. Esperado: {"data": [...]}'}), 400
        
        batch_data = request_data['data']
        
        if not isinstance(batch_data, list):
            return jsonify({'error': 'El campo "data" debe ser una lista'}), 400
        
        if len(batch_data) > 100:  # Límite por seguridad
            return jsonify({'error': 'Máximo 100 solicitudes por lote'}), 400
        
        results = []
        model = model_data['model']
        
        for i, customer_data in enumerate(batch_data):
            try:
                # Preprocesar datos
                df_processed = preprocess_input(customer_data)
                
                # Realizar predicción
                prediction_proba = model.predict_proba(df_processed)[0, 1]
                prediction_class = model.predict(df_processed)[0]
                
                # Obtener recomendación
                recommendation = get_recommendation(prediction_proba)
                
                # Agregar resultado
                results.append({
                    'index': i,
                    'customer_data': customer_data,
                    'prediction': {
                        'default_probability': float(prediction_proba),
                        'predicted_class': int(prediction_class),
                        'risk_level': recommendation['risk_level']
                    },
                    'recommendation': recommendation
                })
                
            except Exception as e:
                logger.error(f"Error procesando cliente {i}: {str(e)}")
                results.append({
                    'index': i,
                    'error': f'Error procesando datos: {str(e)}'
                })
        
        # Preparar respuesta
        response = {
            'processed_count': len(results),
            'results': results,
            'summary': {
                'approvals': len([r for r in results if r.get('recommendation', {}).get('decision') == 'APROBAR']),
                'reviews': len([r for r in results if r.get('recommendation', {}).get('decision') == 'REVISAR']),
                'rejections': len([r for r in results if r.get('recommendation', {}).get('decision') == 'RECHAZAR']),
                'errors': len([r for r in results if 'error' in r])
            },
            'model_info': {
                'model_name': model_data['model_name'],
                'timestamp': pd.Timestamp.now().isoformat()
            }
        }
        
        logger.info(f"Procesamiento por lotes completado: {len(results)} solicitudes")
        
        return jsonify(response), 200
        
    except Exception as e:
        logger.error(f"Error en predicción por lotes: {str(e)}")
        return jsonify({'error': f'Error interno: {str(e)}'}), 500


@app.errorhandler(404)
def not_found(error):
    """Manejar errores 404."""
    return jsonify({'error': 'Endpoint no encontrado'}), 404


@app.errorhandler(500)
def internal_error(error):
    """Manejar errores 500."""
    return jsonify({'error': 'Error interno del servidor'}), 500


def initialize_app():
    """Inicializar la aplicación."""
    logger.info("Inicializando API de evaluación de riesgo crediticio...")
    
    # Cargar modelo
    if not load_model():
        logger.error("No se pudo cargar el modelo. La API funcionará con funcionalidad limitada.")
    
    logger.info("API inicializada correctamente")


if __name__ == '__main__':
    # Inicializar aplicación
    initialize_app()
    
    # Configurar y ejecutar servidor
    host = API_CONFIG.get('host', '0.0.0.0')
    port = API_CONFIG.get('port', 8000)
    debug = API_CONFIG.get('debug', False)
    
    logger.info(f"Iniciando servidor en http://{host}:{port}")
    
    app.run(
        host=host,
        port=port,
        debug=debug,
        use_reloader=False  # Evitar recargar en modo debug
    )