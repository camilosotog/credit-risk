"""
Análisis de importancia de características en el modelo de riesgo crediticio
"""

import pandas as pd
import joblib
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

print("=" * 70)
print("ANÁLISIS DE IMPORTANCIA DE CARACTERÍSTICAS")
print("=" * 70)

# Cargar modelo XGBoost
model_path = Path("models/real_xgboost_model.pkl")

if not model_path.exists():
    print("\n❌ Modelo XGBoost no encontrado, intentando Random Forest...")
    model_path = Path("models/real_random_forest_model.pkl")

if model_path.exists():
    print(f"\n✅ Cargando modelo desde: {model_path}")
    model_data = joblib.load(model_path)
    
    model = model_data['model']
    feature_names = model_data.get('feature_names', [])
    
    print(f"\n📊 Total de características: {len(feature_names)}")
    
    # Obtener importancia de características
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        
        # Crear DataFrame con importancias
        feature_importance_df = pd.DataFrame({
            'Característica': feature_names,
            'Importancia': importances
        }).sort_values('Importancia', ascending=False)
        
        # Calcular porcentajes
        total_importance = feature_importance_df['Importancia'].sum()
        feature_importance_df['Porcentaje'] = (feature_importance_df['Importancia'] / total_importance * 100)
        
        print("\n" + "=" * 70)
        print("🏆 RANKING DE CARACTERÍSTICAS MÁS IMPORTANTES")
        print("=" * 70)
        
        # Mapeo de nombres técnicos a nombres legibles
        name_mapping = {
            'age': 'Edad',
            'income': 'Ingresos Mensuales',
            'socioeconomic_level': 'Estrato Socioeconómico',
            'dependents': 'Número de Dependientes',
            'gender': 'Género',
            'housing_status': 'Tipo de Vivienda',
            'has_disability': 'Tiene Discapacidad',
            'invoice_value': 'Valor de la Factura',
            'approved_limit': 'Cupo Aprobado',
            'invoice_to_income_ratio': 'Ratio Factura/Ingresos',
            'limit_to_income_ratio': 'Ratio Cupo/Ingresos'
        }
        
        print("\n")
        for idx, row in feature_importance_df.iterrows():
            feature = row['Característica']
            importance = row['Importancia']
            percentage = row['Porcentaje']
            readable_name = name_mapping.get(feature, feature)
            
            # Crear barra visual
            bar_length = int(percentage / 2)  # Escala para visualización
            bar = "█" * bar_length
            
            print(f"{readable_name:30s} | {bar:25s} {percentage:5.2f}%")
        
        print("\n" + "=" * 70)
        print("📈 TOP 3 CARACTERÍSTICAS MÁS DETERMINANTES")
        print("=" * 70)
        
        top_3 = feature_importance_df.head(3)
        for idx, (i, row) in enumerate(top_3.iterrows(), 1):
            feature = row['Característica']
            percentage = row['Porcentaje']
            readable_name = name_mapping.get(feature, feature)
            
            emoji = "🥇" if idx == 1 else "🥈" if idx == 2 else "🥉"
            print(f"\n{emoji} {idx}. {readable_name}")
            print(f"   Importancia: {percentage:.2f}%")
            
            # Interpretación
            if 'ratio' in feature.lower():
                print(f"   💡 Esta relación indica la capacidad de pago del solicitante")
            elif feature == 'approved_limit':
                print(f"   💡 El cupo aprobado refleja evaluaciones previas de riesgo")
            elif feature == 'income':
                print(f"   💡 Los ingresos son fundamentales para evaluar capacidad de pago")
            elif feature == 'invoice_value':
                print(f"   💡 El valor de la factura indica el monto de exposición al riesgo")
            elif feature == 'age':
                print(f"   💡 La edad puede correlacionar con estabilidad financiera")
        
        # Análisis adicional
        print("\n" + "=" * 70)
        print("🔍 ANÁLISIS DE CONCENTRACIÓN DE IMPORTANCIA")
        print("=" * 70)
        
        cumulative_importance = feature_importance_df['Porcentaje'].cumsum()
        
        # ¿Cuántas características explican el 80% de las decisiones?
        features_for_80 = (cumulative_importance <= 80).sum() + 1
        features_for_90 = (cumulative_importance <= 90).sum() + 1
        
        print(f"\n✅ Las primeras {features_for_80} características explican el 80% de las decisiones")
        print(f"✅ Las primeras {features_for_90} características explican el 90% de las decisiones")
        
        print("\n💡 CONCLUSIÓN:")
        top_feature = feature_importance_df.iloc[0]
        top_name = name_mapping.get(top_feature['Característica'], top_feature['Característica'])
        print(f"\nLa característica MÁS DETERMINANTE para aprobar o rechazar es:")
        print(f"🎯 {top_name.upper()} ({top_feature['Porcentaje']:.2f}%)")
        
        # Guardar visualización
        plt.figure(figsize=(12, 8))
        
        # Preparar datos para gráfico
        plot_df = feature_importance_df.copy()
        plot_df['Nombre Legible'] = plot_df['Característica'].map(name_mapping)
        
        # Crear gráfico de barras horizontal
        colors = ['#d4af37' if i == 0 else '#c0c0c0' if i == 1 else '#cd7f32' if i == 2 else '#4a90e2' 
                  for i in range(len(plot_df))]
        
        plt.barh(plot_df['Nombre Legible'], plot_df['Porcentaje'], color=colors)
        plt.xlabel('Importancia (%)', fontsize=12, fontweight='bold')
        plt.ylabel('Característica', fontsize=12, fontweight='bold')
        plt.title('Importancia de Características en el Modelo de Riesgo Crediticio\n(XGBoost - 99.29% AUC-ROC)', 
                  fontsize=14, fontweight='bold', pad=20)
        plt.grid(axis='x', alpha=0.3, linestyle='--')
        
        # Agregar valores en las barras
        for i, (idx, row) in enumerate(plot_df.iterrows()):
            plt.text(row['Porcentaje'] + 0.5, i, f"{row['Porcentaje']:.1f}%", 
                    va='center', fontsize=9, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig('plots/feature_importance_analysis.png', dpi=300, bbox_inches='tight')
        print(f"\n📊 Gráfico guardado en: plots/feature_importance_analysis.png")
        
    else:
        print("\n❌ El modelo no tiene atributo 'feature_importances_'")
        
else:
    print("\n❌ No se encontró ningún modelo entrenado")

print("\n" + "=" * 70)
