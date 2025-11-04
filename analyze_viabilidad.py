"""
Análisis detallado de las viabilidades en el dataset original.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyze_viabilidad():
    """Analizar los valores de viabilidad en el dataset."""
    
    print("🔍 ANÁLISIS DE VIABILIDADES EN EL DATASET")
    print("="*60)
    
    # Cargar datos originales
    data_path = Path("data/raw/DataCreditos_baland.csv")
    
    if not data_path.exists():
        print("❌ Archivo no encontrado")
        return
    
    df = pd.read_csv(data_path)
    
    print(f"📊 Total de registros: {len(df):,}")
    print(f"📋 Columnas: {list(df.columns)}")
    
    # Análisis de la columna Viabilidad
    print("\\n" + "="*40)
    print("📈 ANÁLISIS DE VIABILIDAD")
    print("="*40)
    
    viabilidad_counts = df['Viabilidad'].value_counts().sort_index()
    viabilidad_percent = df['Viabilidad'].value_counts(normalize=True).sort_index() * 100
    
    print("\\n📊 Distribución de Viabilidades:")
    print("-" * 35)
    for val in viabilidad_counts.index:
        count = viabilidad_counts[val]
        percent = viabilidad_percent[val]
        print(f"  Viabilidad {val}: {count:,} registros ({percent:.1f}%)")
    
    # Valores únicos
    unique_vals = sorted(df['Viabilidad'].unique())
    print(f"\\n🎯 Valores únicos de Viabilidad: {unique_vals}")
    
    # Estadísticas descriptivas
    print("\\n📈 Estadísticas de Viabilidad:")
    print(f"  Mínimo: {df['Viabilidad'].min()}")
    print(f"  Máximo: {df['Viabilidad'].max()}")
    print(f"  Media: {df['Viabilidad'].mean():.2f}")
    print(f"  Mediana: {df['Viabilidad'].median()}")
    
    # Analizar qué significa cada viabilidad
    print("\\n" + "="*40)
    print("🔍 ANÁLISIS POR VIABILIDAD")
    print("="*40)
    
    for viab in unique_vals:
        subset = df[df['Viabilidad'] == viab]
        print(f"\\n📋 VIABILIDAD {viab} ({len(subset):,} registros):")
        
        # Estadísticas de ingresos
        ingresos = subset['IngresoPrincipalMensual']
        print(f"  💰 Ingresos promedio: ${ingresos.mean():,.0f}")
        print(f"  💰 Ingresos mediana: ${ingresos.median():,.0f}")
        
        # Estadísticas de edad
        edad = subset['Edad']
        print(f"  👤 Edad promedio: {edad.mean():.1f} años")
        
        # Estrato más común
        estrato_comun = subset['Estrato'].mode().iloc[0]
        print(f"  🏠 Estrato más común: {estrato_comun}")
        
        # Cupo aprobado
        cupo = subset['ValorCupoAprobado']
        cupo_promedio = cupo.mean()
        cupo_ceros = (cupo == 0).sum()
        print(f"  💳 Cupo promedio: ${cupo_promedio:,.0f}")
        print(f"  💳 Registros con cupo 0: {cupo_ceros:,} ({cupo_ceros/len(subset)*100:.1f}%)")
    
    # Crear visualización
    plt.figure(figsize=(12, 8))
    
    # Gráfico de barras
    plt.subplot(2, 2, 1)
    viabilidad_counts.plot(kind='bar', color=['red', 'green', 'blue', 'orange'][:len(viabilidad_counts)])
    plt.title('Distribución de Viabilidades')
    plt.xlabel('Viabilidad')
    plt.ylabel('Cantidad de Registros')
    plt.xticks(rotation=0)
    
    # Gráfico de pie
    plt.subplot(2, 2, 2)
    plt.pie(viabilidad_counts.values, labels=viabilidad_counts.index, autopct='%1.1f%%')
    plt.title('Proporción de Viabilidades')
    
    # Boxplot de ingresos por viabilidad
    plt.subplot(2, 2, 3)
    df.boxplot(column='IngresoPrincipalMensual', by='Viabilidad', ax=plt.gca())
    plt.title('Ingresos por Viabilidad')
    plt.xlabel('Viabilidad')
    plt.ylabel('Ingresos Mensuales')
    
    # Boxplot de edad por viabilidad
    plt.subplot(2, 2, 4)
    df.boxplot(column='Edad', by='Viabilidad', ax=plt.gca())
    plt.title('Edad por Viabilidad')
    plt.xlabel('Viabilidad')
    plt.ylabel('Edad')
    
    plt.tight_layout()
    plt.savefig('plots/viabilidad_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # Análisis de correlación con otras variables
    print("\\n" + "="*40)
    print("🔗 CORRELACIONES CON VIABILIDAD")
    print("="*40)
    
    numeric_cols = ['Edad', 'PersonasACargo', 'IngresoPrincipalMensual', 'ValorFactura', 'ValorCupoAprobado', 'Estrato']
    correlations = df[numeric_cols + ['Viabilidad']].corr()['Viabilidad'].drop('Viabilidad')
    
    print("\\n📊 Correlación de Viabilidad con otras variables:")
    print("-" * 50)
    for var, corr in correlations.sort_values(key=abs, ascending=False).items():
        direction = "📈" if corr > 0 else "📉"
        strength = "Fuerte" if abs(corr) > 0.5 else "Moderada" if abs(corr) > 0.3 else "Débil"
        print(f"  {direction} {var:<25}: {corr:+.3f} ({strength})")
    
    return df

def interpret_viabilidad():
    """Interpretar qué significa cada valor de viabilidad."""
    
    print("\\n" + "="*60)
    print("🧠 INTERPRETACIÓN DE VIABILIDADES")
    print("="*60)
    
    interpretaciones = {
        1: {
            'nombre': '✅ VIABLE/APROBADO',
            'descripcion': 'Cliente con buen perfil crediticio',
            'accion': 'Aprobar crédito'
        },
        4: {
            'nombre': '❌ NO VIABLE/RECHAZADO', 
            'descripcion': 'Cliente con perfil de alto riesgo',
            'accion': 'Rechazar crédito'
        }
    }
    
    for viab, info in interpretaciones.items():
        print(f"\\n📋 VIABILIDAD {viab}: {info['nombre']}")
        print(f"   Descripción: {info['descripcion']}")
        print(f"   Acción: {info['accion']}")
    
    print("\\n💡 CONCLUSIÓN:")
    print("   • Viabilidad 1 = Cliente APROBADO (target = 0)")
    print("   • Viabilidad 4 = Cliente RECHAZADO (target = 1)")
    print("   • El modelo predice la probabilidad de ser Viabilidad 4 (default)")

if __name__ == "__main__":
    df = analyze_viabilidad()
    interpret_viabilidad()