"""Análisis de viabilidades en DataCreditos.csv"""
import pandas as pd

# Cargar datos
df = pd.read_csv('docs/DataCreditos.csv')

print("=" * 70)
print("ANÁLISIS DE VIABILIDADES")
print("=" * 70)

# Distribución de viabilidades
print("\n📊 Distribución de Viabilidad:")
print(df['Viabilidad'].value_counts().sort_index())

print(f"\n📈 Total de registros: {len(df):,}")
print(f"✅ Viabilidad 1 (Aprobado): {len(df[df['Viabilidad'] == 1]):,}")
print(f"❌ Viabilidad 4 (Rechazado): {len(df[df['Viabilidad'] == 4]):,}")

# Filtrar solo viabilidades 1 y 4
df_filtered = df[df['Viabilidad'].isin([1, 4])]
print(f"\n🎯 Registros con Viabilidad 1 y 4: {len(df_filtered):,}")

# Porcentajes
total_1_4 = len(df_filtered)
pct_1 = (len(df[df['Viabilidad'] == 1]) / total_1_4) * 100
pct_4 = (len(df[df['Viabilidad'] == 4]) / total_1_4) * 100

print(f"\n📊 Distribución Filtrada:")
print(f"  Aprobado (1):  {pct_1:.2f}%")
print(f"  Rechazado (4): {pct_4:.2f}%")

# Verificar estructura
print(f"\n📋 Columnas disponibles:")
for i, col in enumerate(df.columns, 1):
    print(f"  {i}. {col}")

print("\n" + "=" * 70)
