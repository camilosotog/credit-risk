"""
Análisis de los valores de ValorFactura en el dataset original
"""
import pandas as pd
import numpy as np

# Cargar dataset
df = pd.read_csv('data/raw/DataCreditos_baland.csv')

print("="*70)
print("📊 ANÁLISIS DE VALORES DE FACTURA EN DATASET ORIGINAL")
print("="*70)

print("\n1️⃣ ESTADÍSTICAS DE ValorFactura:")
print(f"   Mínimo:    {df['ValorFactura'].min():.2f}")
print(f"   Máximo:    {df['ValorFactura'].max():.2f}")
print(f"   Promedio:  {df['ValorFactura'].mean():.2f}")
print(f"   Mediana:   {df['ValorFactura'].median():.2f}")
print(f"   Desv. Est: {df['ValorFactura'].std():.2f}")

print("\n2️⃣ DISTRIBUCIÓN DE VALORES:")
percentiles = [10, 25, 50, 75, 90, 95, 99]
print("   Percentiles:")
for p in percentiles:
    value = np.percentile(df['ValorFactura'], p)
    print(f"   {p:2d}%: {value:>15.2f}")

print("\n3️⃣ PRIMERAS 10 FILAS (muestra de datos):")
print(df[['ValorFactura', 'IngresoPrincipalMensual', 'ValorCupoAprobado']].head(10).to_string())

print("\n4️⃣ COMPARACIÓN ValorFactura vs IngresoPrincipalMensual:")
print(f"   Factura Promedio:  ${df['ValorFactura'].mean():,.0f}")
print(f"   Ingresos Promedio: ${df['IngresoPrincipalMensual'].mean():,.0f}")
print(f"   Ratio Promedio:    {(df['ValorFactura'].mean() / df['IngresoPrincipalMensual'].mean()):.4f}")

print("\n5️⃣ ¿ESTÁ EN ESCALA LOGARÍTMICA?")
print(f"   Rango Factura:  {df['ValorFactura'].min():.2f} a {df['ValorFactura'].max():.2f}")
print(f"   Rango Ingresos: ${df['IngresoPrincipalMensual'].min():,.0f} a ${df['IngresoPrincipalMensual'].max():,.0f}")

# Verificar si los valores están en log
sample = df[['ValorFactura', 'IngresoPrincipalMensual', 'ValorCupoAprobado']].head(5)
print("\n6️⃣ CONVERSIÓN DE VALORES LOG A REALES:")
print("   (Si ValorFactura está en log, exp(valor) sería el valor real)")
print()
for idx, row in sample.iterrows():
    factura_real = np.exp(row['ValorFactura'])
    cupo_real = np.exp(row['ValorCupoAprobado']) if row['ValorCupoAprobado'] > 0 else 0
    print(f"   Fila {idx+1}:")
    print(f"      ValorFactura: {row['ValorFactura']:.2f} → exp({row['ValorFactura']:.2f}) = ${factura_real:,.0f}")
    print(f"      Ingresos:     ${row['IngresoPrincipalMensual']:,.0f}")
    print(f"      CupoAprobado: {row['ValorCupoAprobado']:.2f} → exp({row['ValorCupoAprobado']:.2f}) = ${cupo_real:,.0f}")
    print()

print("="*70)
print("💡 CONCLUSIÓN:")
print("   Los valores de ValorFactura parecen estar en ESCALA LOGARÍTMICA")
print("   Para obtener el valor real en pesos: valor_real = exp(ValorFactura)")
print("="*70)
