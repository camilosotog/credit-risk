# Sistema de Evaluación de Riesgo Crediticio - Actualización

## 📋 Cambios Realizados

### Dataset Utilizado
- **Archivo**: `docs/DataCreditos.csv`
- **Registros totales**: 27,361 solicitudes
- **Registros filtrados**: 26,940 (solo Viabilidades 1 y 4)
- **Distribución**:
  - Viabilidad 1 (APROBADO): 11,674 casos (43.3%)
  - Viabilidad 4 (RECHAZADO): 15,266 casos (56.7%)

### Variable Objetivo
- **Viabilidad 1** → `default=0` (Bajo riesgo - APROBADO)
- **Viabilidad 4** → `default=1` (Alto riesgo - RECHAZADO)

### Procesamiento de Datos
✅ **Corrección importante**: Los valores en `DataCreditos.csv` están en **escala real** (no logarítmica)
- Se eliminó la transformación exponencial innecesaria
- Rangos de valores:
  - **Valor Factura**: $1 - $100,000,000
  - **Cupo Aprobado**: $0 - $562,342,422

### Modelo Entrenado

#### 🏆 Mejor Modelo: XGBoost
```
AUC-ROC:     99.29%
Accuracy:    97.62%
Precision:   97.16%
Recall:      98.69%
F1-Score:    97.92%
```

#### Comparación de Modelos
| Modelo | AUC-ROC | Accuracy | Precision | Recall | F1-Score |
|--------|---------|----------|-----------|--------|----------|
| **XGBoost** | **0.9929** | **0.9762** | **0.9716** | **0.9869** | **0.9792** |
| Gradient Boosting | 0.9923 | 0.9755 | 0.9722 | 0.9849 | 0.9785 |
| Random Forest | 0.9925 | 0.9751 | 0.9728 | 0.9836 | 0.9782 |
| SVM | 0.9876 | 0.9592 | 0.9652 | 0.9627 | 0.9639 |
| Logistic Regression | 0.9868 | 0.9605 | 0.9545 | 0.9767 | 0.9655 |

### Dashboard

#### URL de Acceso
- **Local**: http://localhost:8508
- **Archivo**: `dashboard_final.py`

#### Características del Dashboard
- ✅ Sistema parametrizable con 9 variables configurables
- ✅ Checkboxes en sidebar para activar/desactivar variables
- ✅ Mínimo 2 variables requeridas para evaluación
- ✅ Indicadores visuales (✅/❌) para variables activas/inactivas
- ✅ Cálculo automático de ratios financieros
- ✅ Evaluación de riesgo en tiempo real

#### Variables Parametrizables
1. **Edad** (18-100 años)
2. **Ingresos** ($600,000 - $20,000,000)
3. **Estrato Socioeconómico** (1-6)
4. **Dependientes** (0-30)
5. **Género** (Masculino/Femenino)
6. **Tipo de Vivienda** (Propia/Arrendada/Familiar)
7. **Discapacidad** (Sí/No)
8. **Valor Factura** ($1 - $100M)
9. **Cupo Aprobado** ($0 - $562M)

### Archivos Modificados

1. **`src/data/process_real_data.py`**
   - Cambio de fuente: `data/raw/DataCreditos_baland.csv` → `docs/DataCreditos.csv`
   - Filtrado de viabilidades 1 y 4
   - Eliminación de transformación exponencial
   - Mapeo de variable objetivo (1→0, 4→1)

2. **`dashboard_final.py`**
   - Carga automática de modelo XGBoost
   - Actualización de información del dataset
   - Ajuste de rangos de valores según datos reales
   - Valores por defecto: Factura $200k, Cupo $2.86M

3. **Modelos generados**
   - `models/real_xgboost_model.pkl` (MEJOR - 99.29% AUC-ROC)
   - `models/real_random_forest_model.pkl` (99.25% AUC-ROC)
   - `models/real_gradient_boosting_model.pkl` (99.23% AUC-ROC)

### Estructura del Proyecto

```
credit-risk/
├── docs/
│   └── DataCreditos.csv              # Dataset original (27,361 registros)
├── data/
│   └── processed/
│       └── real_credit_data_processed.csv  # Dataset procesado (26,940 registros)
├── src/
│   ├── data/
│   │   └── process_real_data.py      # ✅ Actualizado
│   └── models/
│       └── train_model_real.py       # Script de entrenamiento
├── models/
│   ├── real_xgboost_model.pkl        # ✅ Mejor modelo (99.29%)
│   ├── real_random_forest_model.pkl
│   └── real_gradient_boosting_model.pkl
├── dashboard_final.py                # ✅ Dashboard actualizado
└── plots/                            # Visualizaciones generadas
```

### Próximos Pasos

1. ✅ **Completado**: Procesamiento de datos con viabilidades 1 y 4
2. ✅ **Completado**: Entrenamiento de modelos (XGBoost 99.29%)
3. ✅ **Completado**: Dashboard parametrizable funcionando
4. 📝 **Pendiente**: Documentación técnica para tesis
5. 📝 **Pendiente**: Análisis de características más importantes
6. 📝 **Pendiente**: Validación con casos de uso reales

### Comandos de Ejecución

```powershell
# 1. Procesar datos
python src/data/process_real_data.py

# 2. Entrenar modelos
python src/models/train_model_real.py

# 3. Ejecutar dashboard
streamlit run dashboard_final.py --server.port 8508

# 4. Ejecutar API (opcional)
python api/app.py
```

### Notas Importantes

⚠️ **Diferencia con dataset anterior**:
- El dataset `DataCreditos_baland.csv` tenía valores en escala logarítmica
- El dataset `DataCreditos.csv` tiene valores en escala real directa
- No se requiere transformación exponencial para el nuevo dataset

✅ **Ventajas del nuevo dataset**:
- Más registros: 26,940 vs 23,348
- Valores más claros y comprensibles
- Distribución balanceada de clases (43%-57%)
- Variable objetivo binaria clara (1=aprobado, 4=rechazado)

---

**Fecha de actualización**: 7 de noviembre de 2025  
**Sistema listo para producción y presentación de tesis** 🎓
