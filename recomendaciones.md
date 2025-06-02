# Recomendaciones para Mejoras del Proyecto de Machine Learning

## Análisis del Dataset y Situación Actual

### Dataset
El dataset contiene información sobre partidas de CS:GO con columnas como:
- Información del mapa (`Map`)
- Equipo (`Team`, `InternalTeamId`)
- Información de la partida (`MatchId`, `RoundId`, `RoundWinner`, `MatchWinner`)
- Supervivencia (`Survived`)
- Métricas de rendimiento (`RoundKills`, `RoundAssists`, `RoundHeadshots`, `RoundFlankKills`)
- Equipamiento (`RoundStartingEquipmentValue`, `TeamStartingEquipmentValue`)
- Uso de granadas (`RLethalGrenadesThrown`, `RNonLethalGrenadesThrown`)
- Tipo de arma primaria utilizada (`PrimaryAssaultRifle`, `PrimarySniperRifle`, etc.)

### Estado Actual del Proyecto
1. Existen varios notebooks de análisis exploratorio y preparación de datos
2. Hay problemas con la carga de datos en algunas partes del código
3. Falta una estrategia clara de modelado
4. No hay una definición concreta del objetivo predictivo

## Recomendaciones de Mejora

### 1. Corrección de Problemas de Carga de Datos
- Corregir las rutas de carga de datos en los notebooks, especialmente en `04-Data_preparation2.ipynb`, que muestra errores de carga

### 2. Definición Clara de Objetivos Predictivos
Propongo dos objetivos de predicción viables:

**Modelo de Regresión:**
- **Variable objetivo**: `RoundKills` (Número de eliminaciones por ronda)
- **Métricas de evaluación**: RMSE, MAE, R²
- **Aplicación**: Predecir el rendimiento de un jugador en términos de eliminaciones

**Modelo de Clasificación:**
- **Variable objetivo**: `Survived` (Supervivencia del jugador en la ronda)
- **Métricas de evaluación**: Precisión, Recall, F1-Score, AUC-ROC
- **Aplicación**: Predecir si un jugador sobrevivirá a una ronda dadas ciertas condiciones

### 3. Mejora en la Preparación de Datos
- Implementar proceso de normalización o estandarización para variables numéricas
- Mejorar la codificación de variables categóricas (one-hot encoding para `Map` y `Team`)
- Implementar técnicas de detección y tratamiento de outliers
- Crear nuevas características derivadas, como:
  - Ratio de headshots (RoundHeadshots/RoundKills)
  - Eficiencia económica (RoundKills/RoundStartingEquipmentValue)
  - Variables de interacción entre el tipo de arma y los resultados

### 4. Selección de Características
- Realizar análisis de importancia de características usando:
  - Correlación con la variable objetivo
  - Feature importance de modelos como Random Forest
  - Técnicas como RFE (Recursive Feature Elimination)
- Eliminar variables con alta colinealidad

### 5. Propuesta de Modelos a Implementar

#### Para Regresión (`RoundKills`):
- Regresión Lineal (línea base)
- Regresión Ridge y Lasso (para controlar overfitting)
- Random Forest Regressor
- Gradient Boosting Regressor (XGBoost, LightGBM)
- Redes Neuronales (para capturar relaciones complejas)

#### Para Clasificación (`Survived`):
- Regresión Logística (línea base)
- Random Forest Classifier
- Gradient Boosting Classifier (XGBoost, LightGBM)
- SVM con kernel no lineal
- Redes Neuronales

### 6. Evaluación y Validación
- Implementar validación cruzada (k-fold)
- Estratificar la división de datos para mantener distribuciones similares
- Usar técnicas de ajuste de hiperparámetros como Grid Search o Random Search
- Evaluar el modelo en datos no vistos (hold-out test set)

### 7. Interpretabilidad
- Implementar técnicas de interpretabilidad como:
  - SHAP values
  - Partial Dependence Plots
  - Feature importance plots
- Documentar hallazgos y conclusiones para mejorar la comprensión del modelo

### 8. Flujo de Trabajo Estructurado
- Establecer un pipeline completo desde la carga hasta la evaluación
- Implementar un flujo reproducible con parámetros configurables
- Documentar cada paso del proceso para facilitar la replicación

## Próximos Pasos Inmediatos

1. Corregir el notebook `04-Data_preparation2.ipynb` para asegurar la carga correcta de datos
2. Finalizar la limpieza y transformación de datos
3. Iniciar pruebas con modelos de línea base (Regresión Lineal/Logística)
4. Evaluar resultados iniciales e iterar con modelos más complejos
5. Documentar el proceso completo y los resultados obtenidos

## Conclusión

Con estas mejoras, el proyecto estará mejor estructurado para desarrollar modelos predictivos efectivos, ya sea para predecir el número de eliminaciones por ronda (regresión) o la supervivencia del jugador (clasificación), proporcionando información valiosa sobre los factores que influyen en el rendimiento de los jugadores de CS:GO. 