# Análisis Competitivo de CS:GO mediante Machine Learning

Equipo:

Braihan Gonzalez

Gabriel Hinostroza

Luis Salamanca


## Introducción

Este proyecto analiza datos de partidas profesionales de Counter-Strike: Global Offensive (CS:GO) para identificar patrones y factores determinantes del rendimiento en competiciones de alto nivel. Utilizamos técnicas avanzadas de machine learning para extraer insights valiosos que puedan ser aplicados tanto por jugadores profesionales como por desarrolladores.

## Resumen del Dataset

Trabajamos con el dataset `Anexo_ET_demo_round_traces_2022.csv` que contiene información detallada por ronda de partidas profesionales, incluyendo:

- **Información contextual**: Mapa, equipo, ID de partida y ronda
- **Resultados**: Ganador de ronda, supervivencia, victoria de partida
- **Métricas de movimiento**: Tiempo vivo, distancia recorrida
- **Uso de granadas**: Letales y no letales lanzadas
- **Armamento**: Preferencias de armas (rifles de asalto, francotiradores, etc.)
- **Rendimiento ofensivo**: Tiempo del primer asesinato, eliminaciones, asistencias, headshots
- **Economía**: Valor del equipamiento inicial

## Preguntas de Investigación

1. ¿Cómo afecta la distribución del armamento del equipo a las probabilidades de victoria?
2. ¿Existe correlación entre el valor del equipamiento inicial y el número de eliminaciones?
3. ¿Los equipos que logran el primer asesinato más rápido tienen mayor tasa de victoria?
4. ¿Es la distancia recorrida un indicador significativo del estilo de juego agresivo o defensivo?

## Metodología

Nuestro análisis sigue un enfoque de tres fases:

### 1. Preparación y Exploración
- Limpieza y normalización de datos
- Análisis exploratorio para identificar tendencias iniciales
- Visualización de relaciones entre variables clave

### 2. Modelado Predictivo
- Desarrollo de modelos de clasificación para predecir victorias de ronda
- Modelos de regresión para estimar métricas de rendimiento
- Evaluación comparativa de diferentes algoritmos

### 3. Análisis Estratégico
- Segmentación de estilos de juego por equipo
- Identificación de ventajas por lado (CT/T) en diferentes mapas
- Recomendaciones tácticas basadas en hallazgos estadísticos

## Métricas de Evaluación

Para los modelos de clasificación:
- Accuracy, Precision, Recall y F1-Score
- Curvas ROC y AUC
- Matrices de confusión

Para los modelos de regresión:
- Error Cuadrático Medio (MSE)
- Error Absoluto Medio (MAE)
- Coeficiente de determinación (R²)

## Aplicaciones Potenciales

- **Para equipos profesionales**: Optimización de estrategias y decisiones de compra
- **Para organizadores de torneos**: Balanceo de mapas y reglas de competición
- **Para desarrolladores**: Ajustes al equilibrio de armas y mecánicas de juego
- **Para analistas**: Nuevas métricas para evaluar el rendimiento individual y de equipo

## Estructura del Proyecto

```
/
├── data/
│   ├── 01_raw/            # Datos originales sin procesar
│   ├── 02_intermediate/   # Datos con transformaciones intermedias
│   └── 03_processed/      # Datos listos para modelado
├── notebooks/             # Jupyter notebooks de análisis
├── src/                   # Código fuente para funciones reutilizables
└── models/                # Modelos entrenados y resultados
```

## Tecnologías Utilizadas

- Python 3.8+
- Pandas y NumPy para manipulación de datos
- Scikit-learn para modelado predictivo
- XGBoost y LightGBM para modelos avanzados
- Matplotlib y Seaborn para visualización
- Jupyter para desarrollo interactivo