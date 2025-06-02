#!/usr/bin/env python
# coding: utf-8

"""
Módulo para implementación y evaluación de modelos de regresión para CS:GO
Este script implementa varios modelos de regresión optimizados para predecir RoundKills
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import time
import joblib
import os

from sklearn.model_selection import train_test_split, cross_val_score, KFold, GridSearchCV, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder, RobustScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, explained_variance_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor, SGDRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, StackingRegressor, VotingRegressor, AdaBoostRegressor
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
import xgboost as xgb
import lightgbm as lgb

# Configuración de visualización
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
sns.set_palette('viridis')

# Ignorar advertencias
import warnings
warnings.filterwarnings('ignore')

def load_engineered_data(path='data/03_features/engineered_data.csv'):
    """
    Carga los datos con ingeniería de características
    
    Args:
        path (str): Ruta al archivo CSV con datos preparados
        
    Returns:
        pandas.DataFrame: DataFrame con los datos cargados
    """
    try:
        df = pd.read_csv(path)
        print(f'Datos con ingeniería de características cargados: {df.shape}')
        return df
    except Exception as e:
        print(f"Error al cargar los datos con ingeniería de características: {e}")
        return None

def prepare_data_for_modeling(df, target='RoundKills', test_size=0.2, random_state=42):
    """
    Prepara los datos para el modelado separando características y target
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos
        target (str): Nombre de la variable objetivo
        test_size (float): Proporción de datos para el conjunto de prueba
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        tuple: X_train, X_test, y_train, y_test, feature_names
    """
    # Eliminar columnas de identificación o que podrían causar data leakage
    drop_cols = ['MatchId', 'RoundId']
    if 'RoundWinner' in df.columns:
        drop_cols.append('RoundWinner')
    if 'MatchWinner' in df.columns:
        drop_cols.append('MatchWinner')
    
    # Filtrar solo las columnas que existen en el DataFrame
    drop_cols = [col for col in drop_cols if col in df.columns]
    
    # Separar características y variable objetivo
    X = df.drop(columns=drop_cols + [target])
    y = df[target]
    
    # Dividir en conjuntos de entrenamiento y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    print(f"Dimensiones de los datos: X_train {X_train.shape}, X_test {X_test.shape}")
    
    # Mostrar estadísticas y distribución de la variable objetivo
    print(f"\nEstadísticas de {target}:")
    print(y_train.describe())
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(y_train, kde=True, bins=30)
    plt.title(f'Distribución de {target} en Entrenamiento', fontsize=14)
    plt.xlabel(target, fontsize=12)
    plt.ylabel('Frecuencia', fontsize=12)
    
    plt.subplot(1, 2, 2)
    sns.boxplot(y=y_train)
    plt.title(f'Boxplot de {target} en Entrenamiento', fontsize=14)
    plt.tight_layout()
    plt.savefig('reports/figures/target_distribution.png')
    plt.show()
    
    return X_train, X_test, y_train, y_test, X.columns.tolist()

def create_preprocessor(X):
    """
    Crea un preprocesador para las características
    
    Args:
        X (pandas.DataFrame): DataFrame con características
        
    Returns:
        ColumnTransformer: Preprocesador configurado
    """
    # Identificar columnas numéricas, categóricas y booleanas
    numeric_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
    categorical_cols = X.select_dtypes(include=['category', 'object']).columns.tolist()
    boolean_cols = X.select_dtypes(include=['bool']).columns.tolist()
    
    # Transformadores para diferentes tipos de datos
    numeric_transformer = Pipeline(steps=[
        ('scaler', RobustScaler())  # Más robusto a outliers que StandardScaler
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    boolean_transformer = Pipeline(steps=[
        ('passthrough', 'passthrough')
    ])
    
    # Combinar transformadores
    transformers = []
    if numeric_cols:
        transformers.append(('num', numeric_transformer, numeric_cols))
    if categorical_cols:
        transformers.append(('cat', categorical_transformer, categorical_cols))
    if boolean_cols:
        transformers.append(('bool', boolean_transformer, boolean_cols))
    
    preprocessor = ColumnTransformer(transformers=transformers)
    
    return preprocessor

def define_regression_models(random_state=42):
    """
    Define una variedad de modelos de regresión para evaluar
    
    Args:
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        dict: Diccionario con modelos de regresión
    """
    models = {
        'LinearRegression': Pipeline(steps=[
            ('regressor', LinearRegression())
        ]),
        'Ridge': Pipeline(steps=[
            ('regressor', Ridge(alpha=1.0, random_state=random_state))
        ]),
        'Lasso': Pipeline(steps=[
            ('regressor', Lasso(alpha=0.1, random_state=random_state))
        ]),
        'ElasticNet': Pipeline(steps=[
            ('regressor', ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=random_state))
        ]),
        'HuberRegressor': Pipeline(steps=[
            ('regressor', HuberRegressor(epsilon=1.35, max_iter=200))
        ]),
        'SVR': Pipeline(steps=[
            ('regressor', SVR(kernel='rbf', C=1.0, epsilon=0.1))
        ]),
        'KNeighborsRegressor': Pipeline(steps=[
            ('regressor', KNeighborsRegressor(n_neighbors=5))
        ]),
        'DecisionTreeRegressor': Pipeline(steps=[
            ('regressor', DecisionTreeRegressor(random_state=random_state))
        ]),
        'RandomForestRegressor': Pipeline(steps=[
            ('regressor', RandomForestRegressor(n_estimators=100, random_state=random_state))
        ]),
        'GradientBoostingRegressor': Pipeline(steps=[
            ('regressor', GradientBoostingRegressor(n_estimators=100, random_state=random_state))
        ]),
        'AdaBoostRegressor': Pipeline(steps=[
            ('regressor', AdaBoostRegressor(random_state=random_state))
        ]),
        'XGBRegressor': Pipeline(steps=[
            ('regressor', xgb.XGBRegressor(n_estimators=100, random_state=random_state))
        ]),
        'LGBMRegressor': Pipeline(steps=[
            ('regressor', lgb.LGBMRegressor(n_estimators=100, random_state=random_state))
        ])
    }
    
    return models

def evaluate_regression_models(X_train, X_test, y_train, y_test, models, preprocessor):
    """
    Evalúa múltiples modelos de regresión
    
    Args:
        X_train (pandas.DataFrame): Características de entrenamiento
        X_test (pandas.DataFrame): Características de prueba
        y_train (pandas.Series): Variable objetivo de entrenamiento
        y_test (pandas.Series): Variable objetivo de prueba
        models (dict): Diccionario con modelos de regresión
        preprocessor (ColumnTransformer): Preprocesador para las características
        
    Returns:
        tuple: Resultados, mejor modelo y nombre del mejor modelo
    """
    results = {}
    
    # Iterar sobre cada modelo
    for model_name, model_pipeline in models.items():
        start_time = time.time()
        print(f"\nEntrenando {model_name}...")
        
        # Crear pipeline completo con preprocesador y modelo
        pipeline = Pipeline(steps=[
            ('preprocessor', preprocessor),
            ('model', model_pipeline)
        ])
        
        # Entrenar el modelo con validación cruzada para evaluar estabilidad
        cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring='r2')
        print(f"  Validación cruzada (R²): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        
        # Entrenar modelo en todo el conjunto de entrenamiento
        pipeline.fit(X_train, y_train)
        
        # Predecir en conjunto de prueba
        y_pred = pipeline.predict(X_test)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        explained_variance = explained_variance_score(y_test, y_pred)
        
        # Tiempo de entrenamiento
        training_time = time.time() - start_time
        
        # Almacenar resultados
        results[model_name] = {
            'model': pipeline,
            'y_pred': y_pred,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'explained_variance': explained_variance,
            'cv_scores': cv_scores,
            'training_time': training_time
        }
        
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  Varianza explicada: {explained_variance:.4f}")
        print(f"  Tiempo de entrenamiento: {training_time:.2f} segundos")
    
    # Determinar el mejor modelo basado en R²
    best_model_name = max(results, key=lambda x: results[x]['r2'])
    print(f"\nMejor modelo: {best_model_name} (R² = {results[best_model_name]['r2']:.4f})")
    
    return results, results[best_model_name]['model'], best_model_name

def compare_regression_models(results):
    """
    Compara los resultados de los modelos de regresión con gráficos
    
    Args:
        results (dict): Diccionario con resultados de los modelos
    """
    # Extraer métricas para comparación
    model_names = list(results.keys())
    r2_scores = [results[model]['r2'] for model in model_names]
    rmse_scores = [results[model]['rmse'] for model in model_names]
    mae_scores = [results[model]['mae'] for model in model_names]
    training_times = [results[model]['training_time'] for model in model_names]
    
    # Crear DataFrame para facilitar la visualización
    comparison_df = pd.DataFrame({
        'Modelo': model_names,
        'R²': r2_scores,
        'RMSE': rmse_scores,
        'MAE': mae_scores,
        'Tiempo de entrenamiento (s)': training_times
    })
    
    # Ordenar por R² descendente
    comparison_df = comparison_df.sort_values('R²', ascending=False).reset_index(drop=True)
    
    # Mostrar tabla comparativa
    print("\nComparación de modelos:")
    print(comparison_df)
    
    # Guardar la comparación en un archivo CSV
    os.makedirs('reports/model_comparison', exist_ok=True)
    comparison_df.to_csv('reports/model_comparison/regression_models_comparison.csv', index=False)
    
    # Visualizaciones
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # R² Score
    sns.barplot(x='R²', y='Modelo', data=comparison_df, ax=axes[0, 0])
    axes[0, 0].set_title('Coeficiente de Determinación (R²)', fontsize=14)
    axes[0, 0].set_xlim(0, 1)
    
    # RMSE
    sns.barplot(x='RMSE', y='Modelo', data=comparison_df, ax=axes[0, 1])
    axes[0, 1].set_title('Root Mean Squared Error (RMSE)', fontsize=14)
    
    # MAE
    sns.barplot(x='MAE', y='Modelo', data=comparison_df, ax=axes[1, 0])
    axes[1, 0].set_title('Mean Absolute Error (MAE)', fontsize=14)
    
    # Tiempo de entrenamiento
    sns.barplot(x='Tiempo de entrenamiento (s)', y='Modelo', data=comparison_df, ax=axes[1, 1])
    axes[1, 1].set_title('Tiempo de entrenamiento (segundos)', fontsize=14)
    
    plt.tight_layout()
    plt.savefig('reports/figures/regression_models_comparison.png')
    plt.show()

def analyze_residuals(y_test, y_pred, model_name):
    """
    Analiza los residuos del modelo de regresión
    
    Args:
        y_test (pandas.Series): Valores reales
        y_pred (numpy.ndarray): Predicciones
        model_name (str): Nombre del modelo
    """
    # Calcular residuos
    residuals = y_test - y_pred
    
    # Crear gráficos de análisis de residuos
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Distribución de residuos
    sns.histplot(residuals, kde=True, ax=axes[0, 0])
    axes[0, 0].axvline(x=0, color='red', linestyle='--')
    axes[0, 0].set_title('Distribución de Residuos', fontsize=14)
    axes[0, 0].set_xlabel('Residuos', fontsize=12)
    axes[0, 0].set_ylabel('Frecuencia', fontsize=12)
    
    # Residuos vs Predicciones
    sns.scatterplot(x=y_pred, y=residuals, ax=axes[0, 1])
    axes[0, 1].axhline(y=0, color='red', linestyle='--')
    axes[0, 1].set_title('Residuos vs Predicciones', fontsize=14)
    axes[0, 1].set_xlabel('Predicciones', fontsize=12)
    axes[0, 1].set_ylabel('Residuos', fontsize=12)
    
    # QQ-plot de residuos
    from scipy import stats
    stats.probplot(residuals, dist="norm", plot=axes[1, 0])
    axes[1, 0].set_title('Q-Q Plot de Residuos', fontsize=14)
    
    # Predicciones vs Valores reales
    sns.scatterplot(x=y_test, y=y_pred, ax=axes[1, 1])
    axes[1, 1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--')
    axes[1, 1].set_title('Valores Reales vs Predicciones', fontsize=14)
    axes[1, 1].set_xlabel('Valores Reales', fontsize=12)
    axes[1, 1].set_ylabel('Predicciones', fontsize=12)
    
    plt.suptitle(f'Análisis de Residuos para {model_name}', fontsize=16)
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(f'reports/figures/residual_analysis_{model_name}.png')
    plt.show()
    
    # Calcular estadísticas de residuos
    print("\nEstadísticas de Residuos:")
    print(pd.Series(residuals).describe())
    
    # Prueba de normalidad de residuos
    stat, p_value = stats.shapiro(residuals)
    print(f"Prueba de normalidad Shapiro-Wilk: estadístico={stat:.4f}, p-valor={p_value:.4f}")
    if p_value < 0.05:
        print("Los residuos no siguen una distribución normal (p < 0.05)")
    else:
        print("Los residuos siguen una distribución normal (p >= 0.05)")

def optimize_hyperparameters(X_train, y_train, best_model_name, preprocessor, random_state=42):
    """
    Optimiza los hiperparámetros del mejor modelo
    
    Args:
        X_train (pandas.DataFrame): Características de entrenamiento
        y_train (pandas.Series): Variable objetivo de entrenamiento
        best_model_name (str): Nombre del mejor modelo
        preprocessor (ColumnTransformer): Preprocesador para las características
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        Pipeline: Modelo con los mejores hiperparámetros
    """
    print(f"\nOptimizando hiperparámetros para {best_model_name}...")
    
    # Definir grillas de búsqueda según el tipo de modelo
    if best_model_name == 'LinearRegression':
        # LinearRegression tiene pocos hiperparámetros para ajustar
        param_grid = {
            'model__regressor__fit_intercept': [True, False],
            'model__regressor__copy_X': [True, False],
            'model__regressor__positive': [True, False]
        }
        model = LinearRegression()
        
    elif best_model_name == 'Ridge':
        param_grid = {
            'model__regressor__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
            'model__regressor__fit_intercept': [True, False],
            'model__regressor__solver': ['auto', 'svd', 'cholesky', 'lsqr', 'sparse_cg', 'sag', 'saga']
        }
        model = Ridge(random_state=random_state)
        
    elif best_model_name == 'Lasso':
        param_grid = {
            'model__regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'model__regressor__fit_intercept': [True, False],
            'model__regressor__selection': ['cyclic', 'random']
        }
        model = Lasso(random_state=random_state)
        
    elif best_model_name == 'ElasticNet':
        param_grid = {
            'model__regressor__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
            'model__regressor__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
            'model__regressor__fit_intercept': [True, False],
            'model__regressor__selection': ['cyclic', 'random']
        }
        model = ElasticNet(random_state=random_state)
        
    elif best_model_name == 'HuberRegressor':
        param_grid = {
            'model__regressor__epsilon': [1.1, 1.35, 1.5, 2.0],
            'model__regressor__alpha': [0.0001, 0.001, 0.01],
            'model__regressor__fit_intercept': [True, False],
            'model__regressor__max_iter': [100, 200, 500]
        }
        model = HuberRegressor()
        
    elif best_model_name == 'SVR':
        param_grid = {
            'model__regressor__C': [0.1, 1, 10, 100],
            'model__regressor__kernel': ['linear', 'poly', 'rbf', 'sigmoid'],
            'model__regressor__gamma': ['scale', 'auto', 0.1, 1],
            'model__regressor__epsilon': [0.01, 0.1, 0.2]
        }
        model = SVR()
        
    elif best_model_name == 'KNeighborsRegressor':
        param_grid = {
            'model__regressor__n_neighbors': [3, 5, 7, 9, 11],
            'model__regressor__weights': ['uniform', 'distance'],
            'model__regressor__algorithm': ['auto', 'ball_tree', 'kd_tree', 'brute'],
            'model__regressor__p': [1, 2]
        }
        model = KNeighborsRegressor()
        
    elif best_model_name == 'DecisionTreeRegressor':
        param_grid = {
            'model__regressor__max_depth': [None, 5, 10, 15],
            'model__regressor__min_samples_split': [2, 5, 10],
            'model__regressor__min_samples_leaf': [1, 2, 4],
            'model__regressor__criterion': ['squared_error', 'friedman_mse', 'absolute_error', 'poisson']
        }
        model = DecisionTreeRegressor(random_state=random_state)
        
    elif best_model_name == 'RandomForestRegressor':
        param_grid = {
            'model__regressor__n_estimators': [50, 100, 200],
            'model__regressor__max_depth': [None, 10, 20, 30],
            'model__regressor__min_samples_split': [2, 5, 10],
            'model__regressor__min_samples_leaf': [1, 2, 4],
            'model__regressor__bootstrap': [True, False]
        }
        model = RandomForestRegressor(random_state=random_state)
        
    elif best_model_name == 'GradientBoostingRegressor':
        param_grid = {
            'model__regressor__n_estimators': [50, 100, 200],
            'model__regressor__learning_rate': [0.01, 0.05, 0.1],
            'model__regressor__max_depth': [3, 5, 7],
            'model__regressor__min_samples_split': [2, 5],
            'model__regressor__min_samples_leaf': [1, 2]
        }
        model = GradientBoostingRegressor(random_state=random_state)
        
    elif best_model_name == 'AdaBoostRegressor':
        param_grid = {
            'model__regressor__n_estimators': [50, 100, 200],
            'model__regressor__learning_rate': [0.01, 0.1, 1.0],
            'model__regressor__loss': ['linear', 'square', 'exponential']
        }
        model = AdaBoostRegressor(random_state=random_state)
        
    elif best_model_name == 'XGBRegressor':
        param_grid = {
            'model__regressor__n_estimators': [50, 100, 200],
            'model__regressor__learning_rate': [0.01, 0.1, 0.3],
            'model__regressor__max_depth': [3, 5, 7],
            'model__regressor__subsample': [0.7, 0.8, 0.9],
            'model__regressor__colsample_bytree': [0.7, 0.8, 0.9],
            'model__regressor__min_child_weight': [1, 3, 5]
        }
        model = xgb.XGBRegressor(random_state=random_state)
        
    elif best_model_name == 'LGBMRegressor':
        param_grid = {
            'model__regressor__n_estimators': [50, 100, 200],
            'model__regressor__learning_rate': [0.01, 0.1, 0.3],
            'model__regressor__max_depth': [3, 5, 7],
            'model__regressor__num_leaves': [31, 50, 70],
            'model__regressor__subsample': [0.7, 0.8, 0.9],
            'model__regressor__colsample_bytree': [0.7, 0.8, 0.9]
        }
        model = lgb.LGBMRegressor(random_state=random_state)
    
    else:
        print(f"No se han definido hiperparámetros para {best_model_name}")
        return None
    
    # Crear pipeline con preprocesador y modelo
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', Pipeline(steps=[('regressor', model)]))
    ])
    
    # Usar RandomizedSearchCV en lugar de GridSearchCV para modelos con muchos hiperparámetros
    n_iter = 20  # Número de combinaciones a probar
    
    if len(param_grid) <= 5:
        # Para pocos hiperparámetros, usar GridSearchCV
        search = GridSearchCV(
            pipeline,
            param_grid,
            cv=5,
            scoring='r2',
            verbose=1,
            n_jobs=-1
        )
    else:
        # Para muchos hiperparámetros, usar RandomizedSearchCV
        search = RandomizedSearchCV(
            pipeline,
            param_grid,
            n_iter=n_iter,
            cv=5,
            scoring='r2',
            verbose=1,
            random_state=random_state,
            n_jobs=-1
        )
    
    # Entrenar el modelo con búsqueda de hiperparámetros
    search.fit(X_train, y_train)
    
    # Mostrar los mejores hiperparámetros
    print("\nMejores hiperparámetros:")
    for param, value in search.best_params_.items():
        print(f"  {param}: {value}")
    
    print(f"Mejor puntuación R²: {search.best_score_:.4f}")
    
    return search.best_estimator_

def create_ensemble_model(X_train, y_train, results, preprocessor, random_state=42):
    """
    Crea un modelo de ensamble combinando los mejores modelos
    
    Args:
        X_train (pandas.DataFrame): Características de entrenamiento
        y_train (pandas.Series): Variable objetivo de entrenamiento
        results (dict): Diccionario con resultados de los modelos
        preprocessor (ColumnTransformer): Preprocesador para las características
        random_state (int): Semilla para reproducibilidad
        
    Returns:
        tuple: Modelo de ensamble Voting y Stacking
    """
    print("\nCreando modelos de ensamble...")
    
    # Seleccionar los 5 mejores modelos
    comparison_df = pd.DataFrame({
        'Modelo': list(results.keys()),
        'R2': [results[model]['r2'] for model in results.keys()]
    })
    top_models = comparison_df.sort_values('R2', ascending=False).head(5)['Modelo'].tolist()
    
    print(f"Modelos seleccionados para el ensamble: {top_models}")
    
    # Crear estimadores para los modelos de ensamble
    estimators = []
    for model_name in top_models:
        model = results[model_name]['model'].named_steps['model']
        estimators.append((model_name, model))
    
    # Crear modelo de ensamble Voting
    voting_ensemble = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', VotingRegressor(estimators=estimators))
    ])
    
    # Entrenar modelo Voting
    print("Entrenando modelo de ensamble Voting...")
    voting_ensemble.fit(X_train, y_train)
    
    # Crear modelo de ensamble Stacking
    stacking_ensemble = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', StackingRegressor(
            estimators=estimators,
            final_estimator=Ridge(random_state=random_state),
            cv=5,
            n_jobs=-1
        ))
    ])
    
    # Entrenar modelo Stacking
    print("Entrenando modelo de ensamble Stacking...")
    stacking_ensemble.fit(X_train, y_train)
    
    return voting_ensemble, stacking_ensemble

def save_model(model, model_name, output_dir='data/06_models'):
    """
    Guarda el modelo entrenado
    
    Args:
        model (Pipeline): Modelo entrenado
        model_name (str): Nombre del modelo
        output_dir (str): Directorio de salida
        
    Returns:
        str: Ruta donde se guardó el modelo
    """
    try:
        # Asegurar que el directorio existe
        os.makedirs(output_dir, exist_ok=True)
        
        # Guardar el modelo
        model_path = f"{output_dir}/RoundKills_prediction_{model_name}.pkl"
        joblib.dump(model, model_path)
        print(f"Modelo guardado en: {model_path}")
        return model_path
    except Exception as e:
        print(f"Error al guardar el modelo: {e}")
        return None

def main():
    """Función principal para ejecutar el entrenamiento y evaluación de modelos de regresión"""
    # Cargar datos con ingeniería de características
    df = load_engineered_data()
    if df is None:
        return
    
    # Preparar datos para modelado
    X_train, X_test, y_train, y_test, feature_names = prepare_data_for_modeling(df)
    
    # Crear preprocesador
    preprocessor = create_preprocessor(X_train)
    
    # Definir modelos
    models = define_regression_models()
    
    # Evaluar modelos
    results, best_model, best_model_name = evaluate_regression_models(
        X_train, X_test, y_train, y_test, models, preprocessor
    )
    
    # Comparar modelos
    compare_regression_models(results)
    
    # Analizar residuos para el mejor modelo
    analyze_residuals(y_test, results[best_model_name]['y_pred'], best_model_name)
    
    # Optimizar hiperparámetros para el mejor modelo
    best_model_optimized = optimize_hyperparameters(X_train, y_train, best_model_name, preprocessor)
    
    # Evaluar modelo optimizado
    if best_model_optimized is not None:
        y_pred_optimized = best_model_optimized.predict(X_test)
        
        # Calcular métricas
        mse = mean_squared_error(y_test, y_pred_optimized)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred_optimized)
        r2 = r2_score(y_test, y_pred_optimized)
        
        print("\nRendimiento del modelo optimizado:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        
        # Actualizar el mejor modelo
        best_model = best_model_optimized
        
        # Analizar residuos del modelo optimizado
        analyze_residuals(y_test, y_pred_optimized, f"{best_model_name}_optimizado")
    
    # Crear modelos de ensamble
    voting_ensemble, stacking_ensemble = create_ensemble_model(X_train, y_train, results, preprocessor)
    
    # Evaluar modelos de ensamble
    y_pred_voting = voting_ensemble.predict(X_test)
    y_pred_stacking = stacking_ensemble.predict(X_test)
    
    # Calcular métricas para Voting
    mse_voting = mean_squared_error(y_test, y_pred_voting)
    rmse_voting = np.sqrt(mse_voting)
    mae_voting = mean_absolute_error(y_test, y_pred_voting)
    r2_voting = r2_score(y_test, y_pred_voting)
    
    # Calcular métricas para Stacking
    mse_stacking = mean_squared_error(y_test, y_pred_stacking)
    rmse_stacking = np.sqrt(mse_stacking)
    mae_stacking = mean_absolute_error(y_test, y_pred_stacking)
    r2_stacking = r2_score(y_test, y_pred_stacking)
    
    print("\nRendimiento del modelo de ensamble Voting:")
    print(f"  MSE: {mse_voting:.4f}")
    print(f"  RMSE: {rmse_voting:.4f}")
    print(f"  MAE: {mae_voting:.4f}")
    print(f"  R²: {r2_voting:.4f}")
    
    print("\nRendimiento del modelo de ensamble Stacking:")
    print(f"  MSE: {mse_stacking:.4f}")
    print(f"  RMSE: {rmse_stacking:.4f}")
    print(f"  MAE: {mae_stacking:.4f}")
    print(f"  R²: {r2_stacking:.4f}")
    
    # Determinar si algún modelo de ensamble supera al mejor modelo original
    best_r2 = max(r2, r2_voting, r2_stacking)
    if best_r2 == r2_voting:
        final_model = voting_ensemble
        final_model_name = "VotingEnsemble"
    elif best_r2 == r2_stacking:
        final_model = stacking_ensemble
        final_model_name = "StackingEnsemble"
    else:
        final_model = best_model
        final_model_name = best_model_name
    
    print(f"\nMejor modelo general: {final_model_name} (R² = {best_r2:.4f})")
    
    # Guardar el mejor modelo
    save_model(final_model, final_model_name)
    
if __name__ == "__main__":
    main() 