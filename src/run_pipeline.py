#!/usr/bin/env python
# coding: utf-8

"""
Script principal para ejecutar el pipeline completo de análisis de datos y modelado
para el proyecto de CS:GO.
Este script orquesta cada paso del proceso desde la limpieza de datos hasta la evaluación de modelos.
"""

import os
import sys
import time
import argparse
from datetime import datetime

# Importar módulos del proyecto
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import src.data_preprocessing as dp
import src.feature_engineering as fe
import src.regression_models as rm

def setup_directories():
    """
    Configura los directorios necesarios para almacenar datos y resultados
    
    Returns:
        bool: True si la configuración fue exitosa, False en caso contrario
    """
    try:
        # Directorios para datos
        os.makedirs('data/01_raw', exist_ok=True)
        os.makedirs('data/02_processed', exist_ok=True)
        os.makedirs('data/03_features', exist_ok=True)
        os.makedirs('data/04_models', exist_ok=True)
        os.makedirs('data/05_model_output', exist_ok=True)
        os.makedirs('data/06_models', exist_ok=True)
        
        # Directorios para reportes
        os.makedirs('reports/figures', exist_ok=True)
        os.makedirs('reports/model_comparison', exist_ok=True)
        
        print("✅ Directorios configurados correctamente")
        return True
    except Exception as e:
        print(f"Error al configurar directorios: {e}")
        return False

def run_data_preprocessing(input_path, force=False):
    """
    Ejecuta el preprocesamiento de datos
    
    Args:
        input_path (str): Ruta al archivo CSV con los datos crudos
        force (bool): Si es True, fuerza la ejecución aunque existan datos procesados
        
    Returns:
        bool: True si el procesamiento fue exitoso, False en caso contrario
    """
    output_path = 'data/02_processed/processed_data.csv'
    
    # Verificar si ya existen datos procesados
    if os.path.exists(output_path) and not force:
        print("🔄 Datos ya procesados. Usa --force para reprocesar")
        return True
    
    print("\n" + "="*80)
    print("🔍 Iniciando preprocesamiento de datos...")
    print("="*80)
    
    try:
        start_time = time.time()
        
        # Cargar datos
        df = dp.load_data(input_path)
        if df is None:
            return False
        
        # Limpiar datos
        df_clean = dp.clean_data(df)
        
        # Analizar calidad de datos
        quality_stats = dp.analyze_data_quality(df_clean)
        print("\nEstadísticas de calidad de datos:")
        print(quality_stats)
        
        # Detectar y manejar outliers
        numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
        df_clean = dp.handle_outliers(df_clean, numeric_cols, method='winsorize', limits=(0.05, 0.05))
        
        # Guardar datos procesados
        success = dp.save_processed_data(df_clean, output_path)
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Tiempo de ejecución: {elapsed_time:.2f} segundos")
        
        if success:
            print("✅ Preprocesamiento completado con éxito")
        return success
    
    except Exception as e:
        print(f"❌ Error en el preprocesamiento: {e}")
        return False

def run_feature_engineering(force=False):
    """
    Ejecuta la ingeniería de características
    
    Args:
        force (bool): Si es True, fuerza la ejecución aunque existan datos con características
        
    Returns:
        bool: True si el proceso fue exitoso, False en caso contrario
    """
    input_path = 'data/02_processed/processed_data.csv'
    output_path = 'data/03_features/engineered_data.csv'
    
    # Verificar si ya existen datos con ingeniería de características
    if os.path.exists(output_path) and not force:
        print("🔄 Datos con ingeniería de características ya existentes. Usa --force para reprocesar")
        return True
    
    # Verificar si existen datos procesados
    if not os.path.exists(input_path):
        print("❌ No se encontraron datos procesados. Ejecuta el preprocesamiento primero")
        return False
    
    print("\n" + "="*80)
    print("🛠️ Iniciando ingeniería de características...")
    print("="*80)
    
    try:
        start_time = time.time()
        
        # Cargar datos procesados
        df = fe.load_processed_data(input_path)
        if df is None:
            return False
        
        # Crear características básicas
        df_basic = fe.create_basic_features(df)
        print(f"DataFrame con características básicas: {df_basic.shape}")
        
        # Crear características avanzadas
        df_advanced = fe.create_advanced_features(df_basic)
        print(f"DataFrame con características avanzadas: {df_advanced.shape}")
        
        # Seleccionar características relevantes para RoundKills
        target = 'RoundKills'
        if target in df_advanced.columns:
            y = df_advanced[target]
            X = df_advanced.drop(columns=[target, 'MatchId', 'RoundId'] if 'MatchId' in df_advanced.columns else [target])
            
            # Seleccionar las mejores características mediante diferentes métodos
            selected_features_mi = fe.select_features(X, y, method='mutual_info', k=30)
            selected_features_f = fe.select_features(X, y, method='f_regression', k=30)
            selected_features_lasso = fe.select_features(X, y, method='lasso', k=30)
            
            # Conjunto de características como la unión de las tres listas
            unique_selected_features = list(set(selected_features_mi + selected_features_f + selected_features_lasso))
            
            # Visualizar importancia de características
            fe.plot_feature_importance(X, y, method='rf')
            
            # Crear un DataFrame final con las características seleccionadas y el target
            final_cols = unique_selected_features + [target]
            if 'MatchId' in df_advanced.columns:
                final_cols.append('MatchId')
            if 'RoundId' in df_advanced.columns:
                final_cols.append('RoundId')
            if 'Team' in df_advanced.columns:
                final_cols.append('Team')
                
            df_final = df_advanced[final_cols]
            
            # Guardar datos con ingeniería de características
            success = fe.save_engineered_data(df_final, output_path)
            
            elapsed_time = time.time() - start_time
            print(f"⏱️ Tiempo de ejecución: {elapsed_time:.2f} segundos")
            
            if success:
                print("✅ Ingeniería de características completada con éxito")
            return success
        else:
            print(f"❌ La columna {target} no existe en el DataFrame")
            return False
            
    except Exception as e:
        print(f"❌ Error en la ingeniería de características: {e}")
        return False

def run_regression_modeling(force=False):
    """
    Ejecuta el entrenamiento y evaluación de modelos de regresión
    
    Args:
        force (bool): Si es True, fuerza la ejecución aunque existan modelos
        
    Returns:
        bool: True si el proceso fue exitoso, False en caso contrario
    """
    input_path = 'data/03_features/engineered_data.csv'
    output_dir = 'data/06_models'
    
    # Verificar si ya existen modelos
    if os.path.exists(f"{output_dir}/RoundKills_prediction_VotingEnsemble.pkl") and not force:
        print("🔄 Modelos de regresión ya existentes. Usa --force para reentrenar")
        return True
    
    # Verificar si existen datos con ingeniería de características
    if not os.path.exists(input_path):
        print("❌ No se encontraron datos con ingeniería de características. Ejecuta ese paso primero")
        return False
    
    print("\n" + "="*80)
    print("📈 Iniciando entrenamiento y evaluación de modelos de regresión...")
    print("="*80)
    
    try:
        start_time = time.time()
        
        # Cargar datos con ingeniería de características
        df = rm.load_engineered_data(input_path)
        if df is None:
            return False
        
        # Preparar datos para modelado
        X_train, X_test, y_train, y_test, feature_names = rm.prepare_data_for_modeling(df)
        
        # Crear preprocesador
        preprocessor = rm.create_preprocessor(X_train)
        
        # Definir modelos
        models = rm.define_regression_models()
        
        # Evaluar modelos
        results, best_model, best_model_name = rm.evaluate_regression_models(
            X_train, X_test, y_train, y_test, models, preprocessor
        )
        
        # Comparar modelos
        rm.compare_regression_models(results)
        
        # Analizar residuos para el mejor modelo
        rm.analyze_residuals(y_test, results[best_model_name]['y_pred'], best_model_name)
        
        # Optimizar hiperparámetros para el mejor modelo
        best_model_optimized = rm.optimize_hyperparameters(X_train, y_train, best_model_name, preprocessor)
        
        # Evaluar modelo optimizado
        if best_model_optimized is not None:
            y_pred_optimized = best_model_optimized.predict(X_test)
            
            # Calcular métricas
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
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
            rm.analyze_residuals(y_test, y_pred_optimized, f"{best_model_name}_optimizado")
        
        # Crear modelos de ensamble
        voting_ensemble, stacking_ensemble = rm.create_ensemble_model(X_train, y_train, results, preprocessor)
        
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
        from numpy import sqrt
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
        model_path = rm.save_model(final_model, final_model_name)
        
        elapsed_time = time.time() - start_time
        print(f"⏱️ Tiempo de ejecución: {elapsed_time:.2f} segundos")
        
        if model_path:
            print("✅ Modelado de regresión completado con éxito")
            return True
        else:
            print("❌ Error al guardar el modelo final")
            return False
        
    except Exception as e:
        print(f"❌ Error en el modelado de regresión: {e}")
        return False

def run_full_pipeline(input_path, steps=None, force=False):
    """
    Ejecuta el pipeline completo o los pasos especificados
    
    Args:
        input_path (str): Ruta al archivo CSV con los datos crudos
        steps (list): Lista de pasos a ejecutar (si es None, ejecuta todos)
        force (bool): Si es True, fuerza la ejecución aunque existan resultados previos
    
    Returns:
        bool: True si el pipeline fue exitoso, False en caso contrario
    """
    # Configurar directorios
    if not setup_directories():
        return False
    
    # Definir pasos disponibles
    available_steps = {
        'preprocessing': run_data_preprocessing,
        'feature_engineering': run_feature_engineering,
        'regression': run_regression_modeling
    }
    
    # Si no se especifican pasos, ejecutar todos
    if steps is None:
        steps = list(available_steps.keys())
    
    # Validar pasos
    for step in steps:
        if step not in available_steps:
            print(f"❌ Paso no reconocido: {step}")
            print(f"Pasos disponibles: {list(available_steps.keys())}")
            return False
    
    # Ejecutar pasos seleccionados
    results = {}
    
    # Iniciar cronómetro general
    pipeline_start_time = time.time()
    
    for step in steps:
        print(f"\n🔄 Ejecutando paso: {step}")
        
        if step == 'preprocessing':
            results[step] = available_steps[step](input_path, force)
        else:
            results[step] = available_steps[step](force)
        
        if not results[step]:
            print(f"❌ Error en el paso: {step}")
            return False
    
    # Calcular tiempo total
    pipeline_elapsed_time = time.time() - pipeline_start_time
    
    print("\n" + "="*80)
    print("🎉 Pipeline completado con éxito")
    print(f"⏱️ Tiempo total de ejecución: {pipeline_elapsed_time:.2f} segundos")
    print("="*80)
    
    return True

def parse_arguments():
    """
    Parsea los argumentos de línea de comandos
    
    Returns:
        argparse.Namespace: Argumentos parseados
    """
    parser = argparse.ArgumentParser(
        description="Pipeline para análisis de datos y modelado de CS:GO",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "--input",
        default="data/01_raw/Anexo_ET_demo_round_traces_2022.csv",
        help="Ruta al archivo CSV con los datos crudos"
    )
    
    parser.add_argument(
        "--steps",
        nargs="+",
        choices=["preprocessing", "feature_engineering", "regression"],
        help="Pasos específicos a ejecutar (si no se especifica, ejecuta todos)"
    )
    
    parser.add_argument(
        "--force",
        action="store_true",
        help="Fuerza la ejecución aunque existan resultados previos"
    )
    
    return parser.parse_args()

if __name__ == "__main__":
    import numpy as np
    
    # Configurar formato de la hora para logs
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time}] 🚀 Iniciando pipeline")
    
    # Parsear argumentos
    args = parse_arguments()
    
    # Ejecutar pipeline
    success = run_full_pipeline(args.input, args.steps, args.force)
    
    if success:
        sys.exit(0)
    else:
        sys.exit(1) 