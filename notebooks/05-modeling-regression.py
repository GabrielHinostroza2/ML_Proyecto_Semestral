import sys
import os
import contextlib
import warnings
import logging
import joblib
import time
from datetime import datetime
from typing import Dict, Any, Tuple, Optional
import json

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression, ElasticNet, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import RobustScaler, StandardScaler
from sklearn.model_selection import RandomizedSearchCV, cross_val_score, StratifiedKFold, train_test_split
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, explained_variance_score
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_regression

import xgboost as xgb


def load_data(data_path: Optional[str] = None) -> pd.DataFrame:
    """
    Carga los datos desde el archivo CSV con detección automática de separador
    
    Args:
        data_path: Ruta al archivo de datos. Si es None, usa la ruta por defecto.
    
    Returns:
        DataFrame con los datos cargados
    """
    if data_path is None:
        # Ruta por defecto
        data_path = r'C:\Users\LuisSalamanca\Desktop\Duoc\Machine\ML_Proyecto_Semestral\data\03_features\engineered_data.csv'
    
    def try_load_csv(filepath, separator):
        """Intenta cargar CSV con un separador específico"""
        try:
            data = pd.read_csv(filepath, sep=separator)
            return data
        except Exception as e:
            logger.debug(f"Error con separador '{separator}': {e}")
            return None
    
    def detect_and_fix_columns(data):
        """Detecta si las columnas están concatenadas y las separa"""
        if len(data.columns) == 1:
            # Si solo hay una columna, probablemente esté mal separada
            column_name = data.columns[0]
            if ',' in column_name:
                logger.info("Detectadas columnas concatenadas con ','. Separando...")
                # Dividir el nombre de la columna
                new_columns = column_name.split(',')
                
                # Dividir los datos de cada fila
                data_split = data[column_name].str.split(',', expand=True)
                data_split.columns = new_columns
                
                # Convertir a tipos numéricos donde sea posible
                for col in data_split.columns:
                    try:
                        data_split[col] = pd.to_numeric(data_split[col])
                    except ValueError:
                        logger.debug(f"Columna '{col}' no se pudo convertir a numérico")
                
                return data_split
        return data
    
    try:
        logger.info(f"Intentando cargar datos desde: {data_path}")
        
        # Lista de separadores a probar
        separators = [';', ',', '\t', '|']
        data = None
        
        for sep in separators:
            logger.debug(f"Probando separador: '{sep}'")
            data = try_load_csv(data_path, sep)
            
            if data is not None:
                logger.info(f"Datos cargados exitosamente con separador '{sep}'")
                break
        
        if data is None:
            raise ValueError("No se pudo cargar el archivo con ningún separador estándar")
        
        # Detectar y corregir columnas concatenadas
        data = detect_and_fix_columns(data)
        
        logger.info(f"Forma de los datos: {data.shape}")
        logger.info(f"Columnas: {list(data.columns)}")
        
        # Verificar si hay valores nulos
        null_counts = data.isnull().sum()
        if null_counts.sum() > 0:
            logger.warning(f"Se encontraron valores nulos:\n{null_counts[null_counts > 0]}")
        
        # Mostrar estadísticas básicas
        logger.info("Primeras filas del dataset:")
        logger.info(f"\n{data.head().to_string()}")
        
        return data
        
    except FileNotFoundError:
        logger.error(f"No se pudo encontrar el archivo: {data_path}")
        raise
    except Exception as e:
        logger.error(f"Error al cargar los datos: {str(e)}")
        raise


def validate_required_columns(data: pd.DataFrame) -> bool:
    """
    Valida que el dataset tenga las columnas requeridas
    
    Args:
        data: DataFrame a validar
    
    Returns:
        bool: True si tiene todas las columnas requeridas
    """
    required_features = [
        'EconomicEfficiency',
        'EffectivenessScore',
        'EquipmentAdvantage',
        'KillAssistRatio',
        'StealthKillsRatio'
    ]
    required_target = 'KDA'
    
    all_required = required_features + [required_target]
    available_columns = list(data.columns)
    
    missing_columns = [col for col in all_required if col not in available_columns]
    
    if missing_columns:
        logger.error(f"Columnas faltantes: {missing_columns}")
        logger.info(f"Columnas disponibles: {available_columns}")
        logger.info("Sugerencias:")
        logger.info("1. Verifica que el archivo CSV tenga las columnas correctas")
        logger.info("2. Revisa si las columnas tienen nombres diferentes")
        logger.info("3. Asegúrate de que el separador del CSV sea correcto")
        return False
    
    logger.info("✅ Todas las columnas requeridas están presentes")
    return True


def split_data(data: pd.DataFrame, test_size: float = 0.2, random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide los datos en conjuntos de entrenamiento y prueba
    
    Args:
        data: DataFrame con los datos
        test_size: Proporción de datos para prueba
        random_state: Semilla para reproducibilidad
    
    Returns:
        Tupla con X_train, X_test, y_train, y_test
    """
    # Validar columnas requeridas
    if not validate_required_columns(data):
        # Si no están las columnas exactas, intentar encontrar columnas similares
        logger.warning("Intentando identificar columnas automáticamente...")
        
        available_cols = list(data.columns)
        logger.info(f"Columnas disponibles: {available_cols}")
        
        # Si solo tenemos una columna con todos los datos concatenados, mostrar error específico
        if len(available_cols) == 1 and (',' in available_cols[0] or 'Efficiency' in available_cols[0]):
            logger.error("❌ Parece que los datos no se separaron correctamente.")
            logger.error("El archivo CSV probablemente tiene un formato diferente al esperado.")
            logger.error("Por favor, verifica que:")
            logger.error("1. El archivo tenga columnas separadas por ';' o ','")
            logger.error("2. Las columnas tengan los nombres exactos requeridos")
            logger.error("3. El archivo no esté corrupto")
        
        raise ValueError("No se pueden encontrar las columnas requeridas en el dataset")
    
    # Características específicas del proyecto
    features = [
        'EconomicEfficiency',
        'EffectivenessScore',
        'EquipmentAdvantage',
        'KillAssistRatio',
        'StealthKillsRatio'
    ]
    target = 'KDA'
    
    # Extraer características y objetivo
    X = data[features].copy()
    y = data[target].copy()
    
    logger.info(f"Características seleccionadas: {features}")
    logger.info(f"Variable objetivo: {target}")
    logger.info(f"Forma de X: {X.shape}")
    logger.info(f"Forma de y: {y.shape}")
    
    # Verificar valores nulos en las características seleccionadas
    X_nulls = X.isnull().sum()
    y_nulls = y.isnull().sum()
    
    if X_nulls.sum() > 0:
        logger.warning(f"Valores nulos en características:\n{X_nulls[X_nulls > 0]}")
        # Opcional: llenar valores nulos con la mediana
        X = X.fillna(X.median())
        logger.info("Valores nulos en X llenados con la mediana")
    
    if y_nulls > 0:
        logger.warning(f"Valores nulos en variable objetivo: {y_nulls}")
        # Eliminar filas con valores nulos en y
        mask = ~y.isnull()
        X = X[mask]
        y = y[mask]
        logger.info(f"Eliminadas {(~mask).sum()} filas con valores nulos en y")
    
    # Dividir los datos
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=None  # Para regresión no se usa stratify
    )
    
    logger.info(f"División de datos completada:")
    logger.info(f"  - Entrenamiento: X{X_train.shape}, y{y_train.shape}")
    logger.info(f"  - Prueba: X{X_test.shape}, y{y_test.shape}")
    logger.info(f"  - Proporción de prueba: {test_size}")
    
    # Estadísticas básicas
    logger.info(f"Estadísticas de la variable objetivo:")
    logger.info(f"  - Media entrenamiento: {y_train.mean():.4f}")
    logger.info(f"  - Media prueba: {y_test.mean():.4f}")
    logger.info(f"  - Std entrenamiento: {y_train.std():.4f}")
    logger.info(f"  - Std prueba: {y_test.std():.4f}")
    
    return X_train, X_test, y_train, y_test


def diagnosticar_archivo_csv(data_path: str) -> None:
    """
    Diagnostica problemas comunes con archivos CSV
    
    Args:
        data_path: Ruta al archivo CSV
    """
    logger.info("🔍 DIAGNÓSTICO DEL ARCHIVO CSV")
    logger.info("="*50)
    
    try:
        # Leer las primeras líneas como texto plano
        with open(data_path, 'r', encoding='utf-8') as f:
            first_lines = [f.readline().strip() for _ in range(3)]
        
        logger.info("📄 Primeras 3 líneas del archivo:")
        for i, line in enumerate(first_lines, 1):
            logger.info(f"  Línea {i}: {line[:100]}{'...' if len(line) > 100 else ''}")
        
        # Analizar separadores posibles
        separators = [',', ';', '\t', '|']
        separator_counts = {}
        
        for sep in separators:
            count = first_lines[0].count(sep) if first_lines else 0
            separator_counts[sep] = count
            logger.info(f"  Separador '{sep}': {count} ocurrencias")
        
        # Sugerir el mejor separador
        best_sep = max(separator_counts, key=separator_counts.get)
        if separator_counts[best_sep] > 0:
            logger.info(f"💡 Separador sugerido: '{best_sep}' ({separator_counts[best_sep]} columnas)")
        else:
            logger.warning("⚠️  No se detectaron separadores comunes")
        
    except Exception as e:
        logger.error(f"❌ Error al leer el archivo: {e}")


def load_and_split_data(data_path: Optional[str] = None, test_size: float = 0.2, 
                       random_state: int = 42) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Función combinada para cargar y dividir los datos en un solo paso
    
    Args:
        data_path: Ruta al archivo de datos
        test_size: Proporción de datos para prueba
        random_state: Semilla para reproducibilidad
    
    Returns:
        Tupla con X_train, X_test, y_train, y_test
    """
    logger.info("=== CARGA Y DIVISIÓN DE DATOS ===")
    
    try:
        # Cargar datos
        data = load_data(data_path)
        
        # Dividir datos
        X_train, X_test, y_train, y_test = split_data(data, test_size, random_state)
        
        logger.info("=== DATOS LISTOS PARA ENTRENAMIENTO ===")
        
        return X_train, X_test, y_train, y_test
        
    except Exception as e:
        logger.error(f"❌ Error en carga y división de datos: {e}")
        
        # Realizar diagnóstico si hay problemas
        if data_path is None:
            data_path = r'C:\Users\LuisSalamanca\Desktop\Duoc\Machine\ML_Proyecto_Semestral\data\03_features\engineered_data.csv'
        
        logger.info("\n🔧 Ejecutando diagnóstico automático...")
        diagnosticar_archivo_csv(data_path)
        
        logger.info("\n💡 POSIBLES SOLUCIONES:")
        logger.info("1. Verifica que el archivo CSV existe en la ruta especificada")
        logger.info("2. Asegúrate de que las columnas están separadas correctamente")
        logger.info("3. Revisa que los nombres de las columnas sean exactamente:")
        logger.info("   - EconomicEfficiency")
        logger.info("   - EffectivenessScore") 
        logger.info("   - EquipmentAdvantage")
        logger.info("   - KillAssistRatio")
        logger.info("   - StealthKillsRatio")
        logger.info("   - KDA")
        logger.info("4. Intenta abrir el archivo en Excel/LibreOffice para verificar su estructura")
        
        raise

# Configurar logging sin caracteres especiales
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('model_training.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Clase para evaluar múltiples modelos de machine learning de forma eficiente"""
    
    def __init__(self, random_state: int = 42, n_jobs: int = -1, cv_folds: int = 5, 
                 fast_mode: bool = False, max_samples_svr: int = 10000):
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.cv_folds = cv_folds
        self.results = {}
        self.best_model_info = None
        self.fast_mode = fast_mode
        self.max_samples_svr = max_samples_svr
        
        # Verificar disponibilidad de GPU
        self.gpu_available = self._check_gpu_availability()
        
        # Configurar modelos con parámetros optimizados
        self.models = self._configure_models()
    
    def _check_gpu_availability(self) -> bool:
        """Verifica si GPU está disponible para XGBoost y LightGBM"""
        try:
            # Verificar XGBoost GPU
            xgb_test = xgb.XGBRegressor(tree_method='gpu_hist', gpu_id=0, n_estimators=1)
            xgb_test.fit(np.random.random((10, 5)), np.random.random(10))
            logger.info("GPU disponible para entrenamiento")
            return True
        except Exception as e:
            logger.warning(f"GPU no disponible, usando CPU: {e}")
            return False
    
    def _configure_models(self) -> Dict[str, Dict[str, Any]]:
        """Configura los modelos con sus hiperparámetros optimizados"""
        
        # Modelos base
        models = {
            'LinearRegression': {
                'model': LinearRegression(),
                'params': {},
                'feature_selection': False,
                'fast': True
            },
            'Ridge': {
                'model': Ridge(random_state=self.random_state),
                'params': {
                    'model__alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0],
                    'model__solver': ['auto', 'svd', 'cholesky']
                },
                'feature_selection': True,
                'fast': True
            },
            'ElasticNet': {
                'model': ElasticNet(random_state=self.random_state, max_iter=2000),
                'params': {
                    'model__alpha': [0.0001, 0.001, 0.01, 0.1, 1.0],
                    'model__l1_ratio': [0.1, 0.3, 0.5, 0.7, 0.9],
                    'model__selection': ['cyclic', 'random']
                },
                'feature_selection': True,
                'fast': True
            }
        }
        
        # SVR optimizado para datasets grandes
        if not self.fast_mode:
            models['SVR'] = {
                'model': SVR(),
                'params': {
                    'model__C': [0.1, 1, 10],  # Reducido para mayor velocidad
                    'model__gamma': ['scale', 'auto'],  # Reducido
                    'model__kernel': ['rbf', 'linear'],  # Removido 'poly' que es muy lento
                    'model__epsilon': [0.1, 0.2]  # Reducido
                },
                'feature_selection': True,
                'fast': False,
                'sample_limit': self.max_samples_svr  # Limitar muestras para SVR
            }
        else:
            logger.info("MODO RAPIDO: SVR omitido por ser lento en datasets grandes")
        
        # Modelos de ensemble (más rápidos)
        models.update({
            'RandomForest': {
                'model': RandomForestRegressor(
                    random_state=self.random_state,
                    n_jobs=self.n_jobs if self.n_jobs != -1 else None
                ),
                'params': {
                    'model__n_estimators': [100, 200] if self.fast_mode else [100, 200, 300, 500],
                    'model__max_depth': [10, 20, 30] if self.fast_mode else [None, 10, 20, 30, 50],
                    'model__min_samples_split': [2, 5, 10],
                    'model__min_samples_leaf': [1, 2, 4],
                    'model__max_features': ['sqrt', 'log2']
                },
                'feature_selection': False,
                'fast': True
            },
            'GradientBoosting': {
                'model': GradientBoostingRegressor(random_state=self.random_state),
                'params': {
                    'model__n_estimators': [100, 200] if self.fast_mode else [100, 200, 300],
                    'model__learning_rate': [0.01, 0.1, 0.2],
                    'model__max_depth': [3, 5, 7],
                    'model__subsample': [0.8, 0.9, 1.0]
                },
                'feature_selection': False,
                'fast': True
            }
        })
        
        # Agregar modelos GPU/CPU según disponibilidad con manejo robusto de errores
        if self.gpu_available:
            models.update({
                'XGBoost_GPU': {
                    'model': xgb.XGBRegressor(
                        tree_method='gpu_hist', 
                        gpu_id=0, 
                        random_state=self.random_state, 
                        verbosity=0,
                        eval_metric='rmse'
                    ),
                    'params': {
                        'model__n_estimators': [100, 200] if self.fast_mode else [100, 200, 300, 500],
                        'model__learning_rate': [0.01, 0.1, 0.2],
                        'model__max_depth': [3, 5, 7],
                        'model__subsample': [0.8, 0.9, 1.0]
                    },
                    'feature_selection': False,
                    'fast': True
                }
            })
        else:
            # Versiones CPU optimizadas con manejo robusto
            try:
                models.update({
                    'XGBoost_CPU': {
                        'model': xgb.XGBRegressor(
                            random_state=self.random_state, 
                            verbosity=0,
                            eval_metric='rmse',
                            n_jobs=1  # Evitar paralelización que causa problemas
                        ),
                        'params': {
                            'model__n_estimators': [100, 200],
                            'model__learning_rate': [0.01, 0.1, 0.2],
                            'model__max_depth': [3, 5, 7],
                            'model__subsample': [0.8, 0.9, 1.0]
                        },
                        'feature_selection': False,
                        'fast': True
                    }
                })
            except Exception as e:
                logger.warning(f"Error configurando modelos avanzados: {e}")
                logger.info("Continuando solo con modelos básicos")
        
        return models
    
    @contextlib.contextmanager
    def _suppress_stdout_stderr(self):
        """Suprime stdout y stderr para modelos ruidosos"""
        with open(os.devnull, 'w') as fnull:
            old_stdout = sys.stdout
            old_stderr = sys.stderr
            sys.stdout = fnull
            sys.stderr = fnull
            try:
                yield
            finally:
                sys.stdout = old_stdout
                sys.stderr = old_stderr
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
        """Calcula múltiples métricas de evaluación"""
        return {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'explained_variance': explained_variance_score(y_true, y_pred)
        }
    
    def _create_pipeline(self, model: Any, use_feature_selection: bool = False, 
                        feature_k: int = 10) -> Pipeline:
        """Crea pipeline con escalado y selección de características opcional"""
        steps = [('scaler', RobustScaler())]
        
        if use_feature_selection:
            steps.append(('feature_selection', SelectKBest(f_regression, k=feature_k)))
        
        steps.append(('model', model))
        return Pipeline(steps)
    
    def train_single_model(self, model_name: str, model_info: Dict[str, Any], 
                          X_train: np.ndarray, X_test: np.ndarray, 
                          y_train: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """Entrena y evalúa un modelo individual"""
        start_time = time.time()
        
        # Estimar tiempo y manejar muestreo para modelos lentos
        if 'SVR' in model_name:
            n_samples = X_train.shape[0]
            sample_limit = model_info.get('sample_limit', 10000)
            
            if n_samples > sample_limit:
                logger.warning(f"Dataset grande ({n_samples:,} muestras) detectado para SVR")
                logger.info(f"Usando submuestreo de {sample_limit:,} muestras para SVR por eficiencia")
                
                # Crear índices aleatorios para muestreo
                indices = np.random.choice(n_samples, size=sample_limit, replace=False)
                X_train_sample = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
                y_train_sample = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
                
                logger.info(f"Muestras para SVR: {X_train_sample.shape[0]:,}")
            else:
                X_train_sample = X_train
                y_train_sample = y_train
                logger.info(f"Usando todas las muestras para SVR: {n_samples:,}")
        else:
            X_train_sample = X_train
            y_train_sample = y_train
        
        logger.info(f"Entrenando {model_name}...")
        
        try:
            # Crear pipeline
            pipeline = self._create_pipeline(
                model_info['model'], 
                model_info.get('feature_selection', False),
                min(10, X_train_sample.shape[1] // 2)
            )
            
            # Configurar búsqueda de hiperparámetros
            if model_info['params']:
                # Reducir iteraciones para modelos lentos
                if model_info.get('fast', True):
                    n_iter = min(20, len(list(model_info['params'].values())[0]) * 2)
                else:
                    n_iter = min(10, len(list(model_info['params'].values())[0]))
            else:
                n_iter = 1
            
            logger.info(f"Busqueda de hiperparametros: {n_iter} iteraciones")
            
            # Crear RandomizedSearchCV con manejo robusto
            search = RandomizedSearchCV(
                pipeline,
                model_info['params'],
                n_iter=n_iter,
                cv=min(self.cv_folds, 3) if 'SVR' in model_name else self.cv_folds,
                scoring='r2',
                n_jobs=1,  # Siempre usar 1 para evitar problemas de compatibilidad
                random_state=self.random_state,
                verbose=0,
                return_train_score=True,
                error_score='raise'  # Para debugging mejor
            )
            
            # Entrenar modelo con estimación de tiempo
            if 'XGBoost' in model_name:
                with self._suppress_stdout_stderr():
                    search.fit(X_train_sample, y_train_sample)
            else:
                # Mostrar estimación para modelos potencialmente lentos
                if 'SVR' in model_name:
                    estimated_time = (X_train_sample.shape[0] / 1000) * n_iter * 0.5
                    logger.info(f"Tiempo estimado: ~{estimated_time:.1f} segundos")
                
                search.fit(X_train_sample, y_train_sample)
            
            # Realizar predicciones en el conjunto completo
            y_pred_train = search.predict(X_train)
            y_pred_test = search.predict(X_test)
            
            # Calcular métricas
            train_metrics = self._calculate_metrics(y_train, y_pred_train)
            test_metrics = self._calculate_metrics(y_test, y_pred_test)
            
            # Cross-validation score con manejo robusto
            try:
                cv_scores = cross_val_score(
                    search.best_estimator_, 
                    X_train_sample, 
                    y_train_sample, 
                    cv=min(self.cv_folds, 3) if 'SVR' in model_name else self.cv_folds, 
                    scoring='r2', 
                    n_jobs=1
                )
            except Exception as cv_error:
                logger.warning(f"Error en cross-validation para {model_name}: {cv_error}")
                cv_scores = np.array([test_metrics['r2']])  # Fallback
            
            training_time = time.time() - start_time
            
            result = {
                'model': search,
                'best_estimator': search.best_estimator_,
                'train_metrics': train_metrics,
                'test_metrics': test_metrics,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'best_params': search.best_params_ if model_info['params'] else {},
                'training_time': training_time,
                'predictions': y_pred_test,
                'samples_used': X_train_sample.shape[0],
                'total_samples': X_train.shape[0]
            }
            
            logger.info(f"COMPLETADO {model_name} en {training_time:.2f}s")
            logger.info(f"R2 Test: {test_metrics['r2']:.4f} (CV: {cv_scores.mean():.4f} +/-{cv_scores.std():.4f})")
            if 'SVR' in model_name and X_train_sample.shape[0] != X_train.shape[0]:
                logger.info(f"Entrenado con {X_train_sample.shape[0]:,}/{X_train.shape[0]:,} muestras")
            
            return result
            
        except Exception as e:
            logger.error(f"ERROR entrenando {model_name}: {str(e)}")
            logger.warning(f"Saltando {model_name} y continuando con otros modelos")
            return None
    
    def train_all_models(self, X_train: np.ndarray, X_test: np.ndarray, 
                        y_train: np.ndarray, y_test: np.ndarray) -> None:
        """Entrena todos los modelos configurados"""
        logger.info("Iniciando entrenamiento de todos los modelos...")
        logger.info(f"Tamaño datos entrenamiento: {X_train.shape}")
        logger.info(f"Tamaño datos prueba: {X_test.shape}")
        
        total_start_time = time.time()
        
        for model_name, model_info in self.models.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluando {model_name}")
            logger.info('='*60)
            
            result = self.train_single_model(model_name, model_info, X_train, X_test, y_train, y_test)
            
            if result is not None:
                self.results[model_name] = result
        
        total_time = time.time() - total_start_time
        logger.info(f"\nEntrenamiento completo en {total_time:.2f}s")
        
        # Identificar mejor modelo
        self._find_best_model()
    
    def _find_best_model(self) -> None:
        """Identifica el mejor modelo basado en R2 score con cross-validation"""
        if not self.results:
            logger.warning("No hay resultados para evaluar")
            return
        
        best_score = float('-inf')
        best_model_name = None
        
        for model_name, result in self.results.items():
            # Usar CV mean score como métrica principal
            score = result['cv_mean']
            if score > best_score:
                best_score = score
                best_model_name = model_name
        
        self.best_model_info = {
            'name': best_model_name,
            'score': best_score,
            'result': self.results[best_model_name]
        }
        
        logger.info(f"\nMejor modelo: {best_model_name}")
        logger.info(f"R2 CV Score: {best_score:.4f}")
    
    def save_best_model(self, filepath: Optional[str] = None) -> str:
        """Guarda el mejor modelo con metadata"""
        if not self.best_model_info:
            raise ValueError("No se ha encontrado un mejor modelo")
        
        if filepath is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filepath = f"mejor_modelo_{self.best_model_info['name']}_{timestamp}.joblib"
        
        # Guardar modelo
        joblib.dump(self.best_model_info['result']['best_estimator'], filepath)
        
        # Guardar metadata
        metadata = {
            'model_name': self.best_model_info['name'],
            'best_params': self.best_model_info['result']['best_params'],
            'metrics': self.best_model_info['result']['test_metrics'],
            'cv_score': self.best_model_info['score'],
            'cv_std': self.best_model_info['result']['cv_std'],
            'training_time': self.best_model_info['result']['training_time'],
            'timestamp': datetime.now().isoformat()
        }
        
        metadata_path = filepath.replace('.joblib', '_metadata.json')
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        logger.info(f"Modelo guardado: {filepath}")
        logger.info(f"Metadata guardada: {metadata_path}")
        
        return filepath
    
    def generate_report(self) -> pd.DataFrame:
        """Genera un reporte detallado de todos los modelos"""
        if not self.results:
            return pd.DataFrame()
        
        report_data = []
        for model_name, result in self.results.items():
            row = {
                'Modelo': model_name,
                'R2_Test': result['test_metrics']['r2'],
                'RMSE_Test': result['test_metrics']['rmse'],
                'MAE_Test': result['test_metrics']['mae'],
                'R2_CV_Mean': result['cv_mean'],
                'R2_CV_Std': result['cv_std'],
                'Tiempo_Entrenamiento': result['training_time'],
                'Overfitting': result['train_metrics']['r2'] - result['test_metrics']['r2']
            }
            report_data.append(row)
        
        df_report = pd.DataFrame(report_data)
        df_report = df_report.sort_values('R2_CV_Mean', ascending=False)
        
        return df_report
    
    def plot_results(self, save_plots: bool = True) -> None:
        """Genera gráficos de los resultados"""
        if not self.results:
            logger.warning("No hay resultados para graficar")
            return
        
        # Configurar estilo
        plt.style.use('seaborn-v0_8')
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        fig.suptitle('Comparación de Modelos de Machine Learning', fontsize=16, fontweight='bold')
        
        # Preparar datos
        model_names = list(self.results.keys())
        r2_scores = [self.results[name]['test_metrics']['r2'] for name in model_names]
        rmse_scores = [self.results[name]['test_metrics']['rmse'] for name in model_names]
        cv_means = [self.results[name]['cv_mean'] for name in model_names]
        cv_stds = [self.results[name]['cv_std'] for name in model_names]
        
        # Gráfico 1: R2 Scores
        axes[0, 0].bar(range(len(model_names)), r2_scores, color='skyblue', alpha=0.7)
        axes[0, 0].set_title('R2 Score por Modelo')
        axes[0, 0].set_ylabel('R2 Score')
        axes[0, 0].set_xticks(range(len(model_names)))
        axes[0, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Gráfico 2: RMSE
        axes[0, 1].bar(range(len(model_names)), rmse_scores, color='lightcoral', alpha=0.7)
        axes[0, 1].set_title('RMSE por Modelo')
        axes[0, 1].set_ylabel('RMSE')
        axes[0, 1].set_xticks(range(len(model_names)))
        axes[0, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Gráfico 3: Cross-Validation con barras de error
        axes[1, 0].bar(range(len(model_names)), cv_means, yerr=cv_stds, 
                       color='lightgreen', alpha=0.7, capsize=5)
        axes[1, 0].set_title('Cross-Validation R2 Score (Media ± Std)')
        axes[1, 0].set_ylabel('CV R2 Score')
        axes[1, 0].set_xticks(range(len(model_names)))
        axes[1, 0].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gráfico 4: Tiempo de entrenamiento
        training_times = [self.results[name]['training_time'] for name in model_names]
        axes[1, 1].bar(range(len(model_names)), training_times, color='gold', alpha=0.7)
        axes[1, 1].set_title('Tiempo de Entrenamiento por Modelo')
        axes[1, 1].set_ylabel('Tiempo (segundos)')
        axes[1, 1].set_xticks(range(len(model_names)))
        axes[1, 1].set_xticklabels(model_names, rotation=45, ha='right')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_plots:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            plt.savefig(f'model_comparison_{timestamp}.png', dpi=300, bbox_inches='tight')
            logger.info(f"Gráficos guardados como: model_comparison_{timestamp}.png")
        
        plt.show()

# Función principal mejorada
def evaluar_modelos(X_train=None, X_test=None, y_train=None, y_test=None, 
                   data_path=None, test_size=0.2, n_jobs=-1, cv_folds=5, 
                   generate_plots=True, save_model=True, fast_mode=False, max_samples_svr=10000):
    """
    Función principal para evaluar múltiples modelos de ML
    
    Args:
        X_train, X_test, y_train, y_test: Datos de entrenamiento y prueba (opcional si se proporciona data_path)
        data_path: Ruta al archivo de datos (opcional si se proporcionan los datos divididos)
        test_size: Proporción de datos para prueba (solo si se usa data_path)
        n_jobs: Número de trabajos paralelos (-1 para usar todos los cores)
        cv_folds: Número de folds para cross-validation
        generate_plots: Si generar gráficos de resultados
        save_model: Si guardar el mejor modelo
        fast_mode: Si usar modo rápido (omite SVR y reduce búsqueda de hiperparámetros)
        max_samples_svr: Máximo número de muestras para SVR
    
    Returns:
        tuple: (evaluator, report_df, best_model_path)
    """
    
    logger.info("=== INICIANDO EVALUACIÓN DE MODELOS ===")
    
    if fast_mode:
        logger.info("MODO RAPIDO ACTIVADO")
        logger.info("   - SVR omitido (muy lento en datasets grandes)")
        logger.info("   - Busqueda de hiperparametros reducida")
        logger.info("   - Cross-validation optimizado")
    
    # Si no se proporcionan datos divididos, cargar y dividir desde archivo
    if X_train is None or X_test is None or y_train is None or y_test is None:
        if data_path is None:
            logger.info("No se proporcionaron datos ni ruta de archivo. Usando ruta por defecto.")
        
        X_train, X_test, y_train, y_test = load_and_split_data(data_path, test_size)
    
    # Validar datos
    if X_train.shape[0] == 0 or X_test.shape[0] == 0:
        raise ValueError("Los datos no pueden estar vacíos")
    
    # Recomendar modo rápido para datasets grandes
    total_samples = X_train.shape[0] + X_test.shape[0]
    if total_samples > 50000 and not fast_mode:
        logger.warning(f"Dataset grande detectado ({total_samples:,} muestras)")
        logger.warning("RECOMENDACION: Considera usar fast_mode=True para mayor velocidad")
        logger.warning("   Ejemplo: evaluar_modelos(..., fast_mode=True)")
    
    # Crear evaluador
    evaluator = ModelEvaluator(
        random_state=42, 
        n_jobs=n_jobs, 
        cv_folds=cv_folds,
        fast_mode=fast_mode,
        max_samples_svr=max_samples_svr
    )
    
    # Entrenar modelos
    evaluator.train_all_models(X_train, X_test, y_train, y_test)
    
    # Generar reporte
    report_df = evaluator.generate_report()
    
    # Mostrar resultados
    logger.info("\n" + "="*70)
    logger.info("REPORTE FINAL DE MODELOS")
    logger.info("="*70)
    print(report_df.round(4))
    
    # Generar gráficos
    if generate_plots:
        evaluator.plot_results()
    
    # Guardar mejor modelo
    best_model_path = None
    if save_model and evaluator.best_model_info:
        best_model_path = evaluator.save_best_model()
    
    logger.info("\n=== EVALUACIÓN COMPLETADA ===")
    
    return evaluator, report_df, best_model_path


def evaluar_modelos_completo(data_path=None, test_size=0.2, random_state=42, 
                           n_jobs=-1, cv_folds=5, generate_plots=True, save_model=True,
                           fast_mode=False, max_samples_svr=10000):
    """
    Función completa que incluye carga de datos, división y evaluación de modelos
    
    Args:
        data_path: Ruta al archivo de datos (opcional, usa ruta por defecto si es None)
        test_size: Proporción de datos para prueba
        random_state: Semilla para reproducibilidad
        n_jobs: Número de trabajos paralelos (-1 para usar todos los cores)
        cv_folds: Número de folds para cross-validation
        generate_plots: Si generar gráficos de resultados
        save_model: Si guardar el mejor modelo
        fast_mode: Si usar modo rápido (omite SVR y reduce búsqueda de hiperparámetros)
        max_samples_svr: Máximo número de muestras para SVR
    
    Returns:
        tuple: (data, X_train, X_test, y_train, y_test, evaluator, report_df, best_model_path)
    """
    
    logger.info("=== PROCESO COMPLETO DE MACHINE LEARNING ===")
    
    # 1. Cargar y dividir datos
    logger.info("Paso 1: Cargando y dividiendo datos...")
    X_train, X_test, y_train, y_test = load_and_split_data(data_path, test_size, random_state)
    
    # También mantener los datos originales para análisis
    data = load_data(data_path)
    
    # 2. Evaluar modelos
    logger.info("Paso 2: Evaluando modelos...")
    evaluator, report_df, best_model_path = evaluar_modelos(
        X_train=X_train, X_test=X_test, y_train=y_train, y_test=y_test,
        n_jobs=n_jobs, cv_folds=cv_folds, 
        generate_plots=generate_plots, save_model=save_model,
        fast_mode=fast_mode, max_samples_svr=max_samples_svr
    )
    
    logger.info("=== PROCESO COMPLETO FINALIZADO ===")
    
    return data, X_train, X_test, y_train, y_test, evaluator, report_df, best_model_path


# Función de ejemplo de uso rápido
def ejemplo_uso_rapido():
    """
    Ejemplo de uso rápido del sistema completo
    """
    print("=== EJEMPLO DE USO RÁPIDO ===")
    print("Opción 1 - Proceso completo (recomendado para nuevos usuarios):")
    print("""
    # Ejecutar todo el proceso desde cero
    data, X_train, X_test, y_train, y_test, evaluator, reporte, ruta_modelo = evaluar_modelos_completo()
    
    # Ver mejores modelos
    print("Mejor modelo:", evaluator.best_model_info['name'])
    print("Reporte:")
    print(reporte)
    """)
    
    print("\nOpción 2 - Solo evaluación (si ya tienes datos divididos):")
    print("""
    # Si ya tienes X_train, X_test, y_train, y_test
    evaluator, reporte, ruta_modelo = evaluar_modelos(
        X_train, X_test, y_train, y_test,
        n_jobs=-1,           # Usar todos los cores
        cv_folds=5,          # Cross-validation con 5 folds
        generate_plots=True, # Generar gráficos
        save_model=True      # Guardar el mejor modelo
    )
    """)
    
    print("\nOpción 3 - Carga automática con ruta personalizada:")
    print("""
    # Especificar ruta de datos personalizada
    evaluator, reporte, ruta_modelo = evaluar_modelos(
        data_path="mi_archivo.csv",
        test_size=0.3,       # 30% para prueba
        n_jobs=4,            # Usar 4 cores
        cv_folds=10          # Cross-validation con 10 folds
    )
    """)


if __name__ == "__main__":
    """
    Ejecutar directamente el script para evaluación automática
    """
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluación automática de modelos de ML')
    parser.add_argument('--data_path', type=str, default=None, 
                       help='Ruta al archivo de datos')
    parser.add_argument('--test_size', type=float, default=0.2, 
                       help='Proporción de datos para prueba')
    parser.add_argument('--n_jobs', type=int, default=-1, 
                       help='Número de trabajos paralelos')
    parser.add_argument('--cv_folds', type=int, default=5, 
                       help='Número de folds para cross-validation')
    parser.add_argument('--no_plots', action='store_true', 
                       help='No generar gráficos')
    parser.add_argument('--no_save', action='store_true', 
                       help='No guardar el mejor modelo')
    parser.add_argument('--fast_mode', action='store_true', 
                       help='Modo rápido: omite SVR y reduce búsqueda de hiperparámetros')
    parser.add_argument('--max_samples_svr', type=int, default=10000, 
                       help='Máximo número de muestras para SVR (por defecto: 10000)')
    
    args = parser.parse_args()
    
    # Mostrar configuración
    logger.info("CONFIGURACION DE EJECUCION:")
    logger.info(f"   - Modo rapido: {'SI' if args.fast_mode else 'NO'}")
    logger.info(f"   - Trabajos paralelos: {args.n_jobs}")
    logger.info(f"   - CV folds: {args.cv_folds}")
    logger.info(f"   - Tamaño de prueba: {args.test_size}")
    if not args.fast_mode:
        logger.info(f"   - Limite muestras SVR: {args.max_samples_svr:,}")
    
    # Ejecutar evaluación completa
    try:
        data, X_train, X_test, y_train, y_test, evaluator, report_df, best_model_path = evaluar_modelos_completo(
            data_path=args.data_path,
            test_size=args.test_size,
            n_jobs=args.n_jobs,
            cv_folds=args.cv_folds,
            generate_plots=not args.no_plots,
            save_model=not args.no_save,
            fast_mode=args.fast_mode,
            max_samples_svr=args.max_samples_svr
        )
        
        print("\n¡Evaluación completada exitosamente!")
        if best_model_path:
            print(f"Mejor modelo guardado en: {best_model_path}")
        
        # Mostrar resumen de tiempo
        total_time = sum(result['training_time'] for result in evaluator.results.values())
        print(f"Tiempo total de entrenamiento: {total_time:.2f} segundos")
        print(f"Modelos evaluados: {len(evaluator.results)}")
        
    except Exception as e:
        logger.error(f"Error durante la evaluación: {str(e)}")
        print(f"Error: {str(e)}")
        print("\nSugerencias:")
        print("1. Verifica que el archivo de datos existe")
        print("2. Revisa que las columnas requeridas estén presentes")
        print("3. Intenta con --fast_mode para datasets grandes")
        print("4. Ejecuta ejemplo_uso_rapido() para ver ejemplos de uso")

# Mostrar ejemplo de uso si se importa el módulo
if __name__ != "__main__":
    ejemplo_uso_rapido()