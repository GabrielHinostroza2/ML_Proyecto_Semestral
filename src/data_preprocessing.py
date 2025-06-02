#!/usr/bin/env python
# coding: utf-8

"""
Módulo para limpieza y preprocesamiento de datos de CS:GO
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

def load_data(path='data/01_raw/Anexo_ET_demo_round_traces_2022.csv'):
    """
    Carga los datos desde un archivo CSV
    
    Args:
        path (str): Ruta al archivo CSV
        
    Returns:
        pandas.DataFrame: DataFrame con los datos cargados
    """
    try:
        df = pd.read_csv(path, sep=';', low_memory=False)
        print(f'Dataset cargado: {df.shape}')
        return df
    except Exception as e:
        print(f"Error al cargar los datos: {e}")
        return None

def clean_data(df):
    """
    Limpia el DataFrame eliminando columnas innecesarias y manejando valores nulos
    
    Args:
        df (pandas.DataFrame): DataFrame a limpiar
        
    Returns:
        pandas.DataFrame: DataFrame limpio
    """
    # Hacer una copia para evitar modificar el original
    df_clean = df.copy()
    
    # Eliminar columnas innecesarias
    columns_to_drop = [
        'Unnamed: 0', 'InternalTeamId',
        'TimeAlive', 'TravelledDistance',
        'TimeAlive_ns', 'TimeAlive_s_rel',
        'TravelledDistance_ns', 'TravelledDistance_s_abs'
    ]
    df_clean = df_clean.drop(columns=[c for c in columns_to_drop if c in df_clean.columns])
    
    # Imputación de valores nulos para columnas de combate
    combat_cols = ['RoundKills', 'RoundAssists', 'RoundHeadshots', 'RoundFlankKills', 
                   'MatchKills', 'MatchAssists', 'MatchHeadshots', 'MatchFlankKills']
    for col in combat_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].fillna(0).astype(int)
    
    # Otras columnas numéricas: imputar con la mediana (más robusta que la media para outliers)
    numeric_cols = df_clean.select_dtypes(include=['float64', 'int64']).columns
    for col in numeric_cols:
        if col not in combat_cols and df_clean[col].isnull().sum() > 0:
            df_clean[col] = df_clean[col].fillna(df_clean[col].median())
    
    # Convertir tipos de datos
    categorical_cols = ['Map', 'Team']
    for col in categorical_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('category')
    
    boolean_cols = ['RoundWinner', 'MatchWinner', 'Survived', 'AbnormalMatch']
    for col in boolean_cols:
        if col in df_clean.columns:
            df_clean[col] = df_clean[col].astype('bool')
    
    # Eliminar filas con datos faltantes restantes (si hay alguno)
    df_clean = df_clean.dropna()
    
    # Eliminar duplicados si existen
    df_clean = df_clean.drop_duplicates()
    
    # Eliminar filas con partidas anormales
    if 'AbnormalMatch' in df_clean.columns:
        df_clean = df_clean[~df_clean['AbnormalMatch']]
        
    print(f"Dataset limpio: {df_clean.shape}")
    return df_clean

def detect_outliers(df, columns, method='iqr', threshold=1.5):
    """
    Detecta outliers en las columnas especificadas
    
    Args:
        df (pandas.DataFrame): DataFrame a analizar
        columns (list): Lista de columnas para detectar outliers
        method (str): Método para detectar outliers ('iqr' o 'zscore')
        threshold (float): Umbral para considerar un valor como outlier
        
    Returns:
        dict: Diccionario con índices de outliers por columna
    """
    outliers = {}
    
    for col in columns:
        if col not in df.columns:
            continue
            
        if method == 'iqr':
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            outliers[col] = df[(df[col] < lower_bound) | (df[col] > upper_bound)].index.tolist()
            
        elif method == 'zscore':
            from scipy import stats
            z_scores = np.abs(stats.zscore(df[col]))
            outliers[col] = df[z_scores > threshold].index.tolist()
    
    return outliers

def handle_outliers(df, columns, method='winsorize', **kwargs):
    """
    Maneja outliers en las columnas especificadas
    
    Args:
        df (pandas.DataFrame): DataFrame a procesar
        columns (list): Lista de columnas para manejar outliers
        method (str): Método para manejar outliers ('remove', 'winsorize', o 'cap')
        
    Returns:
        pandas.DataFrame: DataFrame con outliers manejados
    """
    df_clean = df.copy()
    
    for col in columns:
        if col not in df_clean.columns:
            continue
            
        if method == 'remove':
            outliers = detect_outliers(df_clean, [col], **kwargs)
            df_clean = df_clean.drop(index=outliers[col])
            
        elif method == 'winsorize':
            from scipy.stats import mstats
            limits = kwargs.get('limits', (0.05, 0.05))
            df_clean[col] = mstats.winsorize(df_clean[col], limits=limits)
            
        elif method == 'cap':
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            threshold = kwargs.get('threshold', 1.5)
            lower_bound = Q1 - threshold * IQR
            upper_bound = Q3 + threshold * IQR
            df_clean[col] = df_clean[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df_clean

def create_preprocessor(numeric_cols, categorical_cols):
    """
    Crea un preprocesador para transformaciones de características
    
    Args:
        numeric_cols (list): Lista de columnas numéricas
        categorical_cols (list): Lista de columnas categóricas
        
    Returns:
        ColumnTransformer: Preprocesador configurado
    """
    numeric_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_cols),
            ('cat', categorical_transformer, categorical_cols)
        ],
        remainder='passthrough'
    )
    
    return preprocessor

def analyze_data_quality(df):
    """
    Analiza la calidad de los datos
    
    Args:
        df (pandas.DataFrame): DataFrame a analizar
        
    Returns:
        pandas.DataFrame: DataFrame con estadísticas de calidad
    """
    # Calcular estadísticas de calidad
    stats = pd.DataFrame({
        'dtype': df.dtypes,
        'count': df.count(),
        'missing': df.isnull().sum(),
        'missing_pct': (df.isnull().sum() / len(df) * 100).round(2),
        'unique': df.nunique(),
        'unique_pct': (df.nunique() / len(df) * 100).round(2)
    })
    
    # Agregar estadísticas para columnas numéricas
    numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
    for col in numeric_cols:
        stats.loc[col, 'min'] = df[col].min()
        stats.loc[col, 'max'] = df[col].max()
        stats.loc[col, 'mean'] = df[col].mean().round(2)
        stats.loc[col, 'median'] = df[col].median()
        std_value = df[col].std()
        # Usar np.nan para manejar NaN y redondear solo si no es NaN
        stats.loc[col, 'std'] = np.round(std_value, 2) if not np.isnan(std_value) else np.nan
        stats.loc[col, 'skew'] = df[col].skew().round(2)
    
    return stats

def save_processed_data(df, path='data/02_processed/processed_data.csv'):
    """
    Guarda los datos procesados en un archivo CSV
    
    Args:
        df (pandas.DataFrame): DataFrame a guardar
        path (str): Ruta donde guardar el archivo
        
    Returns:
        bool: True si se guardó correctamente, False en caso contrario
    """
    try:
        # Asegurar que el directorio existe
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Guardar el archivo
        df.to_csv(path, index=False)
        print(f"Datos guardados en: {path}")
        return True
    except Exception as e:
        print(f"Error al guardar los datos: {e}")
        return False

def main():
    """Función principal para ejecutar el preprocesamiento de datos"""
    # Cargar datos
    df = load_data()
    if df is None:
        return
    
    # Limpiar datos
    df_clean = clean_data(df)
    
    # Analizar calidad de los datos
    quality_stats = analyze_data_quality(df_clean)
    print("\nEstadísticas de calidad de datos:")
    print(quality_stats)
    
    # Detectar y manejar outliers en columnas numéricas
    numeric_cols = df_clean.select_dtypes(include=['int64', 'float64']).columns
    df_clean = handle_outliers(df_clean, numeric_cols, method='winsorize', limits=(0.05, 0.05))
    
    # Guardar datos procesados
    save_processed_data(df_clean)
    
    # Convertir RoundKills a categorías
    df_clean['RoundKills'] = df_clean['RoundKills'].astype('category')
    
    # Usar train_test_split estratificado
    X = df_clean.drop(columns=['RoundKills'])
    y = df_clean['RoundKills']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Definir modelos de clasificación en lugar de regresión
    models = {
        'LogisticRegression': Pipeline(steps=[
            ('classifier', LogisticRegression(random_state=42))
        ]),
        'RandomForestClassifier': Pipeline(steps=[
            ('classifier', RandomForestClassifier(random_state=42))
        ]),
        'SVC': Pipeline(steps=[
            ('classifier', SVC(random_state=42))
        ])
    }
    
    # Rest of the function remains unchanged
    # ...

if __name__ == "__main__":
    main() 