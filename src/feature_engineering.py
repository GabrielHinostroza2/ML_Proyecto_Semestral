#!/usr/bin/env python
# coding: utf-8

"""
Módulo para ingeniería de características para los datos de CS:GO
Optimizado para modelos de regresión y clasificación
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import PolynomialFeatures
from sklearn.decomposition import PCA
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor
import os

def load_processed_data(path='data/02_processed/processed_data.csv'):
    """
    Carga los datos procesados
    
    Args:
        path (str): Ruta al archivo CSV procesado
        
    Returns:
        pandas.DataFrame: DataFrame con los datos cargados
    """
    try:
        df = pd.read_csv(path)
        print(f'Datos procesados cargados: {df.shape}')
        return df
    except Exception as e:
        print(f"Error al cargar los datos procesados: {e}")
        return None

def create_basic_features(df):
    """
    Crea características básicas para mejorar los modelos
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos procesados
        
    Returns:
        pandas.DataFrame: DataFrame con nuevas características
    """
    df_new = df.copy()
    
    # 1. Ratio de headshots (precisión)
    df_new['HeadshotRatio'] = df_new.apply(
        lambda row: row['RoundHeadshots'] / row['RoundKills'] if row['RoundKills'] > 0 else 0, 
        axis=1
    )
    
    # 2. Eficiencia económica
    df_new['EconomicEfficiency'] = df_new.apply(
        lambda row: row['RoundKills'] / (row['RoundStartingEquipmentValue'] + 1),
        axis=1
    )
    
    # 3. Efectividad de granadas
    if 'RLethalGrenadesThrown' in df_new.columns:
        df_new['GrenadeEffectiveness'] = df_new.apply(
            lambda row: row['RoundKills'] / (row['RLethalGrenadesThrown'] + 1),
            axis=1
        )
    
    # 4. Indicador de efectividad global
    df_new['EffectivenessScore'] = df_new['RoundKills'] * 2 + df_new['RoundAssists']
    
    # 5. Indicador de uso de arma primaria
    weapon_cols = [col for col in df_new.columns if col.startswith('Primary')]
    if weapon_cols:
        df_new['UsingPrimaryWeapon'] = df_new[weapon_cols].max(axis=1) > 0
    
    # 6. Total de granadas lanzadas
    if 'RLethalGrenadesThrown' in df_new.columns and 'RNonLethalGrenadesThrown' in df_new.columns:
        df_new['TotalGrenades'] = df_new['RLethalGrenadesThrown'] + df_new['RNonLethalGrenadesThrown']
    
    # 7. Diferencia de equipamiento entre equipos
    if 'RoundStartingEquipmentValue' in df_new.columns and 'TeamStartingEquipmentValue' in df_new.columns:
        df_new['EquipmentAdvantage'] = df_new['RoundStartingEquipmentValue'] - df_new['TeamStartingEquipmentValue'] / 5
    
    # 8. KDA (Kill-Death-Assist ratio)
    df_new['KDA'] = df_new.apply(
        lambda row: (row['RoundKills'] + row['RoundAssists']) / (1 if row['Survived'] else 2),
        axis=1
    )
    
    # 9. Kill-Assist Ratio
    df_new['KillAssistRatio'] = df_new.apply(
        lambda row: row['RoundKills'] / (row['RoundAssists'] + 1),
        axis=1
    )
    
    # 10. Stealth Kills Ratio (Flank kills / Total kills)
    df_new['StealthKillsRatio'] = df_new.apply(
        lambda row: row['RoundFlankKills'] / row['RoundKills'] if row['RoundKills'] > 0 else 0,
        axis=1
    )
    
    return df_new

def create_advanced_features(df):
    """
    Crea características avanzadas para mejorar los modelos
    
    Args:
        df (pandas.DataFrame): DataFrame con los datos procesados y características básicas
        
    Returns:
        pandas.DataFrame: DataFrame con características avanzadas añadidas
    """
    df_advanced = df.copy()
    
    # 1. Características cuadráticas para variables clave
    key_columns = ['RoundHeadshots', 'RoundAssists', 'RoundFlankKills', 'HeadshotRatio', 'EffectivenessScore']
    key_columns = [col for col in key_columns if col in df_advanced.columns]
    
    for col in key_columns:
        df_advanced[f'{col}_squared'] = df_advanced[col] ** 2
    
    # 2. Interacciones entre variables clave
    for i, col1 in enumerate(key_columns):
        for col2 in key_columns[i+1:]:
            df_advanced[f'{col1}_{col2}_interaction'] = df_advanced[col1] * df_advanced[col2]
    
    # 3. Logaritmo para variables con distribución sesgada (si son mayores que 0)
    skewed_columns = ['EconomicEfficiency', 'EffectivenessScore', 'RoundStartingEquipmentValue', 'TeamStartingEquipmentValue']
    skewed_columns = [col for col in skewed_columns if col in df_advanced.columns]
    
    for col in skewed_columns:
        # Asegurar que no hay valores negativos o cero
        min_val = df_advanced[col].min()
        if min_val <= 0:
            offset = abs(min_val) + 1
            df_advanced[f'{col}_log'] = np.log(df_advanced[col] + offset)
        else:
            df_advanced[f'{col}_log'] = np.log(df_advanced[col])
    
    # 4. Características binarias/categóricas
    
    # Categoría de arma más usada
    weapon_cols = [col for col in df_advanced.columns if col.startswith('Primary')]
    if weapon_cols:
        df_advanced['MainWeaponCategory'] = df_advanced[weapon_cols].idxmax(axis=1)
        df_advanced['MainWeaponCategory'] = df_advanced['MainWeaponCategory'].str.replace('Primary', '')
    
    # Categorización del nivel de equipamiento
    if 'RoundStartingEquipmentValue' in df_advanced.columns:
        equip_value = df_advanced['RoundStartingEquipmentValue']
        df_advanced['EquipmentCategory'] = pd.qcut(
            equip_value, 
            q=4, 
            labels=['Bajo', 'Medio-Bajo', 'Medio-Alto', 'Alto']
        )
    
    # 5. Ratio normalizado (dividir por el máximo del equipo)
    team_features = {}
    if 'Team' in df_advanced.columns and 'MatchId' in df_advanced.columns:
        for team in df_advanced['Team'].unique():
            for col in ['RoundKills', 'RoundAssists', 'RoundHeadshots', 'RoundFlankKills']:
                if col in df_advanced.columns:
                    col_name = f'{col}_TeamNorm'
                    team_features[team] = df_advanced[df_advanced['Team'] == team].groupby(['MatchId', 'RoundId'])[col].transform('max')
                    
        for team, values in team_features.items():
            for col in ['RoundKills', 'RoundAssists', 'RoundHeadshots', 'RoundFlankKills']:
                if col in df_advanced.columns:
                    col_name = f'{col}_TeamNorm'
                    mask = df_advanced['Team'] == team
                    max_value = df_advanced.loc[mask].groupby(['MatchId', 'RoundId'])[col].transform('max')
                    df_advanced.loc[mask, col_name] = df_advanced.loc[mask, col] / (max_value + 0.01)
    
    return df_advanced

def select_features(X, y, method='mutual_info', k=20):
    """
    Selecciona las características más relevantes para la predicción
    
    Args:
        X (pandas.DataFrame): Características
        y (pandas.Series): Variable objetivo
        method (str): Método de selección ('mutual_info', 'f_regression', 'lasso', 'rfe')
        k (int): Número de características a seleccionar
        
    Returns:
        list: Lista de nombres de características seleccionadas
    """
    # Convertir columnas categóricas a numéricas
    X_numeric = X.copy()
    cat_cols = X_numeric.select_dtypes(include=['category']).columns
    for col in cat_cols:
        X_numeric[col] = X_numeric[col].cat.codes
    
    # Convertir booleanos a enteros
    bool_cols = X_numeric.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        X_numeric[col] = X_numeric[col].astype(int)
    
    # Manejar valores infinitos o NaN
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    
    # Asegurarse de que solo se seleccionen columnas numéricas para calcular la mediana
    numeric_cols = X_numeric.select_dtypes(include=['int64', 'float64']).columns
    X_numeric[numeric_cols] = X_numeric[numeric_cols].fillna(X_numeric[numeric_cols].median())
    
    # Filtrar solo las columnas numéricas para la selección de características
    X_numeric = X_numeric[numeric_cols]
    
    if method == 'mutual_info':
        selector = SelectKBest(mutual_info_regression, k=min(k, X_numeric.shape[1]))
        selector.fit(X_numeric, y)
        features_scores = pd.DataFrame({
            'Feature': X_numeric.columns,
            'Score': selector.scores_
        })
        features_scores = features_scores.sort_values('Score', ascending=False)
        selected_features = features_scores.head(k)['Feature'].tolist()
        
    elif method == 'f_regression':
        selector = SelectKBest(f_regression, k=min(k, X_numeric.shape[1]))
        selector.fit(X_numeric, y)
        features_scores = pd.DataFrame({
            'Feature': X_numeric.columns,
            'Score': selector.scores_
        })
        features_scores = features_scores.sort_values('Score', ascending=False)
        selected_features = features_scores.head(k)['Feature'].tolist()
        
    elif method == 'lasso':
        lasso = Lasso(alpha=0.01, random_state=42)
        lasso.fit(X_numeric, y)
        features_scores = pd.DataFrame({
            'Feature': X_numeric.columns,
            'Coefficient': np.abs(lasso.coef_)
        })
        features_scores = features_scores.sort_values('Coefficient', ascending=False)
        selected_features = features_scores.head(k)['Feature'].tolist()
        
    elif method == 'rfe':
        estimator = RandomForestRegressor(n_estimators=100, random_state=42)
        selector = RFE(estimator, n_features_to_select=min(k, X_numeric.shape[1]), step=1)
        selector.fit(X_numeric, y)
        features_scores = pd.DataFrame({
            'Feature': X_numeric.columns,
            'Rank': selector.ranking_
        })
        features_scores = features_scores.sort_values('Rank')
        selected_features = features_scores.head(k)['Feature'].tolist()
    
    return selected_features

def create_polynomial_features(X, degree=2):
    """
    Crea características polinómicas a partir de las características existentes
    
    Args:
        X (pandas.DataFrame): DataFrame con características
        degree (int): Grado del polinomio
        
    Returns:
        pandas.DataFrame: DataFrame con características polinómicas
    """
    # Seleccionar solo columnas numéricas
    X_numeric = X.select_dtypes(include=['int64', 'float64'])
    
    # Crear transformador polinómico
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    
    # Ajustar y transformar los datos
    poly_features = poly.fit_transform(X_numeric)
    
    # Crear nombres de características
    feature_names = poly.get_feature_names_out(X_numeric.columns)
    
    # Crear DataFrame con las nuevas características
    poly_df = pd.DataFrame(poly_features, columns=feature_names, index=X.index)
    
    # Conservar columnas categóricas
    categorical_df = X.select_dtypes(exclude=['int64', 'float64'])
    
    # Unir las características polinómicas con las categóricas
    result = pd.concat([poly_df, categorical_df], axis=1)
    
    return result

def reduce_dimensions(X, n_components=0.95, method='pca'):
    """
    Reduce la dimensionalidad de las características
    
    Args:
        X (pandas.DataFrame): DataFrame con características
        n_components (int o float): Número de componentes o varianza explicada
        method (str): Método de reducción de dimensionalidad ('pca')
        
    Returns:
        pandas.DataFrame: DataFrame con dimensionalidad reducida
    """
    # Seleccionar solo columnas numéricas
    X_numeric = X.select_dtypes(include=['int64', 'float64'])
    
    if method == 'pca':
        # Crear y ajustar PCA
        pca = PCA(n_components=n_components)
        pca_result = pca.fit_transform(X_numeric)
        
        # Crear nombres para las componentes
        if isinstance(n_components, float):
            n_components = pca.n_components_
            
        component_names = [f'PC{i+1}' for i in range(n_components)]
        
        # Crear DataFrame con las componentes principales
        pca_df = pd.DataFrame(pca_result, columns=component_names, index=X.index)
        
        # Mostrar la varianza explicada
        explained_variance = pca.explained_variance_ratio_
        print(f"Varianza explicada por componente: {explained_variance}")
        print(f"Varianza acumulada explicada: {np.sum(explained_variance):.4f}")
        
        # Conservar columnas categóricas
        categorical_df = X.select_dtypes(exclude=['int64', 'float64'])
        
        # Unir las componentes principales con las características categóricas
        result = pd.concat([pca_df, categorical_df], axis=1)
        
        return result, pca
    
    return X, None

def plot_feature_importance(X, y, method='rf', n_top=15):
    """
    Visualiza la importancia de las características
    
    Args:
        X (pandas.DataFrame): DataFrame con características
        y (pandas.Series): Variable objetivo
        method (str): Método para calcular la importancia ('rf' o 'lasso')
        n_top (int): Número de características principales a mostrar
    """
    # Convertir columnas categóricas a numéricas
    X_numeric = X.copy()
    cat_cols = X_numeric.select_dtypes(include=['category']).columns
    for col in cat_cols:
        X_numeric[col] = X_numeric[col].cat.codes
    
    # Convertir booleanos a enteros
    bool_cols = X_numeric.select_dtypes(include=['bool']).columns
    for col in bool_cols:
        X_numeric[col] = X_numeric[col].astype(int)
    
    # Manejar valores infinitos o NaN
    X_numeric = X_numeric.replace([np.inf, -np.inf], np.nan)
    
    # Asegurarse de que solo se seleccionen columnas numéricas para calcular la mediana
    numeric_cols = X_numeric.select_dtypes(include=['int64', 'float64']).columns
    X_numeric[numeric_cols] = X_numeric[numeric_cols].fillna(X_numeric[numeric_cols].median())
    
    # Filtrar solo las columnas numéricas para la importancia
    X_numeric = X_numeric[numeric_cols]
    
    if method == 'rf':
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_numeric, y)
        
        # Obtener importancia de características
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': X_numeric.columns,
            'Importance': importances
        })
        feature_importance = feature_importance.sort_values('Importance', ascending=False).head(n_top)
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title('Importancia de características (Random Forest)', fontsize=14)
        plt.xlabel('Importancia', fontsize=12)
        plt.ylabel('Característica', fontsize=12)
        plt.tight_layout()
        
        # Asegurarse de que el directorio existe antes de guardar
        os.makedirs('reports/figures', exist_ok=True)
        plt.savefig('reports/figures/feature_importance_rf.png')
        plt.show()
        
    elif method == 'lasso':
        model = Lasso(alpha=0.01, random_state=42)
        model.fit(X_numeric, y)
        
        # Obtener coeficientes
        coefficients = model.coef_
        feature_importance = pd.DataFrame({
            'Feature': X_numeric.columns,
            'Coefficient': np.abs(coefficients)
        })
        feature_importance = feature_importance.sort_values('Coefficient', ascending=False).head(n_top)
        
        # Crear gráfico
        plt.figure(figsize=(12, 8))
        sns.barplot(x='Coefficient', y='Feature', data=feature_importance)
        plt.title('Coeficientes absolutos (Lasso)', fontsize=14)
        plt.xlabel('Coeficiente Absoluto', fontsize=12)
        plt.ylabel('Característica', fontsize=12)
        plt.tight_layout()
        plt.savefig('reports/figures/feature_importance_lasso.png')
        plt.show()

def save_engineered_data(df, path='data/03_features/engineered_data.csv'):
    """
    Guarda los datos con ingeniería de características
    
    Args:
        df (pandas.DataFrame): DataFrame con características ingeniadas
        path (str): Ruta donde guardar el archivo
        
    Returns:
        bool: True si se guardó correctamente, False en caso contrario
    """
    try:
        # Asegurar que el directorio existe
        os.makedirs(os.path.dirname(path), exist_ok=True)
        
        # Guardar el archivo
        df.to_csv(path, index=False)
        print(f"Datos con ingeniería de características guardados en: {path}")
        return True
    except Exception as e:
        print(f"Error al guardar los datos: {e}")
        return False

def main():
    """Función principal para ejecutar la ingeniería de características"""
    # Cargar datos procesados
    df = load_processed_data()
    if df is None:
        return
    
    # Crear características básicas
    df_basic = create_basic_features(df)
    print(f"DataFrame con características básicas: {df_basic.shape}")
    
    # Crear características avanzadas
    df_advanced = create_advanced_features(df_basic)
    print(f"DataFrame con características avanzadas: {df_advanced.shape}")
    
    # Seleccionar características relevantes para RoundKills
    target = 'RoundKills'
    if target in df_advanced.columns:
        y = df_advanced[target]
        X = df_advanced.drop(columns=[target, 'MatchId', 'RoundId'])
        
        # Seleccionar las mejores características mediante diferentes métodos
        selected_features_mi = select_features(X, y, method='mutual_info', k=30)
        selected_features_f = select_features(X, y, method='f_regression', k=30)
        selected_features_lasso = select_features(X, y, method='lasso', k=30)
        
        # Conjunto de características como la unión de las tres listas
        unique_selected_features = list(set(selected_features_mi + selected_features_f + selected_features_lasso))
        
        # Visualizar importancia de características
        plot_feature_importance(X, y, method='rf')
        
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
        save_engineered_data(df_final)
    else:
        print(f"La columna {target} no existe en el DataFrame")

if __name__ == "__main__":
    main() 