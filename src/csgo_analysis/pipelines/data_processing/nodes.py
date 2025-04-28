"""
Data processing nodes for CSGO 2 analysis
"""

import pandas as pd
import numpy as np
from typing import Dict, List

def clean_csgo_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean the CSGO 2 dataset by handling missing values,
    removing duplicates, and converting data types.
    """
    # Create a copy of the dataframe
    df_cleaned = df.copy()
    
    # Remove duplicates
    df_cleaned = df_cleaned.drop_duplicates()
    
    # Handle missing values based on column type
    numeric_columns = df_cleaned.select_dtypes(include=['int64', 'float64']).columns
    categorical_columns = df_cleaned.select_dtypes(include=['object']).columns
    
    # Fill numeric missing values with median
    for col in numeric_columns:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].median())
    
    # Fill categorical missing values with mode
    for col in categorical_columns:
        df_cleaned[col] = df_cleaned[col].fillna(df_cleaned[col].mode()[0])
    
    return df_cleaned

def validate_data(df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Validate the cleaned data and generate validation reports.
    """
    validation_results = {
        'missing_values': df.isnull().sum().to_frame('count'),
        'data_types': df.dtypes.to_frame('dtype'),
        'unique_values': df.nunique().to_frame('count')
    }
    
    return validation_results

def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create new features from existing data.
    This is a placeholder - modify based on actual CSGO 2 data structure.
    """
    df_featured = df.copy()
    
    # Example feature engineering (modify based on actual data):
    # - Calculate KDA ratio
    # - Create performance indicators
    # - Generate time-based features
    
    return df_featured

def prepare_analysis_data(
    df: pd.DataFrame,
    categorical_columns: List[str],
    numeric_columns: List[str]
) -> pd.DataFrame:
    """
    Prepare data for analysis by encoding categorical variables
    and scaling numeric variables.
    """
    df_prepared = df.copy()
    
    # Encode categorical variables
    for col in categorical_columns:
        if col in df_prepared.columns:
            df_prepared[col] = pd.Categorical(df_prepared[col]).codes
    
    # Scale numeric variables
    for col in numeric_columns:
        if col in df_prepared.columns:
            df_prepared[col] = (df_prepared[col] - df_prepared[col].mean()) / df_prepared[col].std()
    
    return df_prepared
