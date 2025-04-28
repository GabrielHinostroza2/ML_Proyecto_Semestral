"""
This is a boilerplate pipeline 'eda'
generated using Kedro 0.19.1
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, Tuple

def calculate_summary_statistics(data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    """
    Calculate summary statistics for numerical columns.
    """
    numeric_stats = data.describe()
    categorical_cols = data.select_dtypes(include=['object']).columns
    categorical_stats = data[categorical_cols].nunique()
    
    return {
        "numeric_summary": numeric_stats,
        "categorical_summary": categorical_stats.to_frame('unique_values')
    }

def create_distribution_plots(data: pd.DataFrame) -> Dict[str, plt.Figure]:
    """
    Create distribution plots for numerical variables.
    """
    numeric_cols = data.select_dtypes(include=['int64', 'float64']).columns
    plots = {}
    
    for col in numeric_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.histplot(data=data, x=col, ax=ax)
        ax.set_title(f'Distribution of {col}')
        plots[f"{col}_distribution"] = fig
        plt.close(fig)
    
    return plots

def correlation_analysis(data: pd.DataFrame) -> Tuple[pd.DataFrame, plt.Figure]:
    """
    Perform correlation analysis on numerical variables.
    """
    numeric_data = data.select_dtypes(include=['int64', 'float64'])
    correlation_matrix = numeric_data.corr()
    
    fig, ax = plt.subplots(figsize=(12, 8))
    sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', ax=ax)
    ax.set_title('Correlation Matrix')
    
    return correlation_matrix, fig

def categorical_analysis(data: pd.DataFrame) -> Dict[str, plt.Figure]:
    """
    Analyze categorical variables.
    """
    categorical_cols = data.select_dtypes(include=['object']).columns
    plots = {}
    
    for col in categorical_cols:
        fig, ax = plt.subplots(figsize=(10, 6))
        data[col].value_counts().plot(kind='bar', ax=ax)
        ax.set_title(f'Distribution of {col}')
        ax.tick_params(axis='x', rotation=45)
        plots[f"{col}_distribution"] = fig
        plt.close(fig)
    
    return plots
