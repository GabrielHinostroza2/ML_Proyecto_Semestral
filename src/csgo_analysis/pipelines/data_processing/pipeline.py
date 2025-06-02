"""
Data processing pipeline for CSGO 2 analysis
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    clean_csgo_data,
    validate_data,
    feature_engineering,
    prepare_analysis_data
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=clean_csgo_data,
                inputs="raw_csgo_data",
                outputs="cleaned_csgo_data",
                name="clean_csgo_data_node",
            ),
            node(
                func=validate_data,
                inputs="cleaned_csgo_data",
                outputs="data_validation_report",
                name="validate_data_node",
            ),
            node(
                func=feature_engineering,
                inputs="cleaned_csgo_data",
                outputs="featured_csgo_data",
                name="feature_engineering_node",
            ),
            node(
                func=prepare_analysis_data,
                inputs=["featured_csgo_data", "params:categorical_columns", "params:numeric_columns"],
                outputs="preprocessed_csgo_data",
                name="prepare_analysis_data_node",
            ),
        ]
    )
