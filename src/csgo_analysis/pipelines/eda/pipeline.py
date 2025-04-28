"""
This is a boilerplate pipeline 'eda'
generated using Kedro 0.19.1
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import (
    calculate_summary_statistics,
    create_distribution_plots,
    correlation_analysis,
    categorical_analysis
)

def create_pipeline(**kwargs) -> Pipeline:
    return pipeline(
        [
            node(
                func=calculate_summary_statistics,
                inputs="preprocessed_csgo_data",
                outputs={
                    "numeric_summary": "numeric_summary_statistics",
                    "categorical_summary": "categorical_summary_statistics"
                },
                name="calculate_summary_statistics_node",
            ),
            node(
                func=create_distribution_plots,
                inputs="preprocessed_csgo_data",
                outputs="distribution_plots",
                name="create_distribution_plots_node",
            ),
            node(
                func=correlation_analysis,
                inputs="preprocessed_csgo_data",
                outputs=["correlation_matrix", "correlation_plot"],
                name="correlation_analysis_node",
            ),
            node(
                func=categorical_analysis,
                inputs="preprocessed_csgo_data",
                outputs="categorical_plots",
                name="categorical_analysis_node",
            ),
        ]
    )
