# utils/logging.py

"""
=================================================
EXPERIMENT LOGGING
=================================================
Mencatat hasil eksperimen evaluasi RAG ke file CSV
untuk keperluan analisis dan reproducibility.
=================================================
"""

import pandas as pd
from datetime import datetime
import os


def log_experiment(
    output_path: str,
    experiment_name: str,
    chunking_strategy: str,
    parameters: dict,
    metric_results: pd.DataFrame,
):
    """
    =================================================
    Log Experiment Result
    -------------------------------------------------
    Menyimpan hasil eksperimen ke file CSV.

    Input  :
        - output_path (str)
        - experiment_name (str)
        - chunking_strategy (str)
        - parameters (dict)
        - metric_results (pd.DataFrame)

    Output :
        - None
    =================================================
    """

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    log_data = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "chunking_strategy": chunking_strategy,
        "parameters": str(parameters),
    }

    for col in metric_results.columns:
        log_data[col] = metric_results[col].mean()

    df_log = pd.DataFrame([log_data])

    if os.path.exists(output_path):
        df_log.to_csv(output_path, mode="a", header=False, index=False)
    else:
        df_log.to_csv(output_path, index=False)
