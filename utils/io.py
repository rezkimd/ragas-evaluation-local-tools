# utils/io.py

"""
=================================================
DATA INPUT / OUTPUT UTILITIES
=================================================
File ini menangani:
- Load dataset evaluasi
- Simpan hasil evaluasi

Tidak melakukan:
- Preprocessing
- Evaluasi
- Transformasi semantik
=================================================
"""

import pandas as pd
from typing import Union


def load_csv_dataset(file_path: str) -> pd.DataFrame:
    """
    =================================================
    Load CSV Dataset
    -------------------------------------------------
    Membaca dataset evaluasi dari file CSV.

    Format CSV yang diharapkan:
        - question
        - contexts      (list / string)
        - answer
        - ground_truth  (opsional)

    Input  :
        - file_path (str)

    Output :
        - pd.DataFrame
    =================================================
    """

    df = pd.read_csv(file_path)

    if "question" not in df.columns:
        raise ValueError("Column 'question' not found in dataset")

    return df


def save_evaluation_result(
    result_df: pd.DataFrame,
    output_path: str,
):
    """
    =================================================
    Save Evaluation Result
    -------------------------------------------------
    Menyimpan hasil evaluasi RAGAS ke file CSV.

    Input  :
        - result_df (pd.DataFrame)
        - output_path (str)

    Output :
        - None (file saved)
    =================================================
    """

    result_df.to_csv(output_path, index=False)
