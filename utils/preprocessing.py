# utils/preprocessing.py

"""
=================================================
DATA PREPROCESSING FOR RAGAS
=================================================
File ini bertanggung jawab untuk:
- Membersihkan data
- Mengonversi DataFrame ke format RAGAS
- Validasi struktur dataset

File ini TIDAK:
- Menjalankan evaluasi
- Memanggil LLM
=================================================
"""

import ast
import pandas as pd
from datasets import Dataset
from typing import List


def parse_contexts(contexts) -> List[str]:
    """
    =================================================
    Parse Contexts
    -------------------------------------------------
    Mengonversi kolom contexts menjadi List[str].

    Digunakan jika contexts disimpan sebagai:
    - string JSON
    - string Python list
    - single string

    Input  :
        - contexts (str | list)

    Output :
        - List[str]
    =================================================
    """

    if isinstance(contexts, list):
        return contexts

    if isinstance(contexts, str):
        try:
            parsed = ast.literal_eval(contexts)
            if isinstance(parsed, list):
                return parsed
        except Exception:
            return [contexts]

    return []


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    =================================================
    Clean DataFrame
    -------------------------------------------------
    Membersihkan dan menstandarkan dataset.

    Proses:
    - Drop baris kosong
    - Normalisasi kolom contexts

    Input  :
        - df (pd.DataFrame)

    Output :
        - pd.DataFrame (clean)
    =================================================
    """

    df = df.dropna(subset=["question", "answer"])
    df["contexts"] = df["contexts"].apply(parse_contexts)

    return df


def dataframe_to_ragas_dataset(df: pd.DataFrame) -> Dataset:
    """
    =================================================
    Convert DataFrame to RAGAS Dataset
    -------------------------------------------------
    Mengonversi pandas DataFrame ke HuggingFace Dataset
    dengan format yang kompatibel dengan RAGAS.

    Input  :
        - df (pd.DataFrame)

    Output :
        - datasets.Dataset
    =================================================
    """

    required_columns = {"question", "contexts", "answer"}
    if not required_columns.issubset(df.columns):
        raise ValueError(
            f"Dataset must contain columns: {required_columns}"
        )

    return Dataset.from_pandas(df.reset_index(drop=True))
