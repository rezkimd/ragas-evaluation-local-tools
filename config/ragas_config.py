# config/ragas_config.py

"""
=================================================
RAGAS EVALUATION CONFIGURATION
=================================================
File ini berisi konfigurasi evaluasi RAG menggunakan
framework RAGAS.

Konfigurasi ini mengatur:
- Metrik evaluasi yang digunakan
- Mode eksekusi evaluasi
- Penggunaan tracing (LangSmith)
=================================================
"""

from dataclasses import dataclass
from typing import List

from ragas.metrics.collections import (
    faithfulness,
    answer_relevancy,
    context_recall,
    context_precision,
    context_entity_recall,
    answer_similarity,
    answer_correctness,
    # noise_sensitivity,
)


# =================================================
# RAGAS Metrics Configuration
# =================================================
@dataclass
class RagasMetricsConfig:
    """
    =================================================
    RAGAS Metrics Config
    -------------------------------------------------
    Menentukan metrik evaluasi RAG yang digunakan.

    Semua metrik di sini bersifat LLM-based dan
    membutuhkan judge model.

    Input  :
        - dataset (question, context, answer)
    Output :
        - skor evaluasi per metrik
    =================================================
    """

    metrics: List


RAGAS_METRICS = RagasMetricsConfig(
    metrics=[
        context_precision,
        context_recall,
        context_entity_recall,
        answer_relevancy,
        answer_similarity,
        answer_correctness,
        # noise_sensitivity,
        faithfulness,
    ]
)


# =================================================
# RAGAS Runtime Configuration
# =================================================
@dataclass
class RagasRuntimeConfig:
    """
    =================================================
    RAGAS Runtime Config
    -------------------------------------------------
    Mengatur bagaimana evaluasi RAGAS dijalankan.

    Input  :
        - dataset evaluasi
    Output :
        - hasil evaluasi (DataFrame)
    =================================================
    """

    batch_size: int
    enable_tracing: bool
    experiment_name: str


RAGAS_RUNTIME = RagasRuntimeConfig(
    batch_size=4,                     # kecil agar stabil
    enable_tracing=True,              # aktifkan LangSmith
    experiment_name="ragas-rag-eval",
)
