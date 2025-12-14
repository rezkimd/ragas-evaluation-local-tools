# utils/metrics.py

"""
=================================================
RAGAS METRICS UTILITY
=================================================
File ini bertanggung jawab untuk:
- Mendefinisikan metrik evaluasi RAG
- Mengelompokkan metrik sesuai kebutuhan eksperimen
- Menjadi abstraction layer antara config dan evaluator

File ini TIDAK:
- Memanggil LLM
- Menjalankan evaluasi
=================================================
"""

from typing import List
from ragas.metrics.base import Metric


def load_ragas_metrics(metrics_config) -> List[Metric]:
    """
    =================================================
    Load RAGAS Metrics
    -------------------------------------------------
    Fungsi ini mengambil konfigurasi metrik dari
    ragas_config dan mengembalikan daftar metrik
    yang siap digunakan oleh evaluator.

    Input  :
        - metrics_config (RagasMetricsConfig)
            berisi list metrik RAGAS

    Output :
        - List[Metric]
            daftar objek metrik RAGAS
    =================================================
    """

    if not metrics_config.metrics:
        raise ValueError("RAGAS metrics list is empty")

    return metrics_config.metrics
