# utils/visualization.py

"""
=================================================
EVALUATION VISUALIZATION UTILITIES
=================================================
File ini menangani visualisasi hasil evaluasi RAGAS.

Fungsi di sini:
- Mengubah skor evaluasi menjadi grafik
- Membantu interpretasi performa model

File ini TIDAK:
- Menjalankan evaluasi
- Mengubah data mentah
=================================================
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


def plot_metric_bar(result_df: pd.DataFrame):
    """
    =================================================
    Bar Chart for RAG Metrics
    -------------------------------------------------
    Menampilkan bar chart untuk setiap metrik RAGAS.

    Input  :
        - result_df (pd.DataFrame)
          DataFrame hasil evaluasi RAGAS

    Output :
        - None (plot ditampilkan)
    =================================================
    """

    mean_scores = result_df.mean()

    plt.figure()
    mean_scores.plot(kind="bar")
    plt.title("RAG Evaluation Metrics (Mean Score)")
    plt.ylabel("Score")
    plt.ylim(0, 1)
    plt.show()


def plot_metric_radar(result_df: pd.DataFrame):
    """
    =================================================
    Radar Chart for RAG Metrics
    -------------------------------------------------
    Menampilkan radar chart untuk membandingkan
    performa antar metrik RAG.

    Input  :
        - result_df (pd.DataFrame)

    Output :
        - None (plot ditampilkan)
    =================================================
    """

    labels = result_df.columns.tolist()
    scores = result_df.mean().values.tolist()

    scores += scores[:1]
    angles = np.linspace(0, 2 * np.pi, len(labels) + 1)

    plt.figure()
    ax = plt.subplot(111, polar=True)
    ax.plot(angles, scores)
    ax.fill(angles, scores, alpha=0.25)

    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0, 1)
    plt.title("RAG Evaluation Radar Chart")
    plt.show()
