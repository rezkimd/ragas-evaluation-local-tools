# utils/evaluation.py

"""
=================================================
RAGAS EVALUATION PIPELINE
=================================================
File ini berisi logic utama untuk menjalankan
evaluasi RAG menggunakan RAGAS.

Tanggung jawab:
- Menghubungkan LLM judge
- Menjalankan metrik evaluasi
- Mengatur tracing (LangSmith)
- Menghasilkan hasil evaluasi terstruktur
=================================================
"""

import os
from typing import Any

from ragas import evaluate
from ragas.llms import LangchainLLMWrapper
from langchain.llms import HuggingFacePipeline

from config.model_config import EVALUATOR_MODEL
from config.ragas_config import RAGAS_RUNTIME
from utils.metrics import load_ragas_metrics


def setup_langsmith_tracing(enable: bool, experiment_name: str):
    """
    =================================================
    LangSmith Tracing Setup
    -------------------------------------------------
    Mengaktifkan atau menonaktifkan tracing LangSmith.

    Input  :
        - enable (bool)
        - experiment_name (str)

    Output :
        - None
    =================================================
    """

    if enable:
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGCHAIN_PROJECT"] = experiment_name
    else:
        os.environ["LANGCHAIN_TRACING_V2"] = "false"


def build_evaluator_llm():
    """
    =================================================
    Build Evaluator (Judge) LLM
    -------------------------------------------------
    Membuat LLM evaluator berdasarkan konfigurasi
    di model_config.

    Model ini digunakan oleh RAGAS sebagai judge.

    Input  :
        - None

    Output :
        - LangchainLLMWrapper
    =================================================
    """

    llm_pipeline = HuggingFacePipeline.from_model_id(
        model_id=EVALUATOR_MODEL.model_name,
        task="text-generation",
        model_kwargs={
            "temperature": EVALUATOR_MODEL.temperature,
            "max_new_tokens": EVALUATOR_MODEL.max_tokens,
        },
        device=0 if EVALUATOR_MODEL.device == "cuda" else -1,
    )

    return LangchainLLMWrapper(llm_pipeline)


def run_ragas_evaluation(
    dataset: Any,
    metrics_config,
):
    """
    =================================================
    Run RAGAS Evaluation
    -------------------------------------------------
    Fungsi utama untuk menjalankan evaluasi RAG.

    Input  :
        - dataset
            HuggingFace Dataset atau format RAGAS
            (question, contexts, answer, ground_truth)

        - metrics_config (RagasMetricsConfig)

    Output :
        - result (EvaluationResult)
            berisi skor setiap metrik
    =================================================
    """

    # Setup tracing
    setup_langsmith_tracing(
        enable=RAGAS_RUNTIME.enable_tracing,
        experiment_name=RAGAS_RUNTIME.experiment_name,
    )

    # Load metrics
    metrics = load_ragas_metrics(metrics_config)

    # Build judge model
    evaluator_llm = build_evaluator_llm()

    # Run evaluation
    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=evaluator_llm,
        batch_size=RAGAS_RUNTIME.batch_size,
    )

    return result
