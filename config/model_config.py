# config/model_config.py

"""
=================================================
MODEL CONFIGURATION
=================================================
File ini berisi konfigurasi seluruh model yang digunakan
dalam sistem evaluasi RAG menggunakan RAGAS.

Konfigurasi ini DIPISAHKAN dari logic agar:
- Mudah mengganti model
- Reproducible untuk eksperimen
- Jelas secara metodologi (skripsi-safe)
=================================================
"""

from dataclasses import dataclass


# =================================================
# Generator Model Configuration
# =================================================
@dataclass
class GeneratorModelConfig:
    """
    =================================================
    Generator Model Config
    -------------------------------------------------
    Konfigurasi model LLM utama yang bertugas
    menghasilkan jawaban (output generator).

    Dalam konteks RAG:
    - Model ini menerima question + context
    - Menghasilkan final answer

    Input  :
        - prompt (question + retrieved context)
    Output :
        - generated answer (string)
    =================================================
    """

    model_name: str
    temperature: float
    max_tokens: int
    device: str


GENERATOR_MODEL = GeneratorModelConfig(
    model_name="google/gemma-3-1b-it",  # LLM utama (local)
    temperature=0.2,                    # rendah agar stabil & faktual
    max_tokens=512,
    device="cuda",                      # ganti ke "cpu" jika perlu
)


# =================================================
# Embedding / Encoder Model Configuration
# =================================================
@dataclass
class EmbeddingModelConfig:
    """
    =================================================
    Embedding Model Config
    -------------------------------------------------
    Konfigurasi model encoder untuk mengubah teks
    menjadi vektor embedding.

    Digunakan untuk:
    - Retrieval
    - Semantic similarity
    - Context matching

    Input  :
        - text (string / chunk)
    Output :
        - vector embedding (List[float])
    =================================================
    """

    model_name: str
    normalize_embeddings: bool


EMBEDDING_MODEL = EmbeddingModelConfig(
    model_name="sonoisa/sentence-bert-base-ja-mean-tokens",
    normalize_embeddings=True,
)


# =================================================
# Evaluator / Judge Model Configuration
# =================================================
@dataclass
class EvaluatorModelConfig:
    """
    =================================================
    Evaluator (Judge) Model Config
    -------------------------------------------------
    Konfigurasi model LLM yang bertugas sebagai
    evaluator (judge) untuk metrik berbasis LLM.

    Model ini TIDAK menghasilkan jawaban user,
    melainkan memberikan penilaian terhadap:
    - Faithfulness
    - Context relevancy
    - Noise sensitivity
    dll.

    Input  :
        - question
        - context
        - generated answer
    Output :
        - score (float)
        - reasoning (opsional)
    =================================================
    """

    model_name: str
    temperature: float
    max_tokens: int
    device: str


EVALUATOR_MODEL = EvaluatorModelConfig(
    model_name="Qwen/Qwen2.5-14B-Instruct",  # qwen-14B judge
    temperature=0.0,                         # HARUS stabil
    max_tokens=512,
    device="cuda",
)
