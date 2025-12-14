# utils/chunking/fixed_size_chunking.py

"""
=================================================
FIXED SIZE CHUNKING STRATEGY
=================================================
Strategi chunking paling dasar dengan memotong
teks berdasarkan jumlah token atau karakter tetap.

Digunakan sebagai:
- Baseline retriever
- Pembanding untuk chunking berbasis semantik
=================================================
"""

from typing import List


def fixed_size_chunking(
    text: str,
    chunk_size: int = 512,
    overlap: int = 50,
) -> List[str]:
    """
    =================================================
    Fixed Size Chunking
    -------------------------------------------------
    Memecah teks menjadi potongan berukuran tetap
    dengan optional overlap antar chunk.

    Input  :
        - text (str)
        - chunk_size (int)
            jumlah karakter per chunk
        - overlap (int)
            jumlah karakter overlap antar chunk

    Output :
        - List[str] (chunks)
    =================================================
    """

    if not text:
        return []

    chunks = []
    start = 0
    text_length = len(text)

    while start < text_length:
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap

    return chunks
