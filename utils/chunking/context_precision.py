# utils/metrics/context_precision.py
"""
Context Precision wrapper utility.

- Uses RAGAS v0.4.x official implementation
- Does NOT modify RAGAS internal code
- Provides a stable async interface for local evaluation
"""

from __future__ import annotations

import logging
from typing import List, Optional

from ragas.metrics.collections.context_precision.metric import (
    ContextPrecisionWithReference,
    ContextPrecisionWithoutReference,
)
from ragas.metrics.result import MetricResult

logger = logging.getLogger(__name__)


class ContextPrecisionEvaluator:
    """
    Wrapper class for RAGAS Context Precision metrics.

    This class abstracts:
    - metric initialization
    - input validation
    - logging & error isolation
    """

    def __init__(self, llm):
        """
        Initialize evaluator.

        Args:
            llm: InstructorBaseRagasLLM (from ragas.llms.llm_factory)
        """
        self.llm = llm
        self._metric_with_ref = ContextPrecisionWithReference(llm=llm)
        self._metric_without_ref = ContextPrecisionWithoutReference(llm=llm)

    async def evaluate(
        self,
        question: str,
        retrieved_contexts: List[str],
        reference: Optional[str] = None,
        response: Optional[str] = None,
    ) -> float:
        """
        Evaluate context precision.

        Args:
            question: User query
            retrieved_contexts: List of retrieved chunks
            reference: Ground truth answer (preferred)
            response: Generated answer (fallback)

        Returns:
            Context precision score (0.0 – 1.0)
        """
        self._validate_inputs(question, retrieved_contexts, reference, response)

        try:
            if reference is not None:
                logger.info("Running Context Precision WITH reference")
                result: MetricResult = await self._metric_with_ref.ascore(
                    user_input=question,
                    reference=reference,
                    retrieved_contexts=retrieved_contexts,
                )
            else:
                logger.info("Running Context Precision WITHOUT reference")
                result: MetricResult = await self._metric_without_ref.ascore(
                    user_input=question,
                    response=response,
                    retrieved_contexts=retrieved_contexts,
                )

            logger.info("Context precision score: %.4f", result.value)
            return float(result.value)

        except Exception as e:
            logger.exception("Context Precision evaluation failed")
            raise RuntimeError(f"Context Precision evaluation error: {e}") from e

    @staticmethod
    def _validate_inputs(
        question: str,
        retrieved_contexts: List[str],
        reference: Optional[str],
        response: Optional[str],
    ):
        if not question or not question.strip():
            raise ValueError("question must be a non-empty string")

        if not retrieved_contexts or not all(
            isinstance(c, str) and c.strip() for c in retrieved_contexts
        ):
            raise ValueError("retrieved_contexts must be a non-empty list of strings")

        if reference is None and response is None:
            raise ValueError(
                "Either reference or response must be provided for context precision"
            )
