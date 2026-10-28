"""Plug-compatible module interfaces; concrete research implementations are future work."""

from typing import Protocol, runtime_checkable

from reliable_rag.contracts.records import (
    AnswerResult,
    EvidenceSet,
    ExperimentRecord,
    QuestionRecord,
    RetrievalRequest,
    RetrievalResult,
    SufficiencyDecision,
)


@runtime_checkable
class Retriever(Protocol):
    """Return up to the requested number of evidence chunks for one retrieval call."""

    def retrieve(self, request: RetrievalRequest) -> RetrievalResult:
        """Retrieve evidence matching a request, preserving the question and stage."""
        ...


@runtime_checkable
class SufficiencyAssessor(Protocol):
    """Assess whether the supplied evidence supports answering the question."""

    def assess(self, question: QuestionRecord, evidence: EvidenceSet) -> SufficiencyDecision:
        """Return a structured sufficiency decision without hidden reasoning text."""
        ...


@runtime_checkable
class AnswerGenerator(Protocol):
    """Generate a grounded answer from supplied evidence or return an abstention."""

    def generate(self, question: QuestionRecord, evidence: EvidenceSet) -> AnswerResult:
        """Return an answer result and identify evidence used by the answering component."""
        ...


@runtime_checkable
class AdaptiveController(Protocol):
    """Coordinate implementations through interfaces; the research algorithm is unimplemented."""

    def run(
        self,
        question: QuestionRecord,
        initial_evidence: EvidenceSet,
        *,
        retriever: Retriever,
        assessor: SufficiencyAssessor,
        answer_generator: AnswerGenerator,
    ) -> ExperimentRecord:
        """Execute a bounded case and return its raw experiment record."""
        ...
