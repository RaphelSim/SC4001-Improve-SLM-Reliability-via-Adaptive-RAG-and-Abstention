"""Component construction for baseline tools and future adaptive controllers."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

from reliable_rag.config import Settings
from reliable_rag.contracts.records import EvidenceChunk, QuestionRecord
from reliable_rag.model import (
    MLXLMClient,
    SLMAnswerGenerator,
    SLMSufficiencyAssessor,
    TransformersClient,
)
from reliable_rag.retrieval import DenseRetriever, SentenceTransformerEmbedder, load_embedding_cache
from reliable_rag.retrieval.embedder import validate_embedding_cache


@dataclass(frozen=True)
class ModelComponents:
    """Share one loaded model across independent assessor/answer interfaces."""

    client: MLXLMClient | TransformersClient
    assessor: SLMSufficiencyAssessor
    answer_generator: SLMAnswerGenerator


def create_model_components(
    settings: Settings, *, allow_model_download: bool = False
) -> ModelComponents:
    client: MLXLMClient | TransformersClient
    if settings.slm.runtime.lower() == "transformers":
        client = TransformersClient.from_settings(
            settings, allow_model_download=allow_model_download
        )
    else:
        client = MLXLMClient.from_settings(settings, allow_model_download=allow_model_download)
    return ModelComponents(
        client=client,
        assessor=SLMSufficiencyAssessor(client, settings),
        answer_generator=SLMAnswerGenerator(client, settings),
    )


def create_embedder(
    settings: Settings, *, allow_model_download: bool = False
) -> SentenceTransformerEmbedder:
    return SentenceTransformerEmbedder(
        settings.retrieval.embedding_model,
        revision=settings.retrieval.embedding_revision,
        device=settings.retrieval.embedding_device,
        cache_root=settings.cache_root,
        allow_model_download=allow_model_download,
    )


def create_retriever(
    settings: Settings,
    evidence: Sequence[EvidenceChunk],
    embedding_cache: Path,
    *,
    allow_model_download: bool = False,
    questions: Sequence[QuestionRecord] | None = None,
) -> DenseRetriever:
    """Build a reusable retriever only after corpus/model provenance checks pass."""
    if settings.retrieval.similarity_metric != "cosine":
        raise ValueError("DenseRetriever supports similarity_metric=cosine")
    if questions is not None and len({q.question_id for q in questions}) != len(questions):
        raise ValueError("question IDs must be unique when configuring retrieval scopes")
    ids, vectors, metadata = load_embedding_cache(embedding_cache)
    validate_embedding_cache(
        evidence,
        ids,
        vectors,
        metadata,
        model_name=settings.retrieval.embedding_model,
        revision=settings.retrieval.embedding_revision,
    )
    return DenseRetriever(
        evidence,
        create_embedder(settings, allow_model_download=allow_model_download),
        embeddings=vectors,
        candidate_evidence_ids_by_question={
            question.question_id: question.candidate_evidence_ids for question in questions
        }
        if questions is not None
        else None,
    )
