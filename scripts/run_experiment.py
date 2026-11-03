#!/usr/bin/env python3
"""Run initial retrieval, evidence sufficiency assessment, and answer/abstention."""

import argparse
import logging
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from reliable_rag.config import load_settings
from reliable_rag.contracts.enums import RetrievalStage, SystemVariant
from reliable_rag.contracts.records import (
    EvidenceChunk,
    QuestionRecord,
    RetrievalRequest,
    RetrievalResult,
)
from reliable_rag.io.artifacts import file_sha256
from reliable_rag.io.jsonl import read_jsonl, write_jsonl
from reliable_rag.model.abstention import RAGAbstentionResponder
from reliable_rag.pipeline import run_initial_rag_abstention_case
from reliable_rag.run_artifacts import RunWriter
from reliable_rag.runtime import create_embedder, create_model_components, create_retriever


def retrieve(settings, questions, evidence_path, index_path):
    evidence = read_jsonl(evidence_path, EvidenceChunk)
    embedder = create_embedder(settings)
    retriever = create_retriever(settings, evidence, index_path, questions=questions)
    results = []
    for start in range(0, len(questions), 64):
        batch = questions[start : start + 64]
        vectors = embedder.encode([q.question for q in batch])
        for question, vector in zip(batch, vectors, strict=True):
            request = RetrievalRequest(
                question_id=question.question_id,
                query=question.question,
                stage=RetrievalStage.INITIAL,
                top_k=settings.retrieval.initial_top_k,
            )
            results.append(
                retriever.retrieve_encoded(
                    request,
                    np.asarray(vector),
                    candidate_evidence_ids=question.candidate_evidence_ids or None,
                )
            )
    return results


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument("--subset-size", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--run-id", help="Unique output directory name (default: experiment.run_id)"
    )
    parser.add_argument("--questions", type=Path, help="Override derived prepared questions path")
    parser.add_argument("--evidence", type=Path, help="Override derived evidence path")
    parser.add_argument("--index", type=Path, help="Override derived embedding index path")
    parser.add_argument(
        "--retrieval",
        type=Path,
        help="Use aligned initial RetrievalResult JSONL instead of retrieving",
    )
    parser.add_argument("--limit", type=int, help="Use first N questions from the prepared subset")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace this run directory, including previous results",
    )
    args = parser.parse_args()
    settings = load_settings(args.config)
    if args.subset_size is not None:
        settings.dataset.subset_size = args.subset_size
    if args.seed is not None:
        settings.project_seed = args.seed
    settings = type(settings).model_validate(settings.model_dump())
    if settings.experiment.system_variant != SystemVariant.RAG_ABSTENTION:
        raise ValueError("This runner supports only the rag_abstention baseline")
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    run_id = args.run_id or settings.experiment.run_id
    settings.experiment.run_id = run_id
    questions_path, evidence_path = (
        args.questions or settings.questions_path,
        args.evidence or settings.evidence_path,
    )
    index_path = args.index or settings.index_path
    questions = read_jsonl(questions_path, QuestionRecord)
    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("Questions must be nonempty and have unique IDs")
    if args.limit:
        questions = questions[: args.limit]
    inputs = [questions_path]
    if args.retrieval:
        inputs.append(args.retrieval)
    else:
        inputs.extend(
            [evidence_path, index_path, index_path.with_suffix(index_path.suffix + ".json")]
        )
    if (
        settings.selection_path.exists()
        and questions_path.resolve() == settings.questions_path.resolve()
    ):
        inputs.append(settings.selection_path)
    writer = RunWriter(settings, run_id, inputs=inputs, force=args.force)
    logger = logging.getLogger("reliable_rag")
    logger.addHandler(writer)
    summary = writer.metadata["summary"]
    summary["planned"] = len(questions)
    writer.metadata["limit"] = args.limit
    writer.metadata["questions_path"] = str(questions_path)
    current_question = None
    try:
        writer.save()
        if args.retrieval:
            all_results = read_jsonl(args.retrieval, RetrievalResult)
            retrieval = all_results[: len(questions)]
        else:
            retrieval = retrieve(settings, questions, evidence_path, index_path)
        if [q.question_id for q in questions] != [r.question_id for r in retrieval]:
            raise ValueError("Question and retrieval IDs/counts must align in order")
        for question, result in zip(questions, retrieval, strict=True):
            if (
                result.stage != RetrievalStage.INITIAL
                or result.top_k != settings.retrieval.initial_top_k
            ):
                raise ValueError("Precomputed retrieval must match the configured initial top-k")
            if result.query != question.question:
                raise ValueError("Precomputed retrieval query does not match the question")
        retrieval_path = writer.directory / "retrieval.jsonl"
        write_jsonl(retrieval_path, retrieval)
        writer.metadata["retrieval_sha256"] = file_sha256(retrieval_path)
        components = create_model_components(settings)
        writer.metadata["local_model_path"] = str(components.client.model_path)
        responder = RAGAbstentionResponder(components.assessor, components.answer_generator)
        for question, initial in zip(questions, retrieval, strict=True):
            current_question = question.question_id
            summary["attempted"] += 1
            writer.save()
            result = run_initial_rag_abstention_case(question, initial, responder, run_id=run_id)
            if result.experiment is not None:
                result.experiment.metadata.update(
                    {
                        "model_id": settings.slm.model_id,
                        "model_revision": settings.slm.revision,
                        "prompt_version": settings.slm.prompt_version,
                    }
                )
                writer.append("results", result.experiment.model_dump(mode="json"))
                summary["records_written"] += 1
                summary[
                    "recorded_abstained" if result.outcome.answer.abstained else "recorded_answered"
                ] += 1
            else:
                summary["unrecorded_assessment_failures"] += 1
            if result.outcome.error is not None:
                error = result.outcome.error.as_dict()
                error.update(
                    {
                        "category": "model_contract_failure",
                        "record_written": result.experiment is not None,
                        "final_answer": result.outcome.answer.model_dump(mode="json"),
                        "gold_sufficient": result.evidence.gold_sufficient,
                    }
                )
                writer.append("errors", error)
                summary["model_errors"] += 1
            writer.save()
            logging.info("Completed %s/%s questions", summary["attempted"], summary["planned"])
        writer.metadata["status"] = (
            "completed_with_model_errors" if summary["model_errors"] else "completed"
        )
    except BaseException as exc:
        writer.metadata["status"] = (
            "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed"
        )
        writer.append(
            "errors",
            {
                "category": "runtime_error",
                "question_id": current_question,
                "exception_type": type(exc).__name__,
                "reason": str(exc),
            },
        )
        summary["runtime_errors"] += 1
        raise
    finally:
        writer.metadata["finished_at"] = datetime.now(UTC).isoformat()
        writer.save()
        logger.removeHandler(writer)
        writer.close()
    logging.info("Results saved to %s; summary=%s", writer.directory, summary)
    return 1 if summary["unrecorded_assessment_failures"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
