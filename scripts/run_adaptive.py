#!/usr/bin/env python3

"""Run adaptive retrieval (with or without abstention) over prepared questions."""


# Per Question:
# -> round-0 retrieval -> AdaptiveRetrievalController.run()
# -> results.jsonl (ExperimentRecord) and errors.jsonl (model failures)

# --baseline-results reuses saved rag_abstention round-0 decisions, so the model is called
# again where the baseline needs more evidence or had no valid decision.

import argparse
import json
import logging
from datetime import UTC, datetime
from pathlib import Path

from reliable_rag.adaptive import AdaptiveRetrievalController, label_evidence
from reliable_rag.config import load_settings
from reliable_rag.contracts.enums import RetrievalStage, SystemVariant
from reliable_rag.contracts.records import (
    EvidenceChunk,
    ExperimentRecord,
    QuestionRecord,
    RetrievalRequest,
    RetrievalResult,
)
from reliable_rag.io.artifacts import file_sha256
from reliable_rag.io.jsonl import read_jsonl, write_jsonl
from reliable_rag.model.abstention import RAGAbstentionOutcome
from reliable_rag.run_artifacts import RunWriter
from reliable_rag.runtime import create_model_components, create_retriever

ADAPTIVE_VARIANTS = {
    SystemVariant.ADAPTIVE_RETRIEVAL,
    SystemVariant.ADAPTIVE_RETRIEVAL_ABSTENTION,
}


def load_baseline_outcomes(
    path: Path, settings, retrieval: list[RetrievalResult]
) -> dict[str, RAGAbstentionOutcome]:
    """Return saved round-0 outcomes keyed by Question ID, after checking they are reusable."""

    metadata_path = next(
        (
            path.parent / name
            for name in ("run_metadata.json", "baseline.manifest.json")
            if (path.parent / name).exists()
        ),
        None,
    )

    if metadata_path is None:
        raise ValueError("Baseline results need run_metadata.json or baseline.manifest.json")

    baseline_config = json.loads(metadata_path.read_text(encoding="utf-8"))["config"]
    current_config = settings.model_dump(mode="json")

    if baseline_config["slm"] != current_config["slm"]:
        raise ValueError("Baseline SLM settings differ, rerun round 0 instead of reusing it")

    if baseline_config["retrieval"]["initial_top_k"] != settings.retrieval.initial_top_k:
        raise ValueError("Baseline initial top-k differs from configured value")

    retrieved_ids = {r.question_id: [c.evidence_id for c in r.evidence] for r in retrieval}
    outcomes = {}

    for record in read_jsonl(path, ExperimentRecord):
        question_id = record.question.question_id

        if record.system_variant != SystemVariant.RAG_ABSTENTION or len(record.rounds) != 1:
            raise ValueError("Baseline results must be single-round rag_abstention records")

        if question_id not in retrieved_ids:
            continue

        saved_ids = [c.evidence_id for c in record.rounds[0].retrieval.evidence]

        if saved_ids != retrieved_ids[question_id]:
            raise ValueError(f"Baseline round-0 evidence differs for question {question_id}")

        outcomes[question_id] = RAGAbstentionOutcome(
            sufficiency_decision=record.rounds[0].sufficiency_decision,
            answer=record.final_answer,
        )
    return outcomes


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("configs/base.yaml"))
    parser.add_argument(
        "--run-id", help="Unique output directory name (default: experiment.run_id)"
    )
    parser.add_argument(
        "--variant",
        choices=[variant.value for variant in ADAPTIVE_VARIANTS],
        help="Override experiment.system_variant",
    )
    parser.add_argument("--questions", type=Path, help="Override derived prepared questions path")
    parser.add_argument("--evidence", type=Path, help="Override derived evidence path")
    parser.add_argument("--index", type=Path, help="Override derived embedding index path")
    parser.add_argument(
        "--retrieval", type=Path, help="Use aligned initial RetrievalResult JSONL for round 0"
    )
    parser.add_argument(
        "--baseline-results",
        type=Path,
        help="Reuse round-0 decisions from a rag_abstention results JSONL",
    )
    parser.add_argument("--limit", type=int, help="Use first N questions from the prepared subset")
    parser.add_argument(
        "--force", action="store_true", help="Replace this run directory, including results"
    )

    args = parser.parse_args()
    settings = load_settings(args.config)

    if args.variant:
        settings.experiment.system_variant = SystemVariant(args.variant)
    settings = type(settings).model_validate(settings.model_dump())

    if settings.experiment.system_variant not in ADAPTIVE_VARIANTS:
        raise ValueError("This runner supports only the two adaptive variants, use --variant")

    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")

    run_id = args.run_id or settings.experiment.run_id
    settings.experiment.run_id = run_id
    questions_path = args.questions or settings.questions_path
    evidence_path = args.evidence or settings.evidence_path
    index_path = args.index or settings.index_path
    questions = read_jsonl(questions_path, QuestionRecord)

    if not questions or len({q.question_id for q in questions}) != len(questions):
        raise ValueError("Questions must be nonempty and have unique IDs")

    if args.limit:
        questions = questions[: args.limit]

    # round 1 always searches, so the evidence corpus and index are always inputs
    inputs = [questions_path, evidence_path, index_path]
    index_metadata = index_path.with_suffix(index_path.suffix + ".json")

    if index_metadata.exists():
        inputs.append(index_metadata)

    inputs.extend(path for path in (args.retrieval, args.baseline_results) if path)
    writer = RunWriter(settings, run_id, inputs=inputs, force=args.force)
    logger = logging.getLogger("reliable_rag")
    logger.addHandler(writer)
    writer.metadata["scope"] = "two_round_adaptive_retrieval"
    summary = writer.metadata["summary"]
    summary.update(planned=len(questions), round0_reused=0, two_round_records=0)
    writer.metadata["limit"] = args.limit
    writer.metadata["questions_path"] = str(questions_path)
    current_question = None

    try:
        writer.save()
        evidence = read_jsonl(evidence_path, EvidenceChunk)
        retriever = create_retriever(settings, evidence, index_path, questions=questions)

        if args.retrieval:
            retrieval = read_jsonl(args.retrieval, RetrievalResult)[: len(questions)]
        else:
            retrieval = [
                retriever.retrieve(
                    RetrievalRequest(
                        question_id=question.question_id,
                        query=question.question,
                        stage=RetrievalStage.INITIAL,
                        top_k=settings.retrieval.initial_top_k,
                    )
                )
                for question in questions
            ]
        if [q.question_id for q in questions] != [r.question_id for r in retrieval]:
            raise ValueError("Question and retrieval IDs/counts must align in order")
        for question, initial in zip(questions, retrieval, strict=True):
            if (
                initial.stage != RetrievalStage.INITIAL
                or initial.top_k != settings.retrieval.initial_top_k
            ):
                raise ValueError("Round-0 retrieval must match configured initial top-k")
            if initial.query != question.question:
                raise ValueError("Round-0 retrieval query does not match the question")
        retrieval_path = writer.directory / "retrieval.jsonl"
        write_jsonl(retrieval_path, retrieval)
        writer.metadata["retrieval_sha256"] = file_sha256(retrieval_path)
        reusable = (
            load_baseline_outcomes(args.baseline_results, settings, retrieval)
            if args.baseline_results
            else {}
        )
        components = create_model_components(settings)
        writer.metadata["local_model_path"] = str(components.client.model_path)
        controller = AdaptiveRetrievalController(
            retriever,
            components.assessor,
            components.answer_generator,
            allow_abstention=(
                settings.experiment.system_variant == SystemVariant.ADAPTIVE_RETRIEVAL_ABSTENTION
            ),
            followup_top_k=settings.retrieval.followup_top_k,
        )

        for question, initial in zip(questions, retrieval, strict=True):
            current_question = question.question_id
            summary["attempted"] += 1
            writer.save()
            reused = reusable.get(question.question_id)
            result = controller.run(question, initial, run_id=run_id, initial_outcome=reused)
            summary["round0_reused"] += reused is not None
            if result.experiment is not None:
                result.experiment.metadata.update(
                    {
                        "model_id": settings.slm.model_id,
                        "model_revision": settings.slm.revision,
                        "prompt_version": settings.slm.prompt_version,
                        "round0_reused": reused is not None,
                    }
                )
                writer.append("results", result.experiment.model_dump(mode="json"))
                summary["records_written"] += 1
                summary["two_round_records"] += len(result.experiment.rounds) == 2
                summary[
                    "recorded_abstained" if result.final_answer.abstained else "recorded_answered"
                ] += 1
            else:
                summary["unrecorded_assessment_failures"] += 1
            gold_sufficient = label_evidence(question, initial.evidence).gold_sufficient
            for model_error in result.errors:
                error: dict[str, object] = {
                    **model_error.as_dict(),
                    "category": "model_contract_failure",
                    "record_written": result.experiment is not None,
                    "final_answer": result.final_answer.model_dump(mode="json"),
                    "gold_sufficient": gold_sufficient,
                }
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
