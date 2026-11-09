"""MuSiQue adapter preserves original support and deterministic variable context."""

from collections import Counter

from reliable_rag.config import load_settings
from reliable_rag.data.musique import GOLD_TARGETS, select_musique
from reliable_rag.evaluation.metrics import supporting_evidence_recall_at_k


def _rows():
    for gold_count in (2, 3, 4):
        for number in range(40):
            yield {
                "id": f"{gold_count}hop-{number}",
                "question": f"Question {gold_count}-{number}?",
                "answer": "Answer",
                "answerable": True,
                "paragraphs": [
                    {
                        "idx": index,
                        "title": f"Title {index}",
                        "paragraph_text": f"Evidence {gold_count}-{number}-{index}.",
                        "is_supporting": index < gold_count,
                    }
                    for index in range(20)
                ],
            }


def test_stratified_selection_preserves_gold_and_context_order():
    questions, evidence, contexts, rejected = select_musique(_rows(), seed=88)
    second = select_musique(_rows(), seed=88)
    assert (questions, evidence, contexts, rejected) == second
    assert len(questions) == len(contexts) == 100
    assert len({q.question_id for q in questions}) == 100
    assert Counter(len(q.gold_paragraph_ids) for q in questions) == GOLD_TARGETS
    assert Counter(item["density_band"] for item in contexts) == {
        "low": 33,
        "medium": 33,
        "high": 34,
    }
    assert len({chunk.evidence_id for chunk in evidence}) == len(evidence)
    for question, context in zip(questions, contexts, strict=True):
        gold = set(context["gold_paragraph_ids"])
        distractors = set(context["distractor_paragraph_ids"])
        assert len(gold) == int(question.question_id[0])
        assert gold == set(question.gold_paragraph_ids)
        assert not gold.intersection(distractors)
        assert 6 <= len(distractors) <= min(18, 20 - len(gold))
        assert set(question.candidate_evidence_ids) == gold | distractors
        assert question.candidate_evidence_ids == context["candidate_paragraph_ids_in_order"]
        assert len(question.candidate_evidence_ids) == len(set(question.candidate_evidence_ids))
        assert set(question.metadata["original_gold_paragraph_indices"]) == set(range(len(gold)))
        recall = supporting_evidence_recall_at_k(
            question.gold_paragraph_ids,
            question.candidate_evidence_ids,
            len(question.candidate_evidence_ids),
        )
        assert recall.numerator == recall.denominator == len(gold)


def test_musique_paths_are_separate_from_existing_hotpotqa_paths():
    hotpot = load_settings("configs/retrieval_study_v3.yaml")
    musique = load_settings("configs/musique_retrieval_study_v3.yaml")
    assert str(hotpot.dataset_dir) == "data/processed/hotpotqa/subset-1000-seed88"
    assert str(hotpot.index_path) == "cache/hotpotqa-subset-1000-seed88.npz"
    assert str(musique.dataset_dir) == "data/processed/musique/subset-100-seed88"
    assert str(musique.index_path) == "cache/musique-subset-100-seed88.npz"
