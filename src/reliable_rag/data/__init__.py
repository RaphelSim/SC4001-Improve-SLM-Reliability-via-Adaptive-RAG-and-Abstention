"""Dataset and split helpers will be implemented by the data/retrieval workstream."""
"""Dataset normalization and controlled evidence construction."""

from reliable_rag.data.evidence_conditions import construct_evidence_set
from reliable_rag.data.hotpotqa import HotpotQAExample, load_hotpotqa, normalize_hotpotqa_example
from reliable_rag.data.preparation import prepare_split
from reliable_rag.data.splits import SplitManifestRow, build_split_manifest, write_split_manifest

__all__ = [
    "HotpotQAExample",
    "SplitManifestRow",
    "build_split_manifest",
    "construct_evidence_set",
    "load_hotpotqa",
    "normalize_hotpotqa_example",
    "prepare_split",
    "write_split_manifest",
]
