# Project scope

## Research question

**Can adaptive retrieval and abstention improve the reliability of a small language model when retrieved evidence is incomplete?**

## Problem and intended pipeline

The controlled study creates cases where retrieval initially returns complete, partially incomplete, or fully incomplete supporting evidence. The system assesses whether the evidence can support an answer. When evidence is insufficient, it may use one targeted follow-up retrieval and reassess the merged evidence. It answers when supported and abstains when the final controlled evidence remains insufficient.

The central path is **incomplete evidence → detection → adaptive evidence recovery → answer or abstain**. Reliability and evidence support are primary; inference cost is secondary.

## Foundation scope

This repository stage establishes shared data records, enums, module interfaces, configuration, JSONL and schema validation, synthetic fixtures, tests, development commands, and team documentation. Research components are stubs only.

## Excluded at this stage

Real HotpotQA download/preprocessing, embedding generation, cosine or FAISS retrieval, Qwen or other SLM loading, MLX/Transformers inference, final research prompts, the adaptive algorithm, finalized experimental metrics, plotting, GraphRAG, and fine-tuning are reserved for later member work.

## Target execution environments

The common baseline is Python 3.11 on native ARM Apple Silicon and Google Colab/Linux. MLX is optional and macOS-specific; core contracts do not import or require it.
