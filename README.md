# Semantic-Preserving Summarization

This repository contains the implementation of a framework for **token-efficient and semantic-preserving opinion summarization** with **Large Language Models (LLMs)**.

The goal is to generate **compact, balanced, and semantically faithful summaries** of large collections of user opinions while preserving the diversity of viewpoints expressed in the original corpus.

The framework combines:

- **multidimensional classification** of opinions (e.g., sentiment, topic, emotion, and optional domain-specific facets),
- **distribution-aware stratified sampling** to select representative subsets of opinions,
- **LLM-based summarization** guided by facet-aware prompts.

By selecting a compact but semantically representative subset before generation, the framework reduces the number of input tokens processed by the LLM while maintaining high semantic fidelity and topic coverage.

---

## Overview

Opinion-rich corpora such as product reviews, hotel evaluations, and political discussions often contain:

- large amounts of redundant text,
- imbalanced distributions of viewpoints,
- minority opinions that are easily overlooked.

Directly summarizing the full corpus with an LLM may be expensive, inefficient, and biased toward dominant views.  
This repository implements a pipeline that addresses these issues by structuring opinions across multiple semantic dimensions and selecting a balanced subset before summarization.

---

## Main Features

- **Multidimensional opinion analysis**
  - sentiment
  - topic
  - emotion
  - optional domain-specific facets

- **Stratified sampling strategies**
  - `Knapsack`
  - `Knapsack-KL`
  - `KDE`

- **LLM-based summarization**
  - generation from compact, representative subsets
  - facet-aware prompting

- **Evaluation tools**
  - topic coverage
  - summary-level semantic similarity
  - token usage

---

## Repository Structure

The repository is currently organized as follows:

```text
.
├── classifiers/            # Modules for multidimensional opinion classification
├── stratifiers/            # Stratified sampling algorithms
├── utils/                  # Shared utility functions
├── Main_full_pipeline.py   # Main script for running the full pipeline
└── README.md
```

## Contact

For questions, collaborations, or further information: 

Fabrizio Marozzo (University of Calabria), Email: fmarozzo@dimes.unical.it
