# Semantic-Preserving Summarization

This repository contains the implementation of a framework for **token-efficient and semantic-preserving opinion summarization** with **Large Language Models (LLMs)**.  
The goal is to generate compact and balanced summaries of large collections of user opinions while preserving the diversity of viewpoints expressed in the original corpus.

The framework combines:

- **multidimensional classification** of opinions (e.g., sentiment, topic, emotion, and optional domain-specific facets),
- **distribution-aware stratified sampling** to select representative subsets of opinions,
- **LLM-based summarization** guided by facet-aware prompts.

By selecting a compact but semantically representative subset before generation, the framework reduces the number of input tokens processed by the LLM while maintaining high semantic fidelity and topic coverage.

---

## Overview

Opinion-rich corpora such as product reviews, hotel evaluations, and political discussions often contain:

- large volumes of redundant text,
- imbalanced distributions of viewpoints,
- minority opinions that are easily overlooked.

Directly summarizing the full corpus with an LLM may be expensive, inefficient, and biased toward dominant views.  
This repository implements a pipeline that addresses these issues by structuring opinions across multiple semantic dimensions and selecting a balanced subset before summarization.

---

## Main Features

- Multidimensional opinion analysis:
  - sentiment
  - topic
  - emotion
  - optional domain-specific facets

- Stratified sampling strategies:
  - **Knapsack**
  - **Knapsack-KL**
  - **KDE**

- Token-efficient summarization with LLMs

- Evaluation tools for:
  - topic coverage
  - summary-level semantic similarity
  - token usage

---

## Repository Structure

A possible high-level organization of the repository is:

```text
.
├── data/                 # Input datasets or dataset loaders
├── preprocessing/        # Data cleaning and preparation scripts
├── classification/       # Sentiment, emotion, topic, and facet classification
├── sampling/             # Stratified sampling algorithms
├── summarization/        # Prompting and LLM summarization
├── evaluation/           # Metrics and analysis scripts
├── figures/              # Plots and visual outputs
├── notebooks/            # Optional exploratory notebooks
├── results/              # Experimental outputs
└── README.md
