# Multi-Agent LLM Benchmark for Bengali Abstractive Summarization

An automated, reference-free evaluation pipeline that utilizes a compute-efficient multi-agent architecture to systematically compute Faithfulness Scores for Bengali text summaries.

## Overview
Bengali abstractive summarization has historically been hindered by noisy reference datasets. To overcome this, this project establishes a novel factual consistency benchmark by analyzing linguistic error patterns produced by the pre-trained `mT5-base` model on 200 samples from the XL-Sum Bengali test set.

## Methodology
Instead of relying on flawed human reference labels, this pipeline employs a **Multi-Model Consensus Ensemble** (Llama-3-8B, Gemma-2-9B, and Qwen2.5-7B) acting as automated judges. 

The architecture consists of three stages:
1. **Fact Extractor (Atomic Parser):** Extracts core claims from the generated Bengali summary.
2. **Grounding Verifier (Contextual Investigator):** Cross-references claims against the original source article.
3. **Judicial Auditor (Final Arbiter):** Categorizes specific hallucination types (e.g., Entity, Date, Number) and outputs a final JSON verdict. A summary is only flagged as a hallucination if a strict majority of the models independently corroborate the failure.

## Results

<img width="5936" height="1542" alt="results_chart" src="https://github.com/user-attachments/assets/7eea8326-89c0-4e52-b05b-fe563c26faf0" />

## Files in this Repository
* `SumLLM_MultiAgent_Benchmark.ipynb`: The complete executable Kaggle notebook, heavily optimized with 4-bit quantization and sequential memory clearing to run three massive LLMs on a single GPU.
* `full_200_consensus.csv`: The final output data containing the generated summaries, the individual model votes, and the categorized consensus error types.

## Tech Stack
* **Models:** mT5-base, Llama-3 (8B), Gemma-2 (9B), Qwen-2.5 (7B)
* **Libraries:** HuggingFace Transformers, Datasets, PyTorch, Pandas, Matplotlib, BitsAndBytes
