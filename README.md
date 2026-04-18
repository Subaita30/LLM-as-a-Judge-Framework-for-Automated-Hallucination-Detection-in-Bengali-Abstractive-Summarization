# LLM-as-a-Judge: Automated Hallucination Detection in Bengali Summarization

This repository contains the research paper and implementation code for a fully automated, reference-free evaluation framework designed to detect factual hallucinations in Bengali abstractive summarization. 

##  Project Overview
Abstractive summarization models frequently generate factual hallucinations, a critical vulnerability in low-resource languages like Bengali. Traditional overlap metrics (like ROUGE) and noisy human references fail to evaluate factual consistency accurately. 

This project introduces a **cross-lingual, multi-agent Large Language Model (LLM) architecture** that systematically verifies factual grounding against dynamically retrieved source contexts without requiring human annotators.

### Key Findings
* **Optimal Judge:** `Gemma-2-9B-IT` achieved the highest empirical accuracy (66.67%) for Bengali-to-English reasoning.
* **Baseline Hallucination Rate:** An audit of 200 mT5-generated Bengali summaries from the XL-Sum dataset revealed a severe **61.50% hallucination rate**.
* **Error Typologies:** The most prevalent errors were General Errors (46.3%), Entity Swapping (30.1%), Numerical Alterations (12.2%), and Unsupported Claims (11.4%).

##  Multi-Agent Architecture
The evaluation pipeline operates entirely reference-free using a tripartite agent system:
1. **Fact Extractor:** Decomposes Bengali summaries into atomic English claims.
2. **Grounding Verifier:** Systematically cross-references claims against dynamically retrieved source contexts.
3. **Judicial Auditor:** Synthesizes the verifier's report to cast a final faithfulness verdict and categorize errors.

##  Repository Structure
* `/notebooks`: Contains the Kaggle Jupyter Notebook for the multi-agent pipeline (`CrewAI`, `LangChain`, `Transformers`).
* `/results`: Contains the generated outputs, including Phase 1 model comparisons and the final Phase 2 mT5 hallucination classifications.

## 🚀 Tech Stack
* **Languages & Libraries:** Python, PyTorch, Transformers, LangChain, CrewAI, Hugging Face Hub, Pandas.
* **Models:** `google/gemma-2-9b-it`, `csebuetnlp/mT5_multilingual_XLSum`, `Llama-3`, `Qwen2.5`.
* **Hardware:** Optimized for 4-bit quantized execution on dual NVIDIA T4 GPUs.

## 📝 Usage
The codebase is designed to run in a Kaggle or Google Colab environment with GPU acceleration. Ensure you have your `HF_TOKEN` configured in your environment secrets to download the necessary Hugging Face models.
