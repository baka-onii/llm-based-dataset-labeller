# LLM-based Dataset Labeller

This repository provides a script to **automatically label raw text chunks** into a Chat-style dataset for **fine-tuning instruction-following LLMs**.  
Instead of manually writing questions, a small Hugging Face model (e.g., `phi-3-mini`) is used to generate questions for each text chunk.

## ✨ Features
- Sentence-aware text chunking with overlap
- LLM-based labeling (Hugging Face transformers pipeline)
- Batched + threaded inference for efficiency
- GPU/CPU detection
- Resume from last checkpoint (safe for Colab)
- Saves outputs in **ChatML-style JSONL**

## 📂 Repo Structure
- `scripts/` → Python scripts  
- `dataset/raw_texts/` → Put your raw `.txt` inputs here  
- `dataset/processed_texts/` → Generated JSONL dataset is saved here  
- `examples/` → Sample output for quick reference  

## 🚀 Installation
```bash
git clone https://github.com/YOUR_USERNAME/llm-dataset-labeller.git
cd llm-dataset-labeller
pip install -r requirements.txt
