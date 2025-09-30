'''
This script is made to simplify the process of labelling the text data and create a Chat-Styled chunked dataset for Supervised Training/Fine-tuning, by implementing a small Huggingface LLM model to label the chunk, return it's appropriate instruction/question, and output a jsonl dataset with all labelled chunks. You can customise the script to your needs by changing CHUNK_SIZE, OVERLAP(words overlap to preserve context across chunks), BATCH_SIZE(for writing the processed chunks in the OUTPUT FILE) and MODEL_NAME (Huggingface).

Currently, this script implements following features:

uses sentence-aware chunking,
prints GPU/CUDA device info,
batches prompts for generation (safe, efficient single-GPU),
auto-adjusts batch size if OOM happens (retries with smaller batches),
has robust resume feature using (filename, chunk_id) stored in meta,
incremental flushing to disk,
overall + per-file progress bars,
safe parsing of model outputs (handles different return formats).

#---------- Installing Requirements -------------------#
pip install transformers torch nltk tqdm

#---------- How it works --------------#
Cleaning → removes newlines, page markers, and non-printables.
Sentence-aware chunking → splits text into chunks, overlapping words to preserve context.
Batching → sends multiple chunks in parallel to the labeling model.
LLM labeling → generates one question per chunk.
Dataset output → saves in JSONL with system, user, assistant roles.

#----------- Tunable Parameters ------------#
CHUNK_SIZE → shorter chunks = more samples, longer = richer context.
OVERLAP → set higher (30–50) for more continuity, lower for less redundancy.
BATCH_SIZE → lower if you hit CUDA OOM.
MODEL_NAME → replace with any Hugging Face text-gen model.

#----------- Example Output -------------#
{
  "messages": [
    {"role": "system", "content": "You are a knowledgeable assistant trained on philosophy, religion, and science."},
    {"role": "user", "content": "What does Augustine argue about the nature of God?"},
    {"role": "assistant", "content": "Augustine discusses the eternal, unchanging, and omnipotent nature of God..."}
  ],
  "meta": {"source": "city_of_god.txt", "chunk_id": 58}
}

#-----------------------------------------------------------------------#

This script implements resume feature. Script automatically resumes by checking existing JSONL and skipping completed (source, chunk_id).
It is safe to stop and restart without re-doing processed chunks.

Usage of this script in Google Colab (even free tier works) was kept in mind when creating this script. This script requires a meaty GPU with good amount of VRAM for running the LLM model even if it's small. This is why the resume feature is implemented in this. You can run this script locally if you have atleast 16 gbs of VRAM, or experiment with smaller LLMs if you've a weaker GPU or less VRAM.

Once generated, your dataset can be used with Hugging Face TRL:
```python
from datasets import load_dataset
dataset = load_dataset("json", data_files="phase1_dataset.jsonl")
```
And then passed into SFTTrainer / trl for supervised fine-tuning.

There might be few issues with it, and I'll most probably not update this script in the future. You are free to upgrade and customise this script for yourself.

'''

import os
import re
import json
from pathlib import Path
from tqdm import tqdm
from transformers import pipeline
import torch
import nltk
from concurrent.futures import ThreadPoolExecutor, as_completed

# Download NLTK sentence tokenizer if not already
nltk.download('punkt', quiet=True)
nltk.download('punkt_tab', quiet=True)

# Parameters

INPUT_DIR = "/content/drive/MyDrive/Debate/dataset/raw_texts"
OUTPUT_FILE = "/content/drive/MyDrive/Debate/dataset/processed_texts/phase1_dataset.jsonl"
CHUNK_SIZE = 400       # words per chunk
OVERLAP = 20           # word overlap
BATCH_SIZE = 100         # save every N samples (for flushing)
MODEL_NAME = "microsoft/phi-3-mini-4k-instruct" # the LLM model used for labelling

# ---------------------------
#  Helper functions
# ---------------------------

def clean_text(text: str) -> str:
    """Basic cleanup of raw text."""
    text = text.replace("\n", " ")
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"---\s*Page\s*\d+\s*---", " ", text, flags=re.IGNORECASE) # Remove "Page No. texts"
    text = re.sub(r"[^\x20-\x7E]+", " ", text)  # remove non-printable
    return text.strip()

def sentence_chunk_text(text: str, chunk_size=400, overlap=20):
    """Sentence-aware chunking: group sentences until chunk_size words."""
    sentences = nltk.sent_tokenize(text)
    chunks, current, count = [], [], 0

    for sent in sentences:
        words = sent.split()
        if count + len(words) > chunk_size:
            if current:
                chunks.append(" ".join(current))
            overlap_words = current[-overlap:] if overlap > 0 else []
            current = overlap_words + words
            count = len(current)
        else:
            current.extend(words)
            count += len(words)

    if current:
        chunks.append(" ".join(current))
    return chunks

def load_completed(output_file):
    """Return set of (source, chunk_id) already processed."""
    completed = set()
    if not os.path.exists(output_file):
        return completed
    with open(output_file, "r", encoding="utf-8") as f:
        for line in f:
            try:
                entry = json.loads(line)
                src = entry.get("meta", {}).get("source")
                cid = entry.get("meta", {}).get("chunk_id")
                if src is not None and cid is not None:
                    completed.add((src, cid))
            except Exception:
                continue
    return completed

# ---------------------------
# Batched + Threaded LLM
# ---------------------------
def generate_questions_batch(pipe, file_name, batch):
    """
    Generate questions for a batch of (idx, chunk) tuples.
    Returns list of entry dicts.
    """
    prompts = []
    for idx, chunk in batch:
        prompt = f"""You are a dataset labeler.
Task: Given the passage below, generate ONE natural question or instruction that can be directly answered by it. Assume it is to fine tune an instruction-based LLM model.
Rules:
- Output ONLY the question.
- Do not add anything else (no explanations, no rephrasing, no quotes).
Passage:
{chunk}
"""
        prompts.append(prompt)

    results = pipe(prompts, max_new_tokens=60, do_sample=True, temperature=0.7, return_full_text=False)

    entries = []
    for (idx, chunk), res in zip(batch, results):
        try:
            # res is a list of dicts; take the first one
            question_text = res[0]["generated_text"].strip()
            # 1. Split "Question:" or "Output:"
            question_text = question_text.split(":", 1)[-1].strip()
            # 2. Remove extra quotes
            question_text = question_text.strip(' "\'')
            # 3. Take only first sentence (in case model adds explanation)
            question_text = nltk.tokenize.sent_tokenize(question_text)[0]
            # 4. Remove any trailing punctuation that is not a question mark
            if not question_text.endswith('?'):
                question_text = question_text.rstrip('.!?') + '?'

            entry = {
                "messages": [
                    {"role": "system", "content": "You are a knowledgeable assistant trained on philosophy, religion, and science. Interpret the text and provide clear, insightful summaries, explanations, or key teachings in simple English."},
                    {"role": "user", "content": question_text},
                    {"role": "assistant", "content": chunk}
                ],
                "meta": {"source": file_name, "chunk_id": idx}
            }
            entries.append(entry)
        except Exception as e:
            print(f"⚠️ Skipped chunk {idx} in {file_name}: {e}")

    return entries


def build_dataset_batched_threaded(input_dir, output_file, pipe, max_workers=4, write_batch=50, batch_size=8):
    input_dir = Path(input_dir)
    completed = load_completed(output_file)
    print(f"📂 Resuming with {len(completed)} completed samples in {output_file}")

    total_written = len(completed)
    entries_to_write = []

    with open(output_file, "a", encoding="utf-8") as f_out:
        for file in input_dir.glob("*.txt"):
            with open(file, "r", encoding="utf-8") as f:
                raw_text = f.read()

            text = clean_text(raw_text)
            chunks = sentence_chunk_text(text, CHUNK_SIZE, OVERLAP)

            # Only unprocessed chunks
            tasks = [(idx, chunk) for idx, chunk in enumerate(chunks) if (file.name, idx) not in completed]

            # Split into batches
            batches = [tasks[i:i + batch_size] for i in range(0, len(tasks), batch_size)]

            with ThreadPoolExecutor(max_workers=max_workers) as executor:
                future_to_batch = {executor.submit(generate_questions_batch, pipe, file.name, batch): batch for batch in batches}

                for future in as_completed(future_to_batch):
                    batch_entries = future.result()
                    entries_to_write.extend(batch_entries)
                    total_written += len(batch_entries)

                    # Flush in batches
                    if len(entries_to_write) >= write_batch:
                        for entry in entries_to_write:
                            f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                        f_out.flush()
                        print(f"💾 Saved {total_written} samples so far...")
                        entries_to_write = []

            # Flush remaining
            if entries_to_write:
                for entry in entries_to_write:
                    f_out.write(json.dumps(entry, ensure_ascii=False) + "\n")
                f_out.flush()
                entries_to_write = []

    print(f"✅ Finished. Total {total_written} samples written to {output_file}")

# ---------------------------
# Run
# ---------------------------
if __name__ == "__main__":
    if torch.cuda.is_available():
        print(f"✅ Using GPU: {torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})")
    else:
        print("⚠️ CUDA not available, using CPU (this will be very slow).")

    # Load pipeline with GPU and half precision if possible
    pipe = pipeline("text-generation", model=MODEL_NAME, device_map="auto", torch_dtype=torch.float16)

    build_dataset_batched_threaded(INPUT_DIR, OUTPUT_FILE, pipe,
                                   max_workers=2, write_batch=50, batch_size=8)

    # Preview first 3 entries
    with open(OUTPUT_FILE, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i < 3:
                print(json.loads(line))
