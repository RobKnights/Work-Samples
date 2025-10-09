"""
translation_metrics.py
--------------------------------
Utility functions to evaluate translation outputs across multiple metrics:
- BLEU
- COMET
- chrF-S
- WER
- BERTScore
- chrF++

Model and dataset are modular so that future models (e.g., GPT-4o, GPT-5) can be plugged in easily.
"""

from transformers import pipeline
import pandas as pd
import numpy as np
import sacrebleu
import evaluate
from jiwer import wer
import logging

logging.getLogger("pytorch_lightning.utilities.migration.utils").setLevel(logging.ERROR)


# ====================================================
# 1️⃣ Translation Function
# ====================================================

def generate_translations(model_name: str, df: pd.DataFrame, src_col="Coptic", ref_col="English",
                          sample_size=None, src_lang="cop_Copt", tgt_lang="en_XX", max_length=512):
    """
    Generate translations using a specified model pipeline.

    Args:
        model_name (str): Hugging Face model name (e.g. "facebook/mbart-large-50-many-to-many-mmt").
        df (pd.DataFrame): DataFrame containing source and reference text.
        src_col (str): Source text column.
        ref_col (str): Reference text column.
        sample_size (int, optional): Subsample size for testing.
        src_lang (str): Source language token.
        tgt_lang (str): Target language token.
        max_length (int): Maximum sequence length for translation.

    Returns:
        (list, list, list): source_texts, reference_texts, predicted_texts
    """
    pipe = pipeline("translation", model=model_name)

    if sample_size:
        df = df.sample(n=sample_size, random_state=42).copy()

    df = df.dropna(subset=[ref_col])
    df[ref_col] = df[ref_col].astype(str)

    sources = df[src_col].tolist()
    refs = df[ref_col].tolist()
    preds = []

    for sent in sources:
        try:
            output = pipe(sent, src_lang=src_lang, tgt_lang=tgt_lang, max_length=max_length)
            preds.append(output[0]["translation_text"])
        except Exception as e:
            print(f"Error translating '{sent}': {e}")
            preds.append("")

    return sources, refs, preds


# ====================================================
# 2️⃣ BLEU
# ====================================================

def compute_bleu(predictions, references):
    """Compute corpus-level BLEU score."""
    if len(predictions) == 0 or len(predictions) != len(references):
        raise ValueError("Predictions and references must be non-empty and of equal length.")
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    return {"BLEU": bleu.score}


# ====================================================
# 3️⃣ COMET
# ====================================================

def compute_comet(predictions, references, sources, model_id="unbabel/wmt22-comet-da", batch_size=128):
    """Compute COMET metric with batching for large datasets."""
    comet_metric = evaluate.load("comet", model_id=model_id)

    preds = ["" if p is None else str(p) for p in predictions]
    refs = ["" if r is None else str(r) for r in references]
    srcs = ["" if s is None else str(s) for s in sources]

    all_seg_scores = []
    for i in range(0, len(preds), batch_size):
        chunk_preds = preds[i:i + batch_size]
        chunk_refs = refs[i:i + batch_size]
        chunk_srcs = srcs[i:i + batch_size]
        out = comet_metric.compute(predictions=chunk_preds, references=chunk_refs, sources=chunk_srcs)
        if "scores" in out:
            all_seg_scores.extend(out["scores"])
        elif "score" in out:
            all_seg_scores.extend([float(out["score"])] * len(chunk_preds))
    return {"COMET": float(np.mean(all_seg_scores))}


# ====================================================
# 4️⃣ chrF-S (sentence-level average)
# ====================================================

def compute_chrf_s(predictions, references, char_order=6, word_order=2):
    """Compute chrF-S: average of sentence-level chrF scores."""
    chrf = evaluate.load("chrf")
    scores = []
    for pred, ref in zip(predictions, references):
        result = chrf.compute(predictions=[pred], references=[[ref]], char_order=char_order, word_order=word_order)
        scores.append(result["score"] / 100.0)
    return {"chrF-S": float(np.mean(scores))}


# ====================================================
# 5️⃣ WER
# ====================================================

def compute_wer_metric(predictions, references):
    """Compute average Word Error Rate (WER)."""
    total = sum(wer(ref, pred) for ref, pred in zip(references, predictions))
    return {"WER": total / len(predictions)}


# ====================================================
# 6️⃣ BERTScore
# ====================================================

def compute_bertscore(predictions, references, lang="en"):
    """Compute BERTScore precision, recall, and F1."""
    bert_metric = evaluate.load("bertscore")
    results = bert_metric.compute(predictions=predictions, references=references, lang=lang)
    return {
        "BERTScore_P": float(np.mean(results["precision"])),
        "BERTScore_R": float(np.mean(results["recall"])),
        "BERTScore_F1": float(np.mean(results["f1"]))
    }


# ====================================================
# 7️⃣ chrF++
# ====================================================

def compute_chrfpp(predictions, references):
    """Compute corpus-level chrF++ score."""
    chrf_metric = evaluate.load("chrf")
    results = chrf_metric.compute(predictions=predictions, references=references, word_order=2)
    return {"chrF++": results["score"]}


# ====================================================
# 🧩 Master Wrapper Function (run all)
# ====================================================

def evaluate_all_metrics(predictions, references, sources):
    """Run all metrics and return a summary dict."""
    return {
        **compute_bleu(predictions, references),
        **compute_comet(predictions, references, sources),
        **compute_chrf_s(predictions, references),
        **compute_wer_metric(predictions, references),
        **compute_bertscore(predictions, references),
        **compute_chrfpp(predictions, references)
    }


# ====================================================
# 🧪 Example Usage
# ====================================================

if __name__ == "__main__":
    import sys
    print("Running test evaluation...")

    # Example minimal dataset
    df = pd.DataFrame({
        "Coptic": ["ⲡⲉϩⲟⲟⲩ", "ⲁϥⲥⲱⲧⲙ ⲛ̀ⲧⲉⲛⲛⲟⲩϫ"],
        "English": ["the day", "he heard our voice"]
    })

    model = "facebook/mbart-large-50-many-to-many-mmt"
    sources, refs, preds = generate_translations(model, df, sample_size=2)
    results = evaluate_all_metrics(preds, refs, sources)

    print("\nEvaluation Summary:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")


#========How to Use=============

"""
from translation_metrics import generate_translations, evaluate_all_metrics

model_name = "facebook/mbart-large-50-many-to-many-mmt"
sources, refs, preds = generate_translations(model_name, df, sample_size=1000)
results = evaluate_all_metrics(preds, refs, sources)

print(results)
"""

