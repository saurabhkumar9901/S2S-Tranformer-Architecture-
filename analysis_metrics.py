"""
Analysis & Visualization Metrics for the English-Hindi Translation Model.

Provides:
  1. Token Length Distribution (histogram)
  2. BLEU vs Sentence Length (scatter)
  3. Prediction Entropy Distribution (histogram)
  4. Tokenization Comparison: WordLevel vs BPE (bar chart)
  5. Translation Examples Table (CSV export)
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving plots
import matplotlib.pyplot as plt
import seaborn as sns
import sacrebleu
from pathlib import Path
from tqdm import tqdm
from tokenizers import Tokenizer

from dataset import causal_mask

# ── Style Configuration ──────────────────────────────────────────────────────
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.facecolor": "#f8f9fa",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": "#cccccc",
    "grid.color": "#e0e0e0",
    "font.family": "sans-serif",
})


# ═══════════════════════════════════════════════════════════════════════════════
# 1. TOKEN LENGTH DISTRIBUTION
# ═══════════════════════════════════════════════════════════════════════════════
def plot_length_distribution(lengths, save_path="length_dist.png"):
    """
    Plot a histogram of sentence token lengths.
    
    Args:
        lengths: list of int – token counts per sentence
        save_path: output file path
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(lengths, bins=50, kde=True, color="#4c72b0", edgecolor="#2b4570")
    plt.title("Sentence Length Distribution", fontweight="bold")
    plt.xlabel("Token Length")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Length distribution plot -> {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 2. BLEU vs LENGTH ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════════
def compute_sentence_bleu_scores(predictions, references):
    """
    Compute per-sentence BLEU scores using sacrebleu.
    
    Returns:
        list of float – BLEU score for each (prediction, reference) pair
    """
    scores = []
    for pred, ref in zip(predictions, references):
        bleu = sacrebleu.sentence_bleu(pred, [ref])
        scores.append(bleu.score)
    return scores


def plot_bleu_vs_length(lengths, bleu_scores, save_path="bleu_vs_length.png"):
    """
    Scatter plot of per-sentence BLEU against source sentence token length.
    
    Args:
        lengths: list of int – source token lengths
        bleu_scores: list of float – per-sentence BLEU scores
        save_path: output file path
    """
    plt.figure(figsize=(8, 5))
    plt.scatter(lengths, bleu_scores, alpha=0.5, s=20, color="#dd8452", edgecolors="#a85d32")
    
    # Add trend line
    if len(lengths) > 2:
        z = np.polyfit(lengths, bleu_scores, 1)
        p = np.poly1d(z)
        x_sorted = np.sort(lengths)
        plt.plot(x_sorted, p(x_sorted), "--", color="#c44e52", linewidth=2, label="Trend")
        plt.legend()
    
    plt.xlabel("Sentence Length (tokens)")
    plt.ylabel("BLEU Score")
    plt.title("BLEU vs Sentence Length", fontweight="bold")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] BLEU vs Length plot -> {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 3. CONFIDENCE / ENTROPY PLOT
# ═══════════════════════════════════════════════════════════════════════════════
def greedy_decode_with_entropy(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    """
    Greedy decode that also returns per-step entropy values.
    
    The model's ProjectionLayer outputs log-softmax, so:
        entropy = -sum(p * log(p)) = -sum(exp(log_p) * log_p)
    
    Returns:
        decoded_tokens: tensor of token IDs
        step_entropies: list of float – entropy at each decode step
    """
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    encoder_output = model.encode(source, source_mask)
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)

    step_entropies = []

    while True:
        if decoder_input.size(1) == max_len:
            break
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
        log_prob = model.project(out[:, -1])  # (1, vocab_size) – log-softmax

        # Compute entropy: H = -sum(p * log_p)
        prob = torch.exp(log_prob)
        entropy = -(prob * log_prob).sum(dim=-1).item()
        step_entropies.append(entropy)

        _, next_word = torch.max(log_prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
            dim=1
        )
        if next_word == eos_idx:
            break

    return decoder_input.squeeze(0), step_entropies


def plot_entropy(entropy_values, save_path="entropy.png"):
    """
    Histogram of per-step prediction entropy across all decoded sentences.
    
    Args:
        entropy_values: list of float – entropy at each decoding step (flattened)
        save_path: output file path
    """
    plt.figure(figsize=(8, 5))
    sns.histplot(entropy_values, bins=50, color="#55a868", edgecolor="#357a48")
    plt.title("Prediction Entropy Distribution", fontweight="bold")
    plt.xlabel("Entropy")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Entropy distribution plot -> {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 4. TOKENIZATION COMPARISON (WordLevel vs BPE)
# ═══════════════════════════════════════════════════════════════════════════════
def compute_token_counts(tokenizer, sentences):
    """Return list of token counts for each sentence using the given tokenizer."""
    return [len(tokenizer.encode(s).ids) for s in sentences]


def compare_tokenization(word_tokens, bpe_tokens, save_path="tokenization_comparison.png"):
    """
    Bar chart comparing average tokens per sentence between WordLevel and BPE.
    
    Args:
        word_tokens: list of int – token counts using WordLevel tokenizer
        bpe_tokens: list of int – token counts using BPE tokenizer
        save_path: output file path
    """
    labels = ["WordLevel", "BPE"]
    values = [np.mean(word_tokens), np.mean(bpe_tokens)]
    colors = ["#4c72b0", "#c44e52"]

    plt.figure(figsize=(6, 5))
    bars = plt.bar(labels, values, color=colors, edgecolor="#333333", width=0.5)

    # Annotate bars with values
    for bar, val in zip(bars, values):
        plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                 f"{val:.1f}", ha='center', va='bottom', fontweight='bold')

    plt.title("Average Tokens per Sentence", fontweight="bold")
    plt.ylabel("Token Count")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"[SAVED] Tokenization comparison plot -> {save_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# 5. TRANSLATION EXAMPLES TABLE
# ═══════════════════════════════════════════════════════════════════════════════
def save_translation_examples(examples, save_path="translation_examples.csv"):
    """
    Save a table of (Source, Prediction, Reference) to CSV.
    
    Args:
        examples: list of (source, prediction, reference) tuples
        save_path: output CSV file path
    """
    df = pd.DataFrame(examples, columns=["Source", "Prediction", "Reference"])
    df.to_csv(save_path, index=False, encoding='utf-8-sig')
    print(f"[SAVED] Translation examples -> {save_path}")
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# FULL ANALYSIS RUNNER
# ═══════════════════════════════════════════════════════════════════════════════
def run_full_analysis(
    model,
    sources,
    references,
    predictions,
    tokenizer_src,
    tokenizer_tgt,
    config,
    device,
    output_dir="analysis_output",
    old_tokenizer_src_path="tokenizer_en_old.json",
    old_tokenizer_tgt_path="tokenizer_hi_old.json",
    max_entropy_samples=None,
):
    """
    Run all 5 analysis metrics end-to-end and save outputs.

    Args:
        model: loaded custom Transformer model (in eval mode)
        sources: list of str – source English sentences
        references: list of str – reference Hindi translations
        predictions: list of str – model-predicted Hindi translations
        tokenizer_src: BPE tokenizer for source (English)
        tokenizer_tgt: BPE tokenizer for target (Hindi)
        config: project config dict
        device: torch device
        output_dir: directory for saving all outputs
        old_tokenizer_src_path: path to old WordLevel English tokenizer JSON
        old_tokenizer_tgt_path: path to old WordLevel Hindi tokenizer JSON
        max_entropy_samples: limit entropy computation to N sentences (None = all)
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 60)
    print("  RUNNING FULL ANALYSIS")
    print("=" * 60)

    # ── 1. Token Length Distribution ──────────────────────────────────────────
    print("\n[1/5] Computing token lengths...")
    src_lengths = compute_token_counts(tokenizer_src, sources)
    plot_length_distribution(src_lengths, save_path=str(out / "length_dist.png"))

    # ── 2. BLEU vs Length ─────────────────────────────────────────────────────
    print("\n[2/5] Computing per-sentence BLEU scores...")
    bleu_scores = compute_sentence_bleu_scores(predictions, references)
    plot_bleu_vs_length(src_lengths, bleu_scores, save_path=str(out / "bleu_vs_length.png"))

    # ── 3. Prediction Entropy ─────────────────────────────────────────────────
    print("\n[3/5] Computing prediction entropy (custom model)...")
    all_entropies = []
    entropy_sources = sources if max_entropy_samples is None else sources[:max_entropy_samples]

    if model is not None:
        model.eval()
        with torch.no_grad():
            for src in tqdm(entropy_sources, desc="Entropy computation"):
                src_ids = tokenizer_src.encode(src).ids
                src_tensor = torch.tensor([src_ids], dtype=torch.long).to(device)
                pad_id = tokenizer_src.token_to_id('[PAD]')
                src_mask = (src_tensor != pad_id).unsqueeze(1).unsqueeze(2).to(device)

                _, step_entropies = greedy_decode_with_entropy(
                    model, src_tensor, src_mask,
                    tokenizer_src, tokenizer_tgt,
                    config['seq_len'], device
                )
                all_entropies.extend(step_entropies)

        plot_entropy(all_entropies, save_path=str(out / "entropy.png"))
    else:
        print("  [SKIP] No custom model loaded; entropy plot skipped.")

    # ── 4. Tokenization Comparison ────────────────────────────────────────────
    print("\n[4/5] Comparing tokenizations (WordLevel vs BPE)...")
    bpe_src_counts = src_lengths  # already computed above with BPE tokenizer

    if Path(old_tokenizer_src_path).exists():
        old_tok_src = Tokenizer.from_file(old_tokenizer_src_path)
        word_src_counts = compute_token_counts(old_tok_src, sources)
        compare_tokenization(word_src_counts, bpe_src_counts,
                             save_path=str(out / "tokenization_comparison.png"))
    else:
        print(f"  [SKIP] Old tokenizer not found at {old_tokenizer_src_path}")

    # ── 5. Translation Examples Table ─────────────────────────────────────────
    print("\n[5/5] Saving translation examples...")
    examples = list(zip(sources, predictions, references))
    save_translation_examples(examples, save_path=str(out / "translation_examples.csv"))

    # ── Summary Statistics ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  ANALYSIS SUMMARY")
    print("=" * 60)
    print(f"  Sentences analysed    : {len(sources)}")
    print(f"  Avg source length     : {np.mean(src_lengths):.1f} tokens")
    print(f"  Avg sentence BLEU     : {np.mean(bleu_scores):.2f}")
    if all_entropies:
        print(f"  Mean prediction entropy: {np.mean(all_entropies):.4f}")
        print(f"  Entropy std-dev       : {np.std(all_entropies):.4f}")
    print(f"  Outputs saved to      : {out.resolve()}")
    print("=" * 60 + "\n")
