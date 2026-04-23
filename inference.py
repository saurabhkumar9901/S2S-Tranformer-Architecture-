"""
Inference module with multiple evaluation metrics for English-Hindi translation.

Metrics implemented:
- BLEU: Bilingual Evaluation Understudy
- ROUGE: Recall-Oriented Understudy for Gisting Evaluation
- METEOR: Metric for Evaluation of Translation with Explicit ORdering
- chrF: Character n-gram F-score
- TER: Translation Edit Rate
- BERTScore: Semantic similarity using BERT embeddings
"""

import torch
import evaluate
from tqdm import tqdm
from pathlib import Path
from config import get_config, get_weights_file_path
from train import get_model, get_ds, greedy_decode
from dataset import BilingualDataset, causal_mask


# Load all metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
meteor_metric = evaluate.load("meteor")
chrf_metric = evaluate.load("chrf")
ter_metric = evaluate.load("ter")
bertscore_metric = evaluate.load("bertscore")


def compute_all_metrics(predictions: list, references: list, source_lang: str = "en", target_lang: str = "hi"):
    """
    Compute all evaluation metrics for translation quality.
    
    Args:
        predictions: List of predicted translations
        references: List of reference translations (each reference should be a list)
        source_lang: Source language code
        target_lang: Target language code
    
    Returns:
        Dictionary containing all metric scores
    """
    results = {}
    
    # Flatten references for metrics that need single strings
    flat_references = [ref[0] if isinstance(ref, list) else ref for ref in references]
    
    # 1. BLEU Score
    print("Computing BLEU...")
    bleu_result = bleu_metric.compute(predictions=predictions, references=references)
    results['bleu'] = bleu_result['bleu'] * 100
    results['bleu_precisions'] = [p * 100 for p in bleu_result['precisions']]
    
    # 2. ROUGE Score
    print("Computing ROUGE...")
    rouge_result = rouge_metric.compute(predictions=predictions, references=flat_references)
    results['rouge1'] = rouge_result['rouge1'] * 100
    results['rouge2'] = rouge_result['rouge2'] * 100
    results['rougeL'] = rouge_result['rougeL'] * 100
    results['rougeLsum'] = rouge_result['rougeLsum'] * 100
    
    # 3. METEOR Score
    print("Computing METEOR...")
    meteor_result = meteor_metric.compute(predictions=predictions, references=flat_references)
    results['meteor'] = meteor_result['meteor'] * 100
    
    # 4. chrF Score (Character-level, great for Hindi)
    print("Computing chrF...")
    chrf_result = chrf_metric.compute(predictions=predictions, references=references)
    results['chrf'] = chrf_result['score']
    
    # 5. TER Score (Translation Edit Rate - lower is better)
    print("Computing TER...")
    ter_result = ter_metric.compute(predictions=predictions, references=references)
    results['ter'] = ter_result['score']
    
    # 6. BERTScore (Semantic similarity)
    print("Computing BERTScore (this may take a while)...")
    try:
        bertscore_result = bertscore_metric.compute(
            predictions=predictions, 
            references=flat_references, 
            lang=target_lang,
            model_type="bert-base-multilingual-cased"  # Good for Hindi
        )
        results['bertscore_precision'] = sum(bertscore_result['precision']) / len(bertscore_result['precision']) * 100
        results['bertscore_recall'] = sum(bertscore_result['recall']) / len(bertscore_result['recall']) * 100
        results['bertscore_f1'] = sum(bertscore_result['f1']) / len(bertscore_result['f1']) * 100
    except Exception as e:
        print(f"BERTScore computation failed: {e}")
        results['bertscore_precision'] = None
        results['bertscore_recall'] = None
        results['bertscore_f1'] = None
    
    return results


def print_metrics_report(results: dict):
    """Print a formatted report of all metrics."""
    print("\n" + "=" * 80)
    print("                    TRANSLATION EVALUATION METRICS REPORT")
    print("=" * 80)
    
    print("\n📊 N-GRAM BASED METRICS:")
    print("-" * 40)
    print(f"  BLEU Score:        {results['bleu']:.2f}")
    if 'bleu_precisions' in results:
        print(f"    - 1-gram:        {results['bleu_precisions'][0]:.2f}")
        print(f"    - 2-gram:        {results['bleu_precisions'][1]:.2f}")
        print(f"    - 3-gram:        {results['bleu_precisions'][2]:.2f}")
        print(f"    - 4-gram:        {results['bleu_precisions'][3]:.2f}")
    
    print(f"\n  METEOR Score:      {results['meteor']:.2f}")
    
    print("\n📝 ROUGE SCORES:")
    print("-" * 40)
    print(f"  ROUGE-1:           {results['rouge1']:.2f}")
    print(f"  ROUGE-2:           {results['rouge2']:.2f}")
    print(f"  ROUGE-L:           {results['rougeL']:.2f}")
    print(f"  ROUGE-Lsum:        {results['rougeLsum']:.2f}")
    
    print("\n🔤 CHARACTER-LEVEL METRICS:")
    print("-" * 40)
    print(f"  chrF Score:        {results['chrf']:.2f}")
    
    print("\n✏️ EDIT-BASED METRICS:")
    print("-" * 40)
    print(f"  TER Score:         {results['ter']:.2f}  (lower is better)")
    
    print("\n🧠 SEMANTIC METRICS:")
    print("-" * 40)
    if results.get('bertscore_f1') is not None:
        print(f"  BERTScore F1:      {results['bertscore_f1']:.2f}")
        print(f"  BERTScore Prec:    {results['bertscore_precision']:.2f}")
        print(f"  BERTScore Recall:  {results['bertscore_recall']:.2f}")
    else:
        print("  BERTScore:         (computation failed)")
    
    print("\n" + "=" * 80)
    print("INTERPRETATION GUIDE:")
    print("-" * 40)
    print("  • BLEU, ROUGE, METEOR, chrF, BERTScore: Higher is better (0-100)")
    print("  • TER: Lower is better (measures edits needed)")
    print("  • chrF is especially good for morphologically rich languages like Hindi")
    print("  • BERTScore captures semantic similarity beyond surface-level matching")
    print("=" * 80 + "\n")


def run_evaluation(model, validation_ds, tokenizer_src, tokenizer_tgt, max_len, device, num_examples=100):
    """
    Run full evaluation with all metrics.
    
    Args:
        model: The translation model
        validation_ds: Validation dataloader
        tokenizer_src: Source language tokenizer
        tokenizer_tgt: Target language tokenizer
        max_len: Maximum sequence length
        device: torch device
        num_examples: Number of examples to evaluate
    
    Returns:
        Dictionary with all metric scores
    """
    model.eval()
    predictions = []
    references = []
    sources = []
    
    print(f"\n🔄 Running inference on {num_examples} examples...")
    
    with torch.no_grad():
        for i, batch in enumerate(tqdm(validation_ds, total=num_examples, desc="Evaluating")):
            if i >= num_examples:
                break
                
            encoder_input = batch['encoder_input'].to(device)
            encoder_mask = batch['encoder_mask'].to(device)
            
            # Greedy decode
            model_out = greedy_decode(model, encoder_input, encoder_mask, tokenizer_src, tokenizer_tgt, max_len, device)
            
            source_text = batch['src_text'][0]
            target_text = batch['tgt_text'][0]
            model_out_text = tokenizer_tgt.decode(model_out.detach().cpu().numpy())
            
            sources.append(source_text)
            predictions.append(model_out_text)
            references.append([target_text])
    
    print(f"\n✅ Inference complete. Computing metrics on {len(predictions)} examples...\n")
    
    # Compute all metrics
    results = compute_all_metrics(predictions, references)
    
    # Print formatted report
    print_metrics_report(results)
    
    return results, predictions, references, sources


def show_examples(sources, predictions, references, num_examples=10):
    """Display sample translations."""
    print("\n" + "=" * 80)
    print("                         SAMPLE TRANSLATIONS")
    print("=" * 80)
    
    for i in range(min(num_examples, len(predictions))):
        print(f"\n[Example {i+1}]")
        print(f"  SOURCE:    {sources[i]}")
        print(f"  TARGET:    {references[i][0]}")
        print(f"  PREDICTED: {predictions[i]}")
        print("-" * 80)


def load_model_and_evaluate(model_epoch: str = "23", num_examples: int = 100, show_samples: int = 10):
    """
    Complete evaluation pipeline: load model and run all metrics.
    
    Args:
        model_epoch: Epoch number of the model to load
        num_examples: Number of examples to evaluate
        show_samples: Number of sample translations to display
    
    Returns:
        Dictionary with all metric scores
    """
    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    config = get_config()
    
    # Load data
    print("Loading dataset and tokenizers...")
    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(config)
    
    # Load model
    print(f"Loading model from epoch {model_epoch}...")
    model = get_model(config, tokenizer_src.get_vocab_size(), tokenizer_tgt.get_vocab_size()).to(device)
    model_filename = get_weights_file_path(config, model_epoch)
    state = torch.load(model_filename, map_location=device)
    model.load_state_dict(state['model_state_dict'])
    print("Model loaded successfully!")
    
    # Run evaluation
    results, predictions, references, sources = run_evaluation(
        model, val_dataloader, tokenizer_src, tokenizer_tgt, 
        config['seq_len'], device, num_examples
    )
    
    # Show sample translations
    if show_samples > 0:
        show_examples(sources, predictions, references, show_samples)
    
    return results


if __name__ == "__main__":
    # Run evaluation with default settings
    results = load_model_and_evaluate(
        model_epoch="23",
        num_examples=100,
        show_samples=10
    )
