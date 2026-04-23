import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from datasets import load_dataset
from tokenizers import Tokenizer
import sacrebleu
import evaluate
from bert_score import score
from comet import download_model, load_from_checkpoint
from rapidfuzz import fuzz
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
from analysis_metrics import run_full_analysis
import sys
import json
from pathlib import Path

# Import current model architecture
from config import get_config, get_weights_file_path
from model import build_transformer
from train import greedy_decode
from dataset import BilingualDataset

class EvaluateMT:
    def __init__(self, device=None, max_samples=None):
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.max_samples = max_samples
        self.metrics = {}
        
        # Load HuggingFace metrics
        self.meteor = evaluate.load("meteor")
        
        # Load COMET (Optional, may be heavy)
        print("Loading COMET model...")
        try:
            model_path = download_model("Unbabel/wmt22-comet-da")
            self.comet_model = load_from_checkpoint(model_path)
        except Exception as e:
            print(f"Failed to load COMET: {e}")
            self.comet_model = None

    def load_flores(self):
        print("Loading FLORES-200 from DGME/FLORES-200 (test split)...")
        # Load English and Hindi separately as this repo provides individual configs
        ds_en = load_dataset("DGME/FLORES-200", "flores_en", split="test")
        ds_hi = load_dataset("DGME/FLORES-200", "flores_hi", split="test")
        
        if len(ds_en) != len(ds_hi):
            raise ValueError(f"Dataset size mismatch: EN={len(ds_en)}, HI={len(ds_hi)}")
            
        sources = ds_en["text"]
        references = ds_hi["text"]
        
        if self.max_samples:
            sources = sources[:self.max_samples]
            references = references[:self.max_samples]
            
        return sources, references

    def load_in22(self):
        # Skipping IN22 for now as it's gated or script-based on HF
        print("IN22 loading skipped (using FLORES only).")
        return None, None

    def translate_hf(self, model_id, sources, max_length=128):
        print(f"Translating with {model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForSeq2SeqLM.from_pretrained(model_id).to(self.device)
        model.eval()
        
        translations = []
        for src in tqdm(sources, desc=f"Inference ({model_id})"):
            inputs = tokenizer(src, return_tensors="pt", truncation=True, max_length=max_length).to(self.device)
            with torch.no_grad():
                outputs = model.generate(**inputs, max_length=max_length, num_beams=5)
            translations.append(tokenizer.decode(outputs[0], skip_special_tokens=True))
            
        # Clean up memory
        del model
        del tokenizer
        torch.cuda.empty_cache()
        return translations

    def translate_custom(self, model_path, sources, config):
        print(f"Translating with custom model from {model_path}...")
        
        # Load tokenizers (using paths from config)
        tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
        tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))
        
        # Build model
        model = build_transformer(
            tokenizer_src.get_vocab_size(), 
            tokenizer_tgt.get_vocab_size(), 
            config['seq_len'], 
            config['seq_len'], 
            config['d_model']
        ).to(self.device)
        
        # Load weights
        state = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state['model_state_dict'])
        model.eval()
        
        translations = []
        for src in tqdm(sources, desc="Inference (Custom Model)"):
            # Prepare input
            src_ids = tokenizer_src.encode(src).ids
            src_tensor = torch.tensor([src_ids], dtype=torch.long).to(self.device)
            src_mask = (src_tensor != tokenizer_src.token_to_id('[PAD]')).unsqueeze(1).unsqueeze(2).to(self.device)
            
            # Decode
            out = greedy_decode(model, src_tensor, src_mask, tokenizer_src, tokenizer_tgt, config['seq_len'], self.device)
            
            # Decode to text
            translations.append(tokenizer_tgt.decode(out.detach().cpu().numpy()))
            
        del model
        torch.cuda.empty_cache()
        return translations

    def compute_metrics(self, preds, refs, sources=None):
        results = {}
        
        # 1. SacreBLEU
        bleu = sacrebleu.corpus_bleu(preds, [refs])
        results['BLEU'] = round(bleu.score, 2)
        
        # 2. chrF
        chrf = sacrebleu.corpus_chrf(preds, [refs])
        results['chrF'] = round(chrf.score, 2)
        
        # 3. METEOR
        meteor_res = self.meteor.compute(predictions=preds, references=refs)
        results['METEOR'] = round(meteor_res['meteor'] * 100, 2)
        
        # 4. BERTScore
        P, R, F1 = score(preds, refs, lang="hi", verbose=False, device=self.device)
        results['BERTScore'] = round(F1.mean().item() * 100, 2)
        
        # 5. COMET
        if self.comet_model and sources:
            data = [{"src": s, "mt": m, "ref": r} for s, m, r in zip(sources, preds, refs)]
            comet_res = self.comet_model.predict(data, batch_size=8, gpus=1 if self.device.type == "cuda" else 0)
            results['COMET'] = round(comet_res.system_score, 4)
        else:
            results['COMET'] = "N/A"
            
        return results

    def check_overlap(self, train_path, test_sources, threshold=90):
        print("Checking for train-test overlap...")
        # Note: This requires the training data to be accessible. 
        # If the user has a CSV or dataset object, we would load it here.
        # For now, let's assume we load the training set from the same source.
        config = get_config()
        try:
            ds_raw = load_dataset('Saurabh9901/english-hindi-dataset-2', 'default', split='train')
            train_sources = [item['translation'][config['lang_src']] for item in ds_raw]
            
            overlaps = 0
            # Sample check if too many
            check_test = test_sources[:100] 
            for t in tqdm(check_test, desc="Overlap check"):
                for tr in train_sources:
                    if fuzz.ratio(t, tr) >= threshold:
                        overlaps += 1
                        break
            rate = (overlaps / len(check_test)) * 100
            print(f"Overlap detection complete: {rate:.2f}% overlap found.")
            return rate
        except Exception as e:
            print(f"Overlap check failed: {e}")
            return 0.0

def main():
    evaluator = EvaluateMT(max_samples=50) # Use small number for testing
    
    # Datasets
    # Choosing FLORES for the main report as per requirements.docx Recommendation
    sources, references = evaluator.load_flores()
    
    # Models to evaluate
    MODELS = {
        "OPUS-MT": "Helsinki-NLP/opus-mt-en-hi",
        "M2M-100": "facebook/m2m100_418M",
    }
    
    report_data = []
    all_preds = {}  # Store predictions keyed by model name
    
    # 1. Baseline Evaluations
    for name, model_id in MODELS.items():
        preds = evaluator.translate_hf(model_id, sources)
        metrics = evaluator.compute_metrics(preds, references, sources)
        metrics['Model'] = name
        report_data.append(metrics)
        all_preds[name] = preds
        
    # 2. Custom Model Evaluation
    config = get_config()
    custom_model_path = os.path.join(config['model_folder'], "tmodel_step_240000.pt") 
    custom_model = None
    custom_preds = None

    # Load tokenizers for analysis
    tokenizer_src = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_src']))
    tokenizer_tgt = Tokenizer.from_file(config['tokenizer_file'].format(config['lang_tgt']))

    if os.path.exists(custom_model_path):
        preds = evaluator.translate_custom(custom_model_path, sources, config)
        metrics = evaluator.compute_metrics(preds, references, sources)
        metrics['Model'] = "Your Model"
        report_data.append(metrics)
        all_preds["Your Model"] = preds
        custom_preds = preds

        # Reload custom model for entropy analysis
        custom_model = build_transformer(
            tokenizer_src.get_vocab_size(),
            tokenizer_tgt.get_vocab_size(),
            config['seq_len'],
            config['seq_len'],
            config['d_model']
        ).to(evaluator.device)
        state = torch.load(custom_model_path, map_location=evaluator.device)
        custom_model.load_state_dict(state['model_state_dict'])
        custom_model.eval()
    else:
        print(f"Custom model at {custom_model_path} not found.")
        # Fall back to first available predictions for analysis
        custom_preds = list(all_preds.values())[0] if all_preds else None

    # 3. Create Report
    df = pd.DataFrame(report_data)
    cols = ['Model', 'BLEU', 'chrF', 'METEOR', 'BERTScore', 'COMET']
    df = df[cols]
    
    print("\n" + "="*50)
    print("FINAL EVALUATION RESULTS")
    print("="*50)
    print(df.to_markdown(index=False))
    
    # 4. Overlap Check
    evaluator.check_overlap(None, sources)

    # 5. Run Full Analysis (plots + CSV)
    if custom_preds is not None:
        run_full_analysis(
            model=custom_model,
            sources=sources,
            references=references,
            predictions=custom_preds,
            tokenizer_src=tokenizer_src,
            tokenizer_tgt=tokenizer_tgt,
            config=config,
            device=evaluator.device,
            output_dir="analysis_output",
            old_tokenizer_src_path="tokenizer_en_old.json",
            old_tokenizer_tgt_path="tokenizer_hi_old.json",
            max_entropy_samples=50,
        )
    else:
        print("No predictions available; skipping analysis plots.")

if __name__ == "__main__":
    main()
