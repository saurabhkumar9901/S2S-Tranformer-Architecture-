# S2S Transformer: English to Hindi Translation

This repository contains a complete implementation of a Sequence-to-Sequence Transformer architecture from scratch using PyTorch. The model is designed for Neural Machine Translation (NMT), specifically for translating English text to Hindi.

## Overview

The project implements the original Transformer architecture as described in "Attention Is All You Need." It includes a full training pipeline, custom dataset handling for the IITB English-Hindi corpus, and an extensive evaluation suite.

### Key Features
* Custom implementation of Multi-head Attention, Encoder, and Decoder blocks.
* Sinusoidal Positional Encoding to handle sequence ordering.
* Byte Pair Encoding (BPE) tokenization for both English and Hindi.
* Training pipeline with checkpointing, TensorBoard logging, and label smoothing.
* Comprehensive evaluation metrics including BLEU, chrF, METEOR, and BERTScore.

## Repository Structure

* `model.py`: The core Transformer architecture implementation.
* `train.py`: Training logic, data loading, and validation loops.
* `dataset.py`: Custom PyTorch Dataset class for handling bilingual pairs and causal masking.
* `config.py`: Centralized configuration for hyperparameters and file paths.
* `inference.py`: Module for generating translations and calculating evaluation metrics.
* `evaluate_models.py`: Comparative evaluation script to benchmark against models like OPUS-MT and M2M-100.

## Requirements

The following dependencies are required:
* torch
* datasets
* tokenizers
* evaluate
* sacrebleu
* bert-score
* tqdm
* pandas

## Model Architecture

The implementation follows the standard Transformer design:
1. **Input Embeddings**: Converts tokens to d_model dimensional vectors.
2. **Positional Encoding**: Adds relative position information using sine and cosine functions.
3. **Encoder**: A stack of 6 layers, each containing multi-head self-attention and a feed-forward network.
4. **Decoder**: A stack of 6 layers, each containing masked self-attention, cross-attention (attending to encoder output), and a feed-forward network.
5. **Projection Layer**: A linear layer with log_softmax to project model output to the target vocabulary size.

## Usage

### Training
To train the model from scratch or resume from a checkpoint:
1. Configure hyperparameters in `config.py`.
2. Run the training script:
   ```bash
   python train.py
   ```
The training script will automatically download the IITB English-Hindi corpus, build the BPE tokenizers, and start the training process.

### Inference and Evaluation
To evaluate the model's performance on the validation set:
```bash
python inference.py
```
This will load the specified model checkpoint and provide a detailed report on metrics like BLEU, ROUGE, and chrF.

To run a comparative analysis against baseline models on the FLORES-200 dataset:
```bash
python evaluate_models.py
```

## Configuration

Hyperparameters are managed in `config.py`. Key settings include:
* `batch_size`: 8
* `seq_len`: 350
* `d_model`: 512
* `lr`: 10^-5
* `num_epochs`: 30

## Metrics Implementation

The project supports several NMT evaluation metrics to ensure translation quality:
* **BLEU**: Measures n-gram overlap.
* **chrF**: Character n-gram F-score, particularly useful for morphologically rich languages like Hindi.
* **BERTScore**: Uses multilingual BERT embeddings to measure semantic similarity.
* **METEOR**: Aligns words based on stemming and synonymy.
