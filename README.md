# Build a Large Language Model (From Scratch)

A comprehensive implementation of a GPT-style language model built from the ground up, following best practices for production ML systems.

## Overview

This project implements all core components of a Large Language Model:
- **Tokenization** - Byte-Pair Encoding (BPE) and custom tokenizers
- **Embeddings** - Token and positional embeddings
- **Attention** - Self-attention and multi-head attention mechanisms
- **Transformer** - Complete transformer architecture
- **Training** - Full training pipeline with optimization
- **Generation** - Text generation and inference

## Quick Start

### 1. Environment Setup

```bash
# Create virtual environment using pyenv
pyenv virtualenv 3.12.12 LLM_env
pyenv activate LLM_env

# Install dependencies
pip install -r requirements.txt
```

### 2. Run Tokenization Example

```python
from src.tokenizers.simple_tokenizer import SimpleTokenizerV1, build_vocab_from_text

# Load and tokenize text
with open("data/raw/the-verdict.txt", "r") as f:
    text = f.read()

vocab = build_vocab_from_text(text)
tokenizer = SimpleTokenizerV1(vocab)

# Encode and decode
ids = tokenizer.encode("Hello, world!")
decoded = tokenizer.decode(ids)
```

### 3. Training (Coming Soon)

```bash
python scripts/train.py --config configs/training_config.yaml
```

## Project Structure

```
Build_LLM_from_scratch/
├── README.md                          # Project overview, setup instructions
├── requirements.txt                   # Python dependencies
├── .gitignore                         # Git ignore file
├── .python-version                    # Pyenv version (LLM_env)
│
├── data/                              # All datasets
│   ├── raw/                           # Original, immutable data
│   ├── processed/                     # Cleaned/tokenized data
│   └── vocab/                         # Vocabulary files, token mappings
│
├── notebooks/                         # Jupyter notebooks for exploration
│
├── src/                               # Source code (production-ready)
│   ├── tokenizers/                    # Tokenization modules
│   │   ├── simple_tokenizer.py        # SimpleTokenizerV1 & V2
│   │   ├── bpe_tokenizer.py           # Byte-Pair Encoding
│   │   └── tiktoken_wrapper.py        # GPT tokenizer wrapper
│   │
│   ├── models/                        # Model architectures
│   │   ├── attention.py               # Self-attention mechanism
│   │   ├── transformer.py             # Transformer blocks
│   │   ├── gpt.py                     # GPT model
│   │   └── embeddings.py              # Token + position embeddings
│   │
│   ├── training/                      # Training logic
│   │   ├── trainer.py                 # Training loop
│   │   ├── optimizer.py               # Custom optimizers
│   │   └── losses.py                  # Loss functions
│   │
│   ├── data/                          # Data loading & processing
│   │   ├── dataset.py                 # PyTorch Dataset classes
│   │   └── preprocessing.py           # Data cleaning utilities
│   │
│   └── utils/                         # Helper functions
│       ├── text_utils.py              # Text processing helpers
│       └── metrics.py                 # Evaluation metrics
│
├── scripts/                           # Standalone scripts
│   ├── train.py                       # Training script
│   ├── evaluate.py                    # Evaluation script
│   └── generate_text.py               # Text generation
│
├── tests/                             # Unit tests
│   ├── test_tokenizers.py
│   ├── test_models.py
│   └── test_training.py
│
├── configs/                           # Configuration files
│   ├── model_config.yaml              # Model hyperparameters
│   ├── training_config.yaml           # Training settings
│   └── tokenizer_config.yaml          # Tokenizer settings
│
├── checkpoints/                       # Saved model weights
├── outputs/                           # Training outputs, logs
│   ├── logs/
│   ├── plots/
│   └── generated_text/
│
└── docs/                              # Documentation
    ├── architecture.md                # Model architecture notes
    ├── training.md                    # Training process
    └── notes/                         # Learning notes
```

## Key Components

### Tokenization

- **SimpleTokenizerV1**: Basic tokenization by splitting on whitespace and punctuation
- **SimpleTokenizerV2**: Enhanced version with `<|unk|>` and `<|endoftext|>` tokens
- **BPE Tokenizer**: Byte-Pair Encoding for handling out-of-vocabulary words
- **tiktoken**: GPT-2/GPT-3 compatible tokenizer

### Model Architecture (In Progress)

- Self-attention mechanism
- Multi-head attention
- Transformer blocks
- Full GPT model implementation

## Development

### Running Tests

```bash
pytest tests/
```

### Code Style

```bash
# Format code
black src/

# Lint
flake8 src/
```

## Resources

- **Book**: "Build a Large Language Model (From Scratch)" by Sebastian Raschka
- **Paper**: ["Attention Is All You Need"](https://arxiv.org/abs/1706.03762) (Transformer architecture)
- **GPT-2 Paper**: ["Language Models are Unsupervised Multitask Learners"](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)

## Progress

- [x] Chapter 1: Tokenization basics
- [x] Chapter 2: Special tokens and BPE
- [ ] Chapter 3: Embeddings
- [ ] Chapter 4: Attention mechanisms
- [ ] Chapter 5: Transformer blocks
- [ ] Chapter 6: Training pipeline

## License

This project is for educational purposes.
