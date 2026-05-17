# Build_LLM_from_scratch
This repo is mostly for my learnings to build LLM from scratch

Build_LLM_from_scratch/
├── README.md                          # Project overview, setup instructions
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git ignore file
├── .python-version                   # Pyenv version (LLM_env)
│
├── data/                             # All datasets
│   ├── raw/                          # Original, immutable data
│   │   └── the-verdict.txt
│   ├── processed/                    # Cleaned/tokenized data
│   └── vocab/                        # Vocabulary files, token mappings
│
├── notebooks/                        # Jupyter notebooks for exploration
│   ├── 01_tokenization_exploration.ipynb
│   ├── 02_embeddings_visualization.ipynb
│   └── ...
│
├── src/                              # Source code (production-ready)
│   ├── __init__.py
│   ├── tokenizers/                   # Tokenization modules
│   │   ├── __init__.py
│   │   ├── simple_tokenizer.py       # Your SimpleTokenizerV1
│   │   ├── bpe_tokenizer.py          # Byte-Pair Encoding
│   │   └── tiktoken_wrapper.py       # GPT tokenizer wrapper
│   │
│   ├── models/                       # Model architectures
│   │   ├── __init__.py
│   │   ├── attention.py              # Self-attention mechanism
│   │   ├── transformer.py            # Transformer blocks
│   │   ├── gpt.py                    # GPT model
│   │   └── embeddings.py             # Token + position embeddings
│   │
│   ├── training/                     # Training logic
│   │   ├── __init__.py
│   │   ├── trainer.py                # Training loop
│   │   ├── optimizer.py              # Custom optimizers
│   │   └── losses.py                 # Loss functions
│   │
│   ├── data/                         # Data loading & processing
│   │   ├── __init__.py
│   │   ├── dataset.py                # PyTorch Dataset classes
│   │   └── preprocessing.py          # Data cleaning utilities
│   │
│   └── utils/                        # Helper functions
│       ├── __init__.py
│       ├── text_utils.py             # Text processing helpers
│       └── metrics.py                # Evaluation metrics
│
├── scripts/                          # Standalone scripts
│   ├── download_data.sh              # Download training data
│   ├── train.py                      # Training script
│   ├── evaluate.py                   # Evaluation script
│   └── generate_text.py              # Text generation
│
├── tests/                            # Unit tests
│   ├── __init__.py
│   ├── test_tokenizers.py
│   ├── test_models.py
│   └── test_training.py
│
├── configs/                          # Configuration files
│   ├── model_config.yaml             # Model hyperparameters
│   ├── training_config.yaml          # Training settings
│   └── tokenizer_config.yaml         # Tokenizer settings
│
├── checkpoints/                      # Saved model weights
│   └── .gitkeep
│
├── outputs/                          # Training outputs, logs
│   ├── logs/
│   ├── plots/
│   └── generated_text/
│
└── docs/                             # Documentation
    ├── architecture.md               # Model architecture notes
    ├── training.md                   # Training process
    └── notes/                        # Your learning notes
        ├── chapter_01.md
        ├── chapter_02.md
        └── ...
