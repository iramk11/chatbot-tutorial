# PyTorch Chatbot Tutorial with W&B Integration

This repository contains a complete implementation of a conversational chatbot using PyTorch, with Weights & Biases (W&B) integration for hyperparameter optimization and experiment tracking.

## Features

- **Seq2Seq Model**: Implements an encoder-decoder architecture with attention mechanism
- **W&B Integration**: Full integration with Weights & Biases for experiment tracking and hyperparameter sweeps
- **Hyperparameter Optimization**: Configurable sweep over learning rates, optimizers, and other training parameters
- **Movie Dialog Corpus**: Uses Cornell Movie Dialog dataset for training conversational models

## Files

- `chatbot_tutorial.ipynb`: Main tutorial notebook with W&B integration
- `Organizing_Hyperparameter_Sweeps_in_PyTorch_with_W&B.ipynb`: Additional W&B tutorial
- `conversations.json`, `corpus.json`, `index.json`, `speakers.json`: Dataset files
- `.gitignore`: Git ignore file excluding large data files and model checkpoints

## W&B Sweep Configuration

The notebook includes a comprehensive sweep configuration for hyperparameter optimization:

```python
SWEEP_CONFIG = {
    'method': 'random',
    'metric': {'name': 'loss', 'goal': 'minimize'},
    'parameters': {
        'learning_rate': {'values': [0.0001, 0.00025, 0.0005, 0.001]},
        'optimizer': {'values': ['adam', 'sgd']},
        'clip': {'values': [0, 25, 50, 100]},
        'teacher_forcing_ratio': {'values': [0.0, 0.5, 1.0]},
        'decoder_learning_ratio': {'values': [1.0, 3.0, 5.0, 10.0]}
    }
}
```

## Setup

1. Install required dependencies:
   ```bash
   pip install torch wandb
   ```

2. Login to W&B (if not already logged in):
   ```bash
   wandb login
   ```

3. Run the notebook cells in order to:
   - Load and preprocess the dataset
   - Build the encoder-decoder model
   - Train with W&B logging
   - Run hyperparameter sweeps

## Usage

1. **Single Training Run**: Use `run_training_with_wandb()` with custom config
2. **Hyperparameter Sweep**: Uncomment the sweep agent code to run multiple experiments
3. **Analysis**: View results in the W&B dashboard

## Model Architecture

- **Encoder**: Bidirectional GRU with attention
- **Decoder**: GRU with Luong attention mechanism
- **Embedding**: Word embeddings for input/output sequences
- **Attention**: Multiple attention methods (dot, general, concat)

## Training Features

- Teacher forcing ratio scheduling
- Gradient clipping
- Multiple optimizers (Adam, SGD)
- Configurable learning rates for encoder/decoder
- W&B experiment tracking and model monitoring

## Results

The W&B integration provides:
- Real-time loss tracking
- Hyperparameter impact analysis
- Model performance comparisons
- Gradient and parameter monitoring

## Notes

- Large data files (`utterances.jsonl`, `formatted_movie_lines.txt`) are excluded from git
- Model checkpoints are saved locally and can be logged to W&B
- The tutorial is designed to work with the Cornell Movie Dialog dataset
