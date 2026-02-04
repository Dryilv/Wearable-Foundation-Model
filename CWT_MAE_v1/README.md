# CWT-MAE: Modality-Agnostic CWT-Based Masked Autoencoder

This repository contains the implementation of a CWT-based Masked Autoencoder (CWT-MAE) designed for self-supervised learning on physiological signals. The model leverages Continuous Wavelet Transform (CWT) to capture time-frequency features and uses a Transformer-based architecture with Rotary Position Embeddings (RoPE) for representation learning.

## Project Structure

- `model.py`: Defines the core `CWT_MAE_RoPE` model and CWT utility functions.
- `model_finetune.py`: Defines the `TF_MAE_Classifier` for downstream classification tasks, including an optional Latent Reasoning Head (Chain-of-Thought).
- `train.py`: Script for pre-training the CWT-MAE model.
- `finetune.py`: Script for fine-tuning the pre-trained model on downstream classification tasks.
- `dataset.py`: Defines the `PhysioSignalDataset` for pre-training, handling signal loading, preprocessing, and augmentation.
- `dataset_finetune.py`: Defines the `DownstreamClassificationDataset` for fine-tuning tasks.
- `config.yaml`: Configuration file for training parameters.
- `utils.py`: Utility functions for logging, visualization, and distributed training.

## Requirements

- Python 3.8+
- PyTorch 2.0+
- NumPy
- SciPy
- Matplotlib
- PyYAML
- scikit-learn
- tqdm

## Usage

### Pre-training

To pre-train the model, configure the parameters in `config.yaml` and run:

```bash
python train.py --config config.yaml
```

Distributed training is supported and automatically detected if run with `torchrun`.

### Fine-tuning

To fine-tune the model on a classification task, run:

```bash
python finetune.py --data_root /path/to/data --split_file /path/to/split.json --pretrained_path /path/to/checkpoint.pth
```

Key arguments for fine-tuning:
- `--num_classes`: Number of target classes.
- `--use_cot`: Enable the Latent Reasoning Head (Chain-of-Thought).
- `--embed_dim`: Embedding dimension (default: 768).

## Model Architecture

The CWT-MAE model consists of:
1. **CWT Module**: Converts 1D physiological signals into 2D time-frequency representations using Ricker wavelets.
2. **Patch Embedding**: Splits the time-frequency map into patches and projects them into an embedding space.
3. **Transformer Encoder**: Processes the patch embeddings with RoPE and Tensorized Linear layers for efficient computation.
4. **Decoder**: Reconstructs the original signal (or its CWT representation) from the latent representation.

For fine-tuning, the decoder is removed, and a classification head (Linear or CoT) is attached to the encoder output.

## License

[Specify License Here]
