# CWT-MAE v2: Enhanced Modality-Agnostic Masked Autoencoder

This is the enhanced version (v2) of the CWT-MAE project, incorporating advanced architectural improvements for better performance and efficiency on physiological signals (ECG, PPG, etc.).

## Key Features

1.  **RoPE (Rotary Positional Embeddings)**: Replaces absolute position embeddings to better capture relative timing information in physiological signals.
2.  **Tensorized Linear Layers**: Uses tensor decomposition (low-rank approximation) in the Transformer blocks to significantly reduce parameter count and memory usage without compromising performance.
3.  **Optimized Patch Embedding**: Uses standard Conv2d for memory-efficient patch embedding, resolving issues with large intermediate tensors.
4.  **Latent Reasoning Head (CoT)**: An optional Chain-of-Thought reasoning head for downstream classification, improving performance on complex tasks.
5.  **PPG-Specific Augmentations**: `dataset_cls.py` includes a robust `PPGAugmentor` with flip, scaling, baseline drift, and mask augmentations.

## Project Structure

- `model.py`: Core `CWT_MAE_RoPE` model definition with RoPE and Tensorized blocks.
- `model_finetune.py`: `TF_MAE_Classifier` for downstream tasks, supporting both CoT and MLP heads.
- `dataset.py`: `PhysioSignalDataset` for pre-training, with robust normalization and NaN handling.
- `dataset_cls.py`: `DownstreamClassificationDataset` for fine-tuning, featuring PPG-specific filtering and augmentation.
- `train.py`: Pre-training script supporting DDP, AMP (bfloat16), and compiled models.
- `finetune.py`: Fine-tuning script with Mixup, Threshold Search, and detailed metrics.
- `config.yaml`: Configuration for model hyperparameters and training settings.
- `utils.py`: Utilities for logging, visualization, and layer-wise learning rate decay.

## Requirements

- Python 3.8+
- PyTorch 2.0+ (for `torch.compile` and efficient attention)
- NumPy, SciPy, Matplotlib
- scikit-learn
- PyYAML, tqdm

## Usage

### Pre-training

Configure `config.yaml` (ensure `use_amp: True` for bfloat16 support) and run:

```bash
python train.py --config config.yaml
```

### Fine-tuning

Fine-tune on a specific downstream task (e.g., classification):

```bash
python finetune.py --data_root /path/to/data --split_file /path/to/split.json --pretrained_path ./checkpoints/checkpoint_last.pth --use_cot
```

**Arguments:**
- `--use_cot`: Enable the Latent Reasoning Head.
- `--num_classes`: Number of classes (e.g., 2 for binary).
- `--signal_len`: Length of the input signal (default: 3000).

## Model Configuration

Key parameters in `config.yaml`:
- `mlp_rank_ratio`: Controls the compression rate of Tensorized Linear layers (default: 0.5).
- `mask_ratio`: Pre-training masking ratio (recommended 0.75 for physiological signals).
- `patch_size_time`: Time dimension patch size (default: 50).

## License

[Specify License Here]
