# PatchTST Baseline for Physiological Signals

This project implements the PatchTST model for self-supervised pre-training on physiological signals (ECG, PPG, ACC), adapted from the CWT-MAE v3 project structure.

## Project Structure

- `train.py`: Main training script with DDP and AMP support.
- `patchtst_model.py`: PatchTST model definition.
- `finetune_linear_probe.py`: Linear probing script for downstream classification.
- `dataset.py`: Data loading logic (same as CWT-MAE v3).
- `config.yaml`: Configuration file for hyperparameters.
- `finetune_linear_probe_config.yaml`: Config for downstream linear probing.
- `utils.py`: Helper functions for logging and visualization.
- `utils_metrics.py`: Metrics tracking.

## Requirements

- Python 3.8+
- PyTorch 1.10+
- NumPy, Pandas, Scikit-learn, Matplotlib, PyYAML

## Usage

1. **Data Preparation**:
   Ensure you have the `train_index_cleaned.json` file. This file should contain a list of dictionaries, each with a `path` key pointing to a pickle file containing the signal data.
   
   Update `config.yaml` to point to the correct index file:
   ```yaml
   data:
     index_path: "path/to/train_index_cleaned.json"
   ```

2. **Training**:
   Run the training script:
   ```bash
   python train.py --config config.yaml
   ```

   For distributed training (e.g., on 2 GPUs):
   ```bash
   torchrun --nproc_per_node=2 train.py --config config.yaml
   ```

3. **Visualization**:
   Reconstruction results will be saved in `checkpoints_patchtst/vis_results` during training.

4. **Downstream Linear Probe Classification**:
   ```bash
   python finetune_linear_probe.py --config finetune_linear_probe_config.yaml
   ```
   For distributed training:
   ```bash
   torchrun --nproc_per_node=2 finetune_linear_probe.py --config finetune_linear_probe_config.yaml
   ```

## Configuration

Modify `config.yaml` to adjust:
- `batch_size`: Physical batch size per GPU.
- `accum_iter`: Gradient accumulation steps.
- `patch_len`: Length of time patches.
- `stride`: Stride for patching (non-overlapping if equal to patch_len).
- `mask_ratio`: Masking ratio for pre-training (default 0.6).

## Model Details

The PatchTST model treats each channel independently (Channel Independence) and uses a Transformer backbone to predict masked patches. This implementation supports multi-channel input (default 5 channels: ECG, ACCx3, PPG).
