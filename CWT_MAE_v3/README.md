# CWT-MAE v3: Thorough Solution for Large-Scale Physiological Pre-training

This is the v3 version of the CWT-MAE project, designed specifically for **large-scale (20k hours)** physiological signal pre-training. It addresses the overfitting and frequency leakage issues found in v1/v2 by introducing a more rigorous architectural design.

## Key Features (The "Thorough Solution")

### 1. Vertical Patching
-   **Concept**: Instead of 2D image patches, we use **Vertical Strips** covering the entire frequency range for a specific time window.
-   **Mechanism**: `patch_size_freq` is forced to equal `cwt_scales` (e.g., 64).
-   **Benefit**: Masking a patch now removes **all** frequency information for that time step. This prevents the model from "cheating" by interpolating from visible frequencies at the same timestamp, forcing it to learn temporal dependencies.

### 2. Dual-Objective Reconstruction
-   **Objectives**: The model must reconstruct **BOTH**:
    1.  The CWT Spectrogram (Frequency Domain)
    2.  The Raw Time-domain Signal (Time Domain)
-   **Benefit**: Ensures the model learns fine-grained phase and amplitude details, not just the fuzzy spectral shape.
-   **Loss**: `Total_Loss = Loss_Spec + 2.0 * Loss_Raw`.

### 3. RevIN (Reversible Instance Normalization)
-   **Component**: A learnable, reversible normalization layer applied directly to the input signal.
-   **Benefit**: Handles the statistical distribution shifts between different modalities (ECG, PPG, ACC) and subjects effectively, removing the need for manual static normalization.

### 4. Optimized Model Scaling
-   **Configuration**: Downscaled to a "Small/Medium" size to prevent overfitting on signal data while maintaining depth for non-linearity.
    -   `embed_dim`: 384 (vs 768 in v2)
    -   `num_heads`: 6 (vs 12 in v2)
    -   `decoder_embed_dim`: 256
    -   `mask_ratio`: 0.8 (High masking for high redundancy signals)

## Project Structure

-   `model.py`: Core `CWT_MAE_RoPE` v3 model with RevIN, Vertical Patching, and Dual Heads.
-   `train.py`: Pre-training script updated for v3 architecture.
-   `config.yaml`: Configuration file with optimized hyperparameters for v3.
-   `dataset.py`: (Shared/Copied) Standard dataset loader.
-   `utils.py`: (Shared/Copied) Utility functions.

## Usage

### Pre-training

```bash
cd CWT_MAE_v3
python train.py --config config.yaml
```

**Note**: Ensure `config.yaml` points to the correct dataset index path.

## Comparison: v2 vs v3

| Feature | v2 (Enhanced) | v3 (Thorough) |
| :--- | :--- | :--- |
| **Patching** | 2D (Time x Freq) | Vertical (Time-only strips) |
| **Masking** | Random 2D Blocks | Full Time-Step Masking |
| **Reconstruction** | Spectrogram + Time Aggregation | Dual Head (Spec + Raw Direct) |
| **Normalization** | Manual Instance Norm | Learnable RevIN |
| **Model Size** | Base (768 dim) | Small/Medium (384 dim) |
| **Target Data** | General | Large-Scale (20k hrs) |

## License

[Specify License Here]
