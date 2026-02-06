# CWT-MAE v3 Pretraining Improvements

## Systemic Improvements Overview

This version (v3) introduces a rigorous pretraining evaluation framework designed to ensure model quality, reproducibility, and performance monitoring.

### 1. Data Splitting Strategy (`dataset.py`)
- **Strict Isolation**: 10% of the dataset is reserved as a validation set using `DataSplitter`.
- **Stratified Sampling**: Ensures class distribution consistency between train and val sets.
- **Reproducibility**: Uses MD5 checksums of the index file and strict seeding to guarantee the same split across runs. Split metadata is saved to JSON.

### 2. Performance Optimization & Monitoring (`train.py`)
- **Metrics Tracking**:
  - `grad_norm`: Monitors gradient stability.
  - `throughput`: Tracks samples/second for efficiency analysis.
  - `gpu_mem`: Logs peak GPU memory usage.
  - `loss`: Validation loss is computed every epoch.
- **Dynamic Learning Rate**: Cosine annealing with warmup.

### 3. Feature Quality Verification (`utils_metrics.py`)
- **Periodic Evaluation**: Every 5 epochs, the following are executed:
  - **Linear Probing**: A linear classifier is trained on the frozen encoder outputs to measure representation quality (Accuracy).
  - **Clustering Metrics**: Silhouette Score and Davies-Bouldin Index are calculated to assess feature separability.
  - **t-SNE Visualization**: 2D projection of features is saved to `train_log/tsne_epoch_X.png`.

### 4. Experiment Tracking (`ExperimentTracker`)
- All metrics are logged to `train_log/metrics.csv` for post-hoc analysis.
- **Early Stopping**: Triggered if feature quality (Silhouette Score) does not improve for 3 consecutive checks.

## Usage

1. **Configure**: Update `config.yaml` with your data path.
2. **Train**:
   ```bash
   python CWT_MAE_v3/train.py --config CWT_MAE_v3/config.yaml
   ```
3. **Monitor**:
   - Check `train.log` for real-time progress.
   - View `metrics.csv` for detailed curves.
   - Inspect `tsne_epoch_*.png` for feature evolution.
