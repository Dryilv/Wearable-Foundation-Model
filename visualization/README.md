# Visualization Tools

This folder contains scripts for visualizing physiological signals and model reconstructions.

## Scripts

### 1. `visualize_data.py`
Visualizes raw physiological signals from the dataset index.
- **Usage**: `python visualize_data.py --index path/to/index.json`
- **Features**: Interactive viewer for checking signal quality and channel alignment.

### 2. `visualize_recon.py`
Visualizes the reconstruction capability of the pre-trained CWT-MAE model on the training/test dataset.
- **Usage**: `python visualize_recon.py --checkpoint path/to/checkpoint.pth --config path/to/config.yaml`
- **Output**: Generates plots showing Original Signal, Original CWT, Masked Input, and Reconstructed CWT.

### 3. `visualize_recon_data.py`
Runs inference and visualization on a *new* or custom signal file (not in the dataset index).
- **Usage**: `python visualize_recon_data.py --checkpoint path/to/checkpoint.pth --input_file path/to/signal.pkl`

## Note on Model Version
By default, these scripts are configured to use the **v2** model architecture (`../CWT_MAE_v2`). If you need to visualize **v1** models, please update the `sys.path.append` line in the scripts to point to `../CWT_MAE_v1`.
