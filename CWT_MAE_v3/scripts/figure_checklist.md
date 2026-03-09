# Figure Checklist & Visualization Guidelines

To meet the high standards of top-tier journals (e.g., Nature Medicine, ICLR, NeurIPS), all figures must adhere to the following strict formatting and aesthetic guidelines.

## General Requirements
*   **Format**: Vector graphics only (PDF or EPS). Do not use raster formats (PNG/JPG) unless displaying raw data samples.
*   **Font**: Helvetica or Arial, $\ge$ 8 pt. ensure legibility when resized.
*   **Color Palette**: Must be colorblind-friendly (e.g., Viridis, Okabe-Ito, or Nature Publishing Group palettes). Avoid Red/Green combinations.
*   **Resolution**: $\ge$ 300 DPI for any rasterized components.
*   **Layout**: Single-column (89 mm) or double-column (183 mm) width.

---

## Figure 1: The CWT-MAE-RoPE Foundation Model Framework (Conceptual Overview)
*   **Goal**: Illustrate the "End-to-End" workflow from 1D signals to downstream tasks.
*   **Components**:
    *   **Left Panel (Input)**: Show raw ECG/PPG waveforms + CWT transformation process (1D $\to$ 2D).
    *   **Middle Panel (Pre-training)**:
        *   Show Patch Embedding & Random Masking (visualize missing patches).
        *   Show the **Single-Tower Encoder** with shared weights for ECG/PPG.
        *   Illustrate **RoPE** mechanism (rotating vectors).
        *   Show the **Dual Reconstruction Heads** (Frequency Decoder & Time Reducer).
        *   Highlight the **Cross-Modal Contrastive Loss** (pulling positive ECG-PPG pairs together).
    *   **Right Panel (Downstream)**: Show the Fine-tuning adaptation (Frozen Encoder + CoT/MLP Heads).

## Figure 2: Multi-View Signal Representation & Processing Pipeline
*   **Goal**: Explain *how* the data is prepared and why CWT + Differential Channels work.
*   **Components**:
    *   **Row 1 (Raw Signal)**: A clean ECG segment showing P-QRS-T complex.
    *   **Row 2 (1st Derivative)**: Highlight the sharp slopes (R-peaks).
    *   **Row 3 (2nd Derivative)**: Highlight the inflection points.
    *   **Row 4 (CWT Map)**: A heatmap visualization of the wavelet transform (Time vs. Frequency).
    *   **Mechanism**: Show the `DecomposedPatchEmbed` convolution operation sliding over the CWT map.

## Figure 3: Detailed Architecture: Masking, RoPE, and Dual-Loss
*   **Goal**: A technical deep-dive into the model's internal mechanics.
*   **Components**:
    *   **Masking Strategy**: Visual diagram of `Tubelet Masking` vs. `Random Masking`.
    *   **RoPE Attention**: Mathematical visualization of the rotation operation ($q e^{i\theta}, k e^{i\theta}$).
    *   **Tensorized Linear Layer**: Show the decomposition $W = U \times V$ with a bottleneck rank.
    *   **Loss Flow**:
        *   $\mathcal{L}_{spec}$: MSE on masked patches (Frequency domain).
        *   $\mathcal{L}_{time}$: MSE on recovered 1D signal (Time domain).

## Figure 4: Pre-training Dynamics & Convergence
*   **Goal**: Prove training stability and the effectiveness of the 3-stage curriculum.
*   **Sub-plots**:
    *   **(a) Loss Curves**: Total Loss, MAE Loss, Contrastive Loss over epochs.
    *   **(b) Stage Transitions**: Mark the boundaries of Stage 1 (Warmup), Stage 2 (Transition), and Stage 3 (Joint Optimization).
    *   **(c) Alignment Score**: A metric (e.g., cosine similarity of positive pairs) increasing over time.

## Figure 5: Qualitative Reconstruction Results (The "Wow" Factor)
*   **Goal**: Show that the model "understands" the signal structure.
*   **Layout**:
    *   **Top Row (ECG)**: Original | Masked Input (75% missing) | Reconstructed Spec | Reconstructed Waveform.
    *   **Bottom Row (PPG)**: Same layout for PPG.
    *   **Highlights**: Zoom in on a QRS complex or a dicrotic notch to show fine-grained recovery.

## Figure 6: Downstream Performance Benchmarks (Radar/Bar Charts)
*   **Goal**: Quantitative superiority over baselines.
*   **Sub-plots**:
    *   **(a) AUROC/F1 Comparison**: Bar chart comparing CWT-MAE-RoPE vs. TCN, LSTM, ResNet, MOMENT.
    *   **(b) Low-Data Regime**: Line plot showing performance vs. % of labeled training data (Few-shot learning capability).
    *   **(c) Latency vs. Accuracy**: Scatter plot showing our model is efficient (especially with Tensorized layers).

## Figure 7: t-SNE Visualization of Learned Embeddings
*   **Goal**: Demonstrate class separability.
*   **Content**: 2D scatter plot of the [CLS] token embeddings for the test set, colored by class labels (e.g., Normal vs. Afib).
*   **Expected Outcome**: Clear clusters for different arrhythmia types, showing the model learned discriminative features.

## Figure 8: Ablation Study & Sensitivity Analysis (New)
*   **Goal**: Robustness check and component justification.
*   **Sub-plots**:
    *   **(a) Masking Ratio Sensitivity**: Performance (F1-score) vs. Masking Ratio (0.1 to 0.9). Show error bars ($\mu \pm \sigma$).
    *   **(b) Component Ablation**: Bar chart showing performance drop when removing CWT, RoPE, or Contrastive Loss.
    *   **(c) Input Length Robustness**: Performance vs. Input Sequence Length (Extrapolation capability).

## Figure 9: Computational Complexity Analysis (New)
*   **Goal**: Demonstrate efficiency.
*   **Sub-plots**:
    *   **(a) FLOPs vs. Sequence Length**: Comparison between Vanilla ViT and CWT-MAE-RoPE (with Tensorized Layers).
    *   **(b) Memory Footprint**: GPU memory usage vs. Batch Size.
