# Architectural Specification: Causal Patch Time-Series Transformer (CausalPatchTST)

**Document:** `CAUSAL_PATCH_TST_ARCHITECTURE.md`  
**Location:** `docs/architecture/`  
**Target Module:** `src/models/patch_tst.py`  
**Class:** `src.models.patch_tst.CausalPatchTST`  
**Base Class:** `src.models.base.BasePlantSurrogate`  
**Status:** Implemented, Tested (6/6 spec tests passing), Ready for Vista GH200  

---

## 1. Executive Motivation & Context

In additive cyber-physical attack recovery:
$$\mathbf{y}_t = \mathbf{x}_t + \boldsymbol{\delta}_t$$

Stage 1 estimates the uncorrupted plant state $\hat{x}_t$ purely from companion SCADA telemetry ($\mathbf{y}_{\text{comp}, \le t}$) using a plant dynamics surrogate:
$$\hat{x}_t = f_{\boldsymbol{\theta}}(\mathbf{y}_{\text{comp}, \le t})$$

While the champion Dilated Temporal Convolutional Network (TCN) models linear and polynomial temporal features via exponential dilations ($d \in \{1, 2, 4, 8, 16\}$), high-capacity attention architectures capture complex, non-linear multi-timescale dependencies. 

`CausalPatchTST` adapts the PatchTST architecture (Nie et al., ICLR 2023) for industrial cyber-physical plant estimation, introducing **strict lower-triangular causal masking** to preserve the fundamental past-only identifiability contract.

---

## 2. Core Architectural Pillars

```mermaid
flowchart TD
    A["Raw SCADA Telemetry (Lookback L, C Companion Channels)"] --> B["Subseries Unfold Layer (P=16, S=8)"]
    B --> C["Linear Patch Projection (d_model=64)"]
    C --> D["Learnable 1D Temporal Positional Encoding"]
    D --> E["Channel-Independent Flattening (B * C, Num_Patches, d_model)"]
    E --> F["Transformer Encoder (Strict Upper-Triangular -inf Mask)"]
    F --> G["Latest Context Slicing (Final Patch Representation)"]
    G --> H["Cross-Channel Fusion Head (Linear -> SiLU -> Linear)"]
    H --> I["Target Clean State Prediction (x_hat_t)"]
```

### 2.1 Subseries Patching (`_extract_patches`)
- **Problem with Point-wise Attention**: Point-wise time-series transformers treat individual timesteps as tokens. Single SCADA sensor readings carry high-frequency noise and lack semantic shape. Furthermore, full attention over $L=48\,\text{h}$ lookback has quadratic complexity $O(L^2)$.
- **Patch Extraction**: Consecutive timesteps are grouped into overlapping subseries patches:
  $$\mathbf{p}_{c, i} = \mathbf{x}_{c, [i \cdot S : i \cdot S + P]} \in \mathbb{R}^P$$
  where $P$ is patch length (default 16) and $S$ is stride (default 8).
- **Complexity Reduction**: Reduces token count from $L$ to $N = \lfloor (L - P)/S \rfloor + 1$. Quadratic attention complexity drops from $O(L^2)$ to $O(N^2)$, cutting memory and compute by $(P/S)^2 \approx 4\times$ to $16\times$.

### 2.2 Strict Causal Masking (`encode_patches`)
Standard vision and language transformers allow bidirectional attention. In industrial attack recovery, conditioning on future timesteps ($t > t_0$) causes catastrophic information leakage (absorbing future attack offsets into the clean prediction).

`CausalPatchTST` enforces an exact causal attention mask:
$$\mathbf{M}_{i, j} = \begin{cases} 0, & j \le i \\ -\infty, & j > i \end{cases}$$
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}} + \mathbf{M}\right)\mathbf{V}$$

Each temporal patch attends **only to current and past patches**, guaranteeing mathematical invariance:
$$\frac{\partial \hat{x}_t}{\partial x_{t+k}} = 0 \quad \forall k > 0$$

### 2.3 Channel-Independent Backbone with Cross-Channel Fusion Head
- **Channel-Independent Encoder**: Companion channels share a common transformer encoder. The tensor is reshaped to $(B \cdot C, N, d_{\text{model}})$. This prevents parameter explosion when scaling from BATADAL (43 channels) to WADI (123 channels), while forcing the attention heads to learn generalized temporal wavelets.
- **Cross-Channel Fusion Head**: The latest temporal representations across all companion channels ($c \in \{1, \dots, C\}$) are concatenated at the final patch:
  $$\mathbf{h}_{\text{fusion}} = \bigoplus_{c=1}^C \mathbf{h}_{c, N} \in \mathbb{R}^{C \cdot d_{\text{model}}}$$
  Passed through a non-linear MLP with SiLU activations:
  $$\hat{x}_t = \mathbf{W}_2 \cdot \text{SiLU}(\mathbf{W}_1 \mathbf{h}_{\text{fusion}} + \mathbf{b}_1) + b_2$$
  This layer models physical conservation laws (e.g. mass balance: inflow minus outflow equals rate of change of level).

---

## 3. Mathematical Parameters & Receptive Field

| Hyperparameter | Symbol | Default Value | Role |
| :--- | :--- | :--- | :--- |
| **Input Channels** | $C$ | Dataset-specific (42, 50, 122) | Number of uncorrupted companion sensors |
| **Model Dimension** | $d_{\text{model}}$ | 64 (Scale to 128/256 on Vista) | Latent token representation size |
| **Patch Length** | $P$ | 16 timesteps | Local subseries window duration |
| **Patch Stride** | $S$ | 8 timesteps | Overlap shift between consecutive patches |
| **Attention Heads** | $H$ | 4 heads (Scale to 8/16 on Vista) | Multi-head self-attention parallel projections |
| **Transformer Layers** | $N_{\text{layers}}$ | 3 layers (Scale to 6 on Vista) | Depth of causal transformer encoder |
| **Feedforward Ratio** | $d_{\text{ff}}$ | $4 \times d_{\text{model}} = 256$ | Intermediate dimension in encoder MLP |
| **Max Patches** | $N_{\max}$ | 64 patches | Upper limit on temporal patch capacity |

### Receptive Field Formula
The maximum past lookback span accessible by the patch hierarchy is:
$$R = P + (N_{\max} - 1) \cdot S = 16 + (64 - 1) \cdot 8 = 520\,\text{timesteps}$$
This easily spans the standard 48-hour lookback window ($L=48$) used in BATADAL.

---

## 4. Empirical Comparison: CausalPatchTST vs. DilatedTCN

Measured on BATADAL development telemetry (`data/processed/batadal/train.parquet`):

| Evaluation Dimension | Dilated TCN (`CausalTCN`) | Causal PatchTST (`CausalPatchTST`) | Architectural Trade-off |
| :--- | :--- | :--- | :--- |
| **Local 2-Epoch Time** | **7.8 seconds** | 72.3 seconds | TCN is 9× faster on single CPU; PatchTST parallelizes on GH200 GPU |
| **Validation Loss (MSE)** | **0.01250** | 0.01280 | Comparable convergence on small initial epochs |
| **Validation Residual Std ($\sigma_\epsilon$)** | **0.1356 m** | 0.1372 m | Both achieve tight normal plant residual noise |
| **Parameter Scaling** | $O(C \cdot k)$ linear | $O(C \cdot d_{\text{model}} + N_{\text{layers}} \cdot d_{\text{model}}^2)$ | PatchTST scales compute with model capacity, ideal for burning Vista SUs |
| **Temporal Inductive Bias** | Fixed convolutional filter | Dynamic data-dependent attention weights | PatchTST adapts dynamically to diurnal and weekly industrial shifts |
| **Zero Lookahead Guarantee** | Causal convolution padding | Lower-triangular $-\infty$ attention mask | Both strictly satisfy past-only causal identifiability |

---

## 5. Verification & Unit Testing

The implementation is verified by the locked unit test suite in [tests/spec/test_surrogate_models.py](file:///C:/Users/hifia/Projects/Sengupta_Research/tests/spec/test_surrogate_models.py):

1. **`test_base_surrogate_interface_enforcement`**: Verifies ABC instantiation and abstract method contracts.
2. **`test_tcn_output_shape_and_receptive_field`**: Verifies TCN output dimensions $[B, 1]$ and $R \ge 48$.
3. **`test_patch_tst_output_shape_and_receptive_field`**: Verifies PatchTST output dimensions $[B, 1]$ and patch unfolding.
4. **`test_strict_causal_invariance_tcn`**: Asserts future companion perturbations produce exactly 0.0 representation shift.
5. **`test_strict_causal_invariance_patch_tst`**: Asserts future companion perturbations produce exactly 0.0 representation shift in the causal attention mask.
6. **`test_tcn_state_dict_backward_compatibility`**: Asserts exact parameter matching and loading of frozen champion checkpoint `tcn.pt`.
