# RGD-Triton

Collection of Triton operators for transformer models.

> **Note:** Code will be uploaded within a few days following internal reorganization.

---

## Dependencies

- GPUs: Tested on Quadro RTX 4000 and A6000
- PyTorch: 2.5.1
- Triton: 3.1.0

---

## Structure

All operators are implemented as parameter-less `torch.nn.Module` subclasses. Each module supports both forward and backward passes unless otherwise noted.

This design enables:
- Persistent cache state to bypass Triton's kernel lookup overhead.
- Efficient chaining of operations using single kernel launches via virtual tensors (e.g., fused QK normalization).

---

## Operations

#### Modules without Parameters

- **FlashAttention**  
  Based on [Tri Dao's module](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py). Updated to work on a standard Triton branch and GPUs other than A100. Supports arbitrary bias tensors (including of shape B,h,Q,K), enabling flexible and dynamic masking strategies.

- **L2Normalize**  
  Performs L2 normalization over the last dimension with internal promotion to FP32 for stability.

- **FusedL2Normalize2**  
  Applies L2 normalization on two tensors treated as a single virtual tensor, reducing launch overhead. Useful for computing cosine similarity.

- **FusedHeadLayerNorm**  
  Applies layer normalization to Q and K tensors in a single kernel launch. Does not support affine parameters.

- **RoPE**  
  Implements partial-head rotary position embeddings (rotary on the first R channels) with selective sequence application (on first N, M tokens). Applies both Q and K in a single kernel launch.

- **FusedLNRoPE**  
  Combines FusedHeadLayerNorm and RoPE into a single kernel launch.

#### Modules with Parameters

- **FusedRMSNorm**  
  Applies RMS normalization with optional affine weights. Kernel fusion and internal FP32 promotion improve performance over Torch and Apex, especially under AMP.

- **FusedRMSNorm2**  
  Same as FusedRMSNorm but applies normalization to two tensors simultaneously. Each tensor can have its own affine parameters. Useful for multi-modal models like MMDiT.

---

## Testing

PyTest-based unit tests are included to verify numerical correctness of both forward and backward passes, along with detection of race conditions. All operations support FP16, BF16, and FP32, except for FlashAttention, which is limited to 16-bit types.

---

## Optimization

These operations do not currently use Triton’s auto-tuning features. Launch configurations were selected based on empirical performance on an A6000 GPU, without exhaustive hyperparameter search. Performance may vary on other hardware.

## License

This repository is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

The `FlashAttention` operator is adapted from [Tri Dao's FlashAttention implementation](https://github.com/Dao-AILab/flash-attention), originally licensed under the **BSD-3-Clause** license. The original copyright and license notice have been retained in the source file in accordance with the terms of that license. See `LICENSE.BSD-3-Clause` for details.

