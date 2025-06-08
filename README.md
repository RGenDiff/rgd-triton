# RGD-Triton

Collection of Triton operators for transformer models.


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

## Usage

Import operators as needed from ops.

```python

import torch
import torch.nn as nn
from rdg_triton.ops import FusedHeadLayerNorm, FlashAttention

class Attention(nn.Module):
    def __init__(self, dim, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads

        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.out_proj = nn.Linear(dim, dim)
        self.attn_op = FlashAttention()
        self.head_norm = FusedHeadLayerNorm()

    def forward(self, x, bias=None):
        B, T, C = x.shape
        q, k, v = (
            self.qkv(x)
            .view(B, T, 3, self.n_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
            .unbind(dim=0)
        )

        q, k = self.head_norm(q, k)
        attn_out = self.attn_op(q, k, v, bias=bias)

        attn_out = attn_out.transpose(1, 2).reshape(B, T, C)
        return self.out_proj(attn_out)
```

Caching can be enabled or disabled using each rgd_triton.ops member function.

```python
for module in model.modules():
    if hasattr(module, "enable_cache"):
        module.enable_cache(enabled) # True or False
```

The cache is enabled by default and will be filled by the first kernel launch, and can be cleared by setting enable=False. 


---


## Operations

#### Modules without Parameters

- **FlashAttention**  
  Based on [FlashAttention-Triton](https://github.com/Dao-AILab/flash-attention/blob/main/flash_attn/flash_attn_triton.py). Updated to work on a standard Triton branch and GPUs other than A100. Supports arbitrary bias tensors (including of shape B,h,Q,K), enabling flexible and dynamic masking strategies.

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

---

## TODO

- [ ] Track down numerical instability with FusedRMSNorm2 (only occurs with bf16 weights)

---

## License

This repository is licensed under the **GNU Affero General Public License v3.0 (AGPL-3.0)**.

The `FlashAttention` operator is adapted from [FlashAttention](https://github.com/Dao-AILab/flash-attention), originally licensed under the **BSD-3-Clause** license. The original copyright and license notice have been retained in the source file in accordance with the terms of that license. See `LICENSE.BSD-3-Clause` for details.

