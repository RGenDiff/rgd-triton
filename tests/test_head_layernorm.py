import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from rgd_triton.triton.normalization.head_layernorm_cached import triton_fused_ln_func_fn 


is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)

def get_tolerance(dtype):
    if dtype == torch.float32:
        return 1.3e-6, 1e-5 # from torch doc
        #return 1e-3, 1e-5
    elif dtype == torch.float16:
        return 1e-3, 1e-5 # from torch doc
        #return 1e-2, 1e-3
    elif dtype == torch.bfloat16:
        return 1.6e-2, 1e-5 # from torch doc
        #return 1e-2, 1e-3
    else:
        raise ValueError(f"Unhandled dtype: {dtype}")
        
def ref_fused_layernorm(
    q, 
    k, 
    heads_second=True,
    upcast=True,
    eps=1e-6
):
    dtype_og = q.dtype
    
    if upcast:
        q, k = q.float(), k.float()
    
    ql = F.layer_norm(q, (q.shape[-1],), weight=None, bias=None, eps=eps)
    kl = F.layer_norm(k, (k.shape[-1],), weight=None, bias=None, eps=eps)
    
    return ql.to(dtype=dtype_og), kl.to(dtype=dtype_og)
    

def make_inputs(B,H,NQ,NK,C, do_shift=False, heads_second=True, device="cuda", dtype=torch.float):
    if heads_second:
        q = torch.randn(B,H,NQ,C, device=device, dtype=dtype)
        k = torch.randn(B,H,NK,C, device=device, dtype=dtype)    
    else:
        q = torch.randn(B,NQ,H,C, device=device, dtype=dtype)
        k = torch.randn(B,NK,H,C, device=device, dtype=dtype)      

    if do_shift:
        bias_q = torch.rand_like(q)
        bias_k = torch.rand_like(k)
        scale_q = torch.rand_like(q)
        scale_k = torch.rand_like(k)

        q = q*scale_q + bias_q
        k = k*scale_k - bias_k

    # enable gradients
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)

    return q,k


@pytest.mark.parametrize("dtype", ([torch.float32, torch.float16] if is_sm75 else [torch.float32, torch.float16, torch.bfloat16]))
@pytest.mark.parametrize('d', [32, 64, 96, 128])
@pytest.mark.parametrize('heads_second', [True, False])
@pytest.mark.parametrize("seqlen", [14, 31, 97, 128, 200, 384, 768, 1024] )
def test_fused_head_ln(seqlen, d, heads_second, dtype):
    if seqlen >= 2048 and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    q,k = make_inputs(B=batch_size,
                      H=nheads,
                      NQ=seqlen,
                      NK=seqlen,
                      C=d, 
                      do_shift=True, 
                      heads_second=heads_second, 
                      device=device, 
                      dtype=dtype
                     )
    
    qo, ko = triton_fused_ln_func_fn(q, k, heads_second)

    qo_ref, ko_ref = ref_fused_layernorm(
        q,
        k,
        heads_second,
        upcast=True,
    )
        


    print(f"Q Output max diff: {(qo - qo_ref).abs().max().item()}")
    print(f"K Output max diff: {(ko - ko_ref).abs().max().item()}")
    print(f"Q Output mean diff: {(qo - qo_ref).abs().mean().item()}")
    print(f"K Output mean diff: {(ko - ko_ref).abs().mean().item()}")

    gq = torch.randn_like(qo)
    gk = torch.randn_like(ko)
    
    dq, dk = torch.autograd.grad(outputs=(qo, ko), inputs=(q, k), grad_outputs=(gq, gk))
    dq_ref, dk_ref = torch.autograd.grad(outputs=(qo_ref, ko_ref), inputs=(q, k), grad_outputs=(gq, gk))
    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
    print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
    
    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    rtol, atol = get_tolerance(qo.dtype)  # Use the dtype-sensitive function we discussed
    
    torch.testing.assert_close(qo, qo_ref, rtol=rtol, atol=atol, msg="Mismatch in qo")
    torch.testing.assert_close(ko, ko_ref, rtol=rtol, atol=atol, msg="Mismatch in ko")
    torch.testing.assert_close(dq, dq_ref, rtol=rtol, atol=atol, msg="Mismatch in dq")
    torch.testing.assert_close(dk, dk_ref, rtol=rtol, atol=atol, msg="Mismatch in dk")


@pytest.mark.parametrize("dtype", ([torch.float32, torch.float16] if is_sm75 else [torch.float32, torch.float16, torch.bfloat16]))
@pytest.mark.parametrize('heads_second', [True, False])
@pytest.mark.parametrize('d', [32, 64, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (31, 11),
        (23, 138),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (512, 256),
        (1024, 1024),
        (1023, 1024),
        (1024, 1023),
        (2048, 2048),
    ],
)
def test_fused_head_ln_seqdiff(
    seqlen_q, seqlen_k, d, heads_second, dtype
):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 8 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM

    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 4
    q,k = make_inputs(B=batch_size,
                      H=nheads,
                      NQ=seqlen_q,
                      NK=seqlen_k,
                      C=d, 
                      do_shift=True, 
                      heads_second=heads_second, 
                      device=device, 
                      dtype=dtype
                     )

    qo, ko = triton_fused_ln_func_fn(q, k, heads_second)

    qo_ref, ko_ref = ref_fused_layernorm(
        q,
        k,
        heads_second,
        upcast=True,
    )
        

    print(f"Output Q max diff: {(qo - qo_ref).abs().max().item()}")
    print(f"Output K max diff: {(ko - ko_ref).abs().max().item()}")
    print(f"Output Q mean diff: {(qo - qo_ref).abs().mean().item()}")
    print(f"Output K mean diff: {(ko - ko_ref).abs().mean().item()}")

    gq = torch.randn_like(qo)
    gk = torch.randn_like(ko)
    #do_o = (g.float() * out.float()).sum(-1)
    (
        dq,
        dk,
    ) = torch.autograd.grad(outputs=(qo, ko), inputs=(q, k), grad_outputs=(gq, gk))
    (
        dq_ref,
        dk_ref,
    ) = torch.autograd.grad(outputs=(qo_ref, ko_ref), inputs=(q, k), grad_outputs=(gq, gk))

    print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
    print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
    print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
    print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")

    rtol, atol = get_tolerance(qo.dtype)  # Use the dtype-sensitive function we discussed
    
    torch.testing.assert_close(qo, qo_ref, rtol=rtol, atol=atol, msg="Mismatch in qo")
    torch.testing.assert_close(ko, ko_ref, rtol=rtol, atol=atol, msg="Mismatch in ko")
    torch.testing.assert_close(dq, dq_ref, rtol=rtol, atol=atol, msg="Mismatch in dq")
    torch.testing.assert_close(dk, dk_ref, rtol=rtol, atol=atol, msg="Mismatch in dk")


@pytest.mark.parametrize("dtype", ([torch.float32, torch.float16] if is_sm75 else [torch.float32, torch.float16, torch.bfloat16]))
@pytest.mark.parametrize('heads_second', [True, False])
@pytest.mark.parametrize('d', [32, 64, 128])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (239, 1),
        (3, 799),
        (799, 3),
        (1024, 128),
        (97, 97),
        (128, 128),
        (200, 200),
        (256, 256),
        (257, 257),
        (384, 384),
        (512, 512),
        (768, 768),
        (1024, 1024),
    ],
)
def test_fused_head_ln_race_condition(seqlen_q, seqlen_k, d, heads_second, dtype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 60  # Sometimes we need large batch size for the race conditions to trigger
    nheads = 4
    
    q,k = make_inputs(B=batch_size,
                      H=nheads,
                      NQ=seqlen_q,
                      NK=seqlen_k,
                      C=d, 
                      do_shift=True, 
                      heads_second=heads_second, 
                      device=device, 
                      dtype=dtype
                     )
        
    torch.random.manual_seed(42)

    qo0, ko0 = triton_fused_ln_func_fn(q, k, heads_second)
    
    gq = torch.randn_like(qo0)
    gk = torch.randn_like(ko0)

    (
        dq0,
        dk0,
    ) = torch.autograd.grad((qo0,ko0), (q, k), (gq,gk))
    # Numerical error if we just do any arithmetic on dq
    dq_atol = 2 * ((dq0 + 0.3 - 0.3) - dq0).abs().max().item()

    for i in range(250):
        torch.random.manual_seed(42)
        qo, ko = triton_fused_ln_func_fn(q, k, heads_second)
        #out, lse, _ = flash_attn_func(q, k, v, dropout_p, causal=causal, return_attn_probs=True)
        assert torch.equal(qo, qo0)
        assert torch.equal(ko, ko0)


        (
            dq,
            dk,
        ) = torch.autograd.grad((qo,ko), (q, k), (gq,gk))
        
        dq_equal = torch.allclose(dq, dq0, atol=dq_atol)
        if not dq_equal:
            print(f"Iter {i}, {dq_atol = }, dQ max diff: {(dq - dq0).abs().max().item()}")
        assert torch.equal(dk, dk0)
        assert dq_equal

        
