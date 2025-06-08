import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from rgd_triton.triton.normalization.fused_l2normalize_cached import triton_fused_norm_func_fn 


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
        
def ref_fused_norm(x1,x2, eps=1e-6, upscast=True):
    orig_dtype = x1.dtype
    if upscast:
        x1 = x1.float()
        x2 = x2.float()
        
    x1l = F.normalize(x1, p=2, eps=eps, dim=-1)
    x2l = F.normalize(x2, p=2, eps=eps, dim=-1)
    return x1l.to(orig_dtype), x2l.to(orig_dtype)
    

def make_inputs(B,NQ,NK,CQ,CK, do_shift=False, do_zero_one=False, device="cuda", dtype=torch.float):

    x1 = torch.randn(B,NQ,CQ, device=device, dtype=dtype)
    x2 = torch.randn(B,NK,CK, device=device, dtype=dtype)    

    if do_shift:
        bias_x1 = torch.rand_like(x1)
        bias_x2 = torch.rand_like(x2)
        scale_x1 = torch.rand_like(x1)*100
        scale_x2 = torch.rand_like(x2)*100

        x1 = x1*scale_x1 + bias_x1
        x2 = x2*scale_x2 - bias_x2
    if do_zero_one:
        x1 = torch.zeros_like(x1)
        x2 = torch.ones_like(x2)
        
    x1 = x1.clone().detach().requires_grad_(True)
    x2 = x2.clone().detach().requires_grad_(True)
    
    return x1,x2

def safe_assert_close_allow_inf(actual, expected, rtol=1e-5, atol=1e-8, msg=None):
    if actual.shape != expected.shape:
        raise AssertionError(f"Shape mismatch: {actual.shape} vs {expected.shape}")

    # Check for NaNs – we do NOT allow them
    if torch.any(torch.isnan(actual)) or torch.any(torch.isnan(expected)):
        raise AssertionError("Found NaNs in actual or expected tensors")

    # +inf check
    posinf_actual = actual == float('inf')
    posinf_expected = expected == float('inf')
    if not torch.equal(posinf_actual, posinf_expected):
        mismatch = (posinf_actual != posinf_expected).nonzero(as_tuple=False)
        idx = tuple(mismatch[0].tolist())
        raise AssertionError(
            f"Mismatch in positions of +inf values at index {idx}: "
            f"actual={actual[idx].item()}, expected={expected[idx].item()}"
        )

    # -inf check
    neginf_actual = actual == float('-inf')
    neginf_expected = expected == float('-inf')
    if not torch.equal(neginf_actual, neginf_expected):
        mismatch = (neginf_actual != neginf_expected).nonzero(as_tuple=False)
        idx = tuple(mismatch[0].tolist())
        raise AssertionError(
            f"Mismatch in positions of -inf values at index {idx}: "
            f"actual={actual[idx].item()}, expected={expected[idx].item()}"
        )

    # Mask out positions where both tensors have infs of the same sign
    inf_mask = posinf_actual | neginf_actual
    filtered_actual = actual[~inf_mask]
    filtered_expected = expected[~inf_mask]

    # Compare finite values, catch and enhance error
    try:
        torch.testing.assert_close(
            filtered_actual,
            filtered_expected,
            rtol=rtol,
            atol=atol,
            msg=msg
        )
    except AssertionError as e:
        # Find first mismatch
        diff = torch.abs(filtered_actual - filtered_expected)
        mismatch = (diff > (atol + rtol * torch.abs(filtered_expected))).nonzero(as_tuple=False)
        if mismatch.numel() > 0:
            i = mismatch[0].item()
            a_val = filtered_actual[i].item()
            e_val = filtered_expected[i].item()
            extra_info = f"\nFirst mismatch at flat index {i}: actual={a_val}, expected={e_val}"
        else:
            extra_info = "\nFailed, but could not locate specific mismatch"

        # Re-raise with added context
        raise AssertionError(str(e) + extra_info) from e
    

@pytest.mark.parametrize("dtype", ([torch.float32, torch.float16] if is_sm75 else [torch.float32, torch.float16, torch.bfloat16]))
@pytest.mark.parametrize('d', [32, 63, 97, 122, 257, 384, 510, 740, 1023])
#@pytest.mark.parametrize('zero_one', [True, False])
@pytest.mark.parametrize('zero_one', [False])
@pytest.mark.parametrize("seqlen", [14, 31, 97, 128, 200, 384, 768, 1024] )
def test_fused_l2_norm(seqlen, d, zero_one, dtype):
    if seqlen >= 2048 and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4

    print(f"{batch_size=}, {seqlen=}, {d=}, {zero_one=}, {dtype=}")
    
    q,k = make_inputs(B=batch_size,
                      NQ=seqlen,
                      NK=seqlen,
                      CQ=d, 
                      CK=d,
                      do_shift=True, 
                      do_zero_one=zero_one, 
                      device=device, 
                      dtype=dtype
                     )

    print(f"{q.shape=}, {k.shape=}")
    
    qo, ko = triton_fused_norm_func_fn(q, k)

    qo_ref, ko_ref = ref_fused_norm(
        q,
        k,
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
    safe_assert_close_allow_inf(dq, dq_ref, rtol=rtol, atol=atol, msg="Mismatch in dq")
    safe_assert_close_allow_inf(dk, dk_ref, rtol=rtol, atol=atol, msg="Mismatch in dk")


@pytest.mark.parametrize("dtype", ([torch.float32, torch.float16] if is_sm75 else [torch.float32, torch.float16, torch.bfloat16]))
#@pytest.mark.parametrize('zero_one', [True, False])
@pytest.mark.parametrize('zero_one', [False])
@pytest.mark.parametrize('d', [32, 63, 97, 122, 257, 384, 510, 740, 1023])
@pytest.mark.parametrize('dfactor', [-0.75,-0.3,0.2,0.6,1.0])
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
def test_fused_l2_norm_seqdiff(
    seqlen_q, seqlen_k, d, zero_one, dfactor, dtype
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

    if dfactor < 0.0:
        # set q to be smaller
        CQ = int((-dfactor)*d)
        CK = d
    else:
        # set k to be smaller
        CQ = d
        CK = int(dfactor*d)
        
    
    print(f"{batch_size=}, {seqlen_q=}, {seqlen_k=}, {CQ=}, {CK=}, {zero_one=}, {dtype=} ")
    
    q,k = make_inputs(B=batch_size,
                      NQ=seqlen_q,
                      NK=seqlen_k,
                      CQ=CQ, 
                      CK=CK,
                      do_shift=True, 
                      do_zero_one=zero_one, 
                      device=device, 
                      dtype=dtype
                     )

    qo, ko = triton_fused_norm_func_fn(q, k)

    qo_ref, ko_ref = ref_fused_norm(
        q,
        k,
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
    safe_assert_close_allow_inf(dq, dq_ref, rtol=rtol, atol=atol, msg="Mismatch in dq")
    safe_assert_close_allow_inf(dk, dk_ref, rtol=rtol, atol=atol, msg="Mismatch in dk")

@pytest.mark.focus
@pytest.mark.parametrize("dtype", ([torch.float32, torch.float16] if is_sm75 else [torch.float32, torch.float16, torch.bfloat16]))
#@pytest.mark.parametrize('zero_one', [True, False])
@pytest.mark.parametrize('zero_one', [False])
@pytest.mark.parametrize('dfactor', [-0.75,-0.3,0.2,0.6,1.0])
@pytest.mark.parametrize('d', [32, 63, 97, 122, 257, 384, 510, 740, 1023])
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
def test_fused_l2_norm_race_condition(seqlen_q, seqlen_k, d, dfactor, zero_one, dtype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 60  # Sometimes we need large batch size for the race conditions to trigger

    if dfactor < 0.0:
        # set q to be smaller
        CQ = int((-dfactor)*d)
        CK = d
    else:
        # set k to be smaller
        CQ = d
        CK = int(dfactor*d)
        
    
    print(f"{batch_size=}, {seqlen_q=}, {seqlen_k=}, {CQ=}, {CK=}, {zero_one=}, {dtype=} ")
    
    q,k = make_inputs(B=batch_size,
                      NQ=seqlen_q,
                      NK=seqlen_k,
                      CQ=CQ, 
                      CK=CK,
                      do_shift=True, 
                      do_zero_one=zero_one, 
                      device=device, 
                      dtype=dtype
                     )
        
    torch.random.manual_seed(42)

    qo0, ko0 = triton_fused_norm_func_fn(q, k)
    
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
        qo, ko = triton_fused_norm_func_fn(q, k)
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

        
