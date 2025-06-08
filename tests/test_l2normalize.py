import math

import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from rgd_triton.triton.normalization.l2normalize_cached import triton_norm_func_fn 


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
        
def ref_norm(x, eps=1e-6, upscast=True):
    orig_dtype = x.dtype
    if upscast:
        x = x.float()
        
    xl = F.normalize(x, p=2, eps=eps, dim=-1)
    return xl.to(orig_dtype)
    

def make_inputs(B,N,C, do_shift=False, do_zero_one=False, device="cuda", dtype=torch.float):

    x = torch.randn(B,N,C, device=device, dtype=dtype)

    if do_shift:
        bias_x = torch.rand_like(x)
        scale_x = torch.rand_like(x)*100

        x = x*scale_x + bias_x

    if do_zero_one:
        x = torch.zeros_like(x)
        
    x = x.clone().detach().requires_grad_(True)

    return x

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
def test_l2_norm(seqlen, d, zero_one, dtype):
    if seqlen >= 2048 and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4

    print(f"{batch_size=}, {seqlen=}, {d=}, {zero_one=}, {dtype=}")
    
    x = make_inputs(B=batch_size,
                      N=seqlen,
                      C=d, 
                      do_shift=True, 
                      do_zero_one=zero_one, 
                      device=device, 
                      dtype=dtype
                     )

    print(f"{x.shape=}")
    
    xo = triton_norm_func_fn(x)

    xo_ref = ref_norm(
        x,
    )
        


    print(f"X Output max diff: {(xo - xo_ref).abs().max().item()}")
    print(f"X Output mean diff: {(xo - xo_ref).abs().mean().item()}")

    gx = torch.randn_like(xo)
    
    dx = torch.autograd.grad(outputs=(xo,), inputs=(x,), grad_outputs=(gx,))[0]
    dx_ref = torch.autograd.grad(outputs=(xo_ref,), inputs=(x,), grad_outputs=(gx,))[0]
    print(f"dX max diff: {(dx - dx_ref).abs().max().item()}")
    print(f"dX mean diff: {(dx - dx_ref).abs().mean().item()}")

    
    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    rtol, atol = get_tolerance(xo.dtype)  # Use the dtype-sensitive function we discussed
    
    torch.testing.assert_close(xo, xo_ref, rtol=rtol, atol=atol)
    safe_assert_close_allow_inf(dx, dx_ref, rtol=rtol, atol=atol)



@pytest.mark.focus
@pytest.mark.parametrize("dtype", ([torch.float32, torch.float16] if is_sm75 else [torch.float32, torch.float16, torch.bfloat16]))
#@pytest.mark.parametrize('zero_one', [True, False])
@pytest.mark.parametrize('zero_one', [False])
@pytest.mark.parametrize('d', [32, 63, 97, 122, 257, 384, 510, 740, 1023])
@pytest.mark.parametrize("seqlen", [14, 31, 97, 128, 200, 384, 768, 1024] )
def test_l2_norm_race_condition(seqlen, d, zero_one, dtype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 60  # Sometimes we need large batch size for the race conditions to trigger
        
    
    print(f"{batch_size=}, {seqlen=}, {d=}, {zero_one=}, {dtype=} ")
    
    x = make_inputs(B=batch_size,
                      N=seqlen,
                      C=d,
                      do_shift=True, 
                      do_zero_one=zero_one, 
                      device=device, 
                      dtype=dtype
                     )
        
    torch.random.manual_seed(42)

    xo0 = triton_norm_func_fn(x)
    
    gx = torch.randn_like(xo0)


    dx0 = torch.autograd.grad((xo0,), (x,), (gx,))[0]
    
    # Numerical error if we just do any arithmetic on dq
    dx_atol = 2 * ((dx0 + 0.3 - 0.3) - dx0).abs().max().item()

    for i in range(250):
        torch.random.manual_seed(42)
        xo = triton_norm_func_fn(x)
        #out, lse, _ = flash_attn_func(q, k, v, dropout_p, causal=causal, return_attn_probs=True)
        assert torch.equal(xo, xo0)


        dx = torch.autograd.grad((xo,), (x,), (gx,))[0]
        
        dx_equal = torch.allclose(dx, dx0, atol=dx_atol)
        if not dx_equal:
            print(f"Iter {i}, {dx_atol = }, dQ max diff: {(dx - dx0).abs().max().item()}")
        assert dx_equal

        
