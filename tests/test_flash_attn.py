import math
import pytest
import torch
import torch.nn.functional as F
from einops import rearrange, repeat
from rgd_triton.triton.flash_attn.flash_attn_triton_cached import flash_attn_func

MAX_HEADDIM_SM75 = 64
MAX_HEADDIM_SM8x = 128

is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)



def attention_ref(
    q,
    k,
    v,
    attn_bias=None,
    upcast=True,
    reorder_ops=False,
):
    """
    Arguments:
        q: (batch_size, seqlen_q, nheads, head_dim)
        k: (batch_size, seqlen_k, nheads_k, head_dim)
        v: (batch_size, seqlen_k, nheads_k, head_dim)
        query_padding_mask: (batch_size, seqlen_q)
        key_padding_mask: (batch_size, seqlen_k)
        attn_bias: broadcastable to (batch_size, nheads, seqlen_q, seqlen_k)
        upcast: whether to cast all inputs to fp32, do all computation in fp32, then cast
            output back to fp16/bf16.
        reorder_ops: whether to change the order of operations (scaling k instead of scaling q, etc.)
            without changing the math. This is to estimate the numerical error from operation
            reordering.
    Output:
        output: (batch_size, seqlen_q, nheads, head_dim)
        attention: (batch_size, nheads, seqlen_q, seqlen_k), softmax after dropout
    """

    dtype_og = q.dtype
    if upcast:
        q, k, v = q.float(), k.float(), v.float()
    seqlen_q, seqlen_k = q.shape[1], k.shape[1]
    k = repeat(k, "b s h d -> b s (h g) d", g=q.shape[2] // k.shape[2])
    v = repeat(v, "b s h d -> b s (h g) d", g=q.shape[2] // v.shape[2])
    d = q.shape[-1]
    if not reorder_ops:
        scores = torch.einsum("bthd,bshd->bhts", q / math.sqrt(d), k)
    else:
        scores = torch.einsum("bthd,bshd->bhts", q, k / math.sqrt(d))

    if attn_bias is not None:
        scores = scores + attn_bias
    attention = torch.softmax(scores, dim=-1).to(v.dtype)
   
    output = torch.einsum("bhts,bshd->bthd", attention, v)

    return output.to(dtype=dtype_og), attention.to(dtype=dtype_og)





@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("bias_type", ["none", "static", "batch", "head", "cube"])
#@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
@pytest.mark.parametrize('d', [32, 64])
#@pytest.mark.parametrize("d", [64])
# @pytest.mark.parametrize('seqlen', [128, 256, 384, 512, 768, 1024, 2048])
@pytest.mark.parametrize("seqlen", [14, 31, 97, 128, 200, 384, 768] + ([] if is_sm75 else [1024, 1025, 2048]) )
# @pytest.mark.parametrize("seqlen", [512])
def test_flash_attn_qkv(seqlen, d, bias_type, dtype):
    if seqlen >= 2048 and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    qkv = torch.randn(
        3, batch_size, seqlen, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    q,k,v = qkv.unbind(0)
    
    if bias_type == "none":
        bias = None
    elif bias_type == "static":
        bias = torch.randn(1,1,seqlen,seqlen, device=device, dtype=dtype, requires_grad=False)
    elif bias_type == "batch":
        bias = torch.randn(batch_size,1,seqlen,seqlen, device=device, dtype=dtype, requires_grad=False)
    elif bias_type == "head":
        bias = torch.randn(1,nheads,seqlen,seqlen, device=device, dtype=dtype, requires_grad=False)
    else: #cube
        bias = torch.randn(batch_size,nheads,seqlen,seqlen, device=device, dtype=dtype, requires_grad=False)
        
    class KernelCache:
        def __init__(self):
            self.fwd = None
            self.bwd = None
            self.bwd_odot = None
    kc = KernelCache()

    out = flash_attn_func(q, k, v, bias, False, None, kernel_cache=kc)
    out = flash_attn_func(q, k, v, bias, False, None, kernel_cache=kc)

    out_ref,_ = attention_ref(
        q,
        k,
        v,
        attn_bias=bias,
        upcast=True,
        reorder_ops=False,
    )
        
    out_pt,_ = attention_ref(
        q,
        k,
        v,
        attn_bias=bias,
        upcast=False,
        reorder_ops=True,
    )

    # v = qkv[:, :, 2].float()
    # qk = torch.einsum('bshd,bthd->bhst', qkv[:, :, 0], qkv[:, :, 1]).float()
    # if causal:
    #     causal_mask = torch.triu(torch.ones(seqlen, seqlen, dtype=torch.bool, device=qkv.device), 1)
    #     qk.masked_fill_(causal_mask, float('-inf'))
    # m = qk.amax(-1, keepdim=True)
    # s_tmp = torch.exp((qk - m) / math.sqrt(d))
    # p_tmp = torch.softmax(qk / math.sqrt(d), -1)
    # p_dropped = p_tmp if dropout_mask is None else p_tmp.masked_fill(~dropout_mask, 0)
    # lse_ref = torch.logsumexp(qk / math.sqrt(d), -1)
    # qk_max1 = torch.max(qk[:, :, 128:, 192:], -1, keepdim=True).values
    # qk_max2 = torch.max(qk[:, :, 128:, 128:], -1, keepdim=True).values
    # qk_max3 = torch.max(qk[:, :, 128:, 64:], -1, keepdim=True).values
    # qk_max4 = torch.max(qk[:, :, 128:, :], -1, keepdim=True).values
    # o1 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, 192:] - qk_max1) / math.sqrt(d)), v[:, 192:])
    # o2 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, 128:] - qk_max2) / math.sqrt(d)), v[:, 128:])
    # o3 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, 64:] - qk_max3) / math.sqrt(d)), v[:, 64:])
    # o4 = torch.einsum('bhst,bthd->bshd', torch.exp((qk[:, :, 128:, :] - qk_max4) / math.sqrt(d)), v[:, :])
    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    # do_o = (g.float() * out.float()).sum(-1)
    # dv_tmp = torch.einsum('bhts,bthd->bshd', attn_pt[:, :, :64], g[:, :64])
    # dv_tmp1 = torch.einsum('bhts,bthd->bshd', attn_pt[:, :, 64:], g[:, 64:])
    if (d <= MAX_HEADDIM_SM8x) or (is_sm80 or is_sm90):
        (dqkv,) = torch.autograd.grad(out, qkv, g)
        out = flash_attn_func(q, k, v, bias, False, None, kernel_cache=kc)
        (dqkv,) = torch.autograd.grad(out, qkv, g)
        (dqkv_ref,) = torch.autograd.grad(out_ref, qkv, g)
        (dqkv_pt,) = torch.autograd.grad(out_pt, qkv, g)
        print(f"dQ max diff: {(dqkv[0] - dqkv_ref[0]).abs().max().item()}")
        print(f"dK max diff: {(dqkv[1] - dqkv_ref[1]).abs().max().item()}")
        print(f"dV max diff: {(dqkv[2] - dqkv_ref[2]).abs().max().item()}")
        print(f"dQKV mean diff: {(dqkv - dqkv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dqkv_pt[0] - dqkv_ref[0]).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dqkv_pt[1] - dqkv_ref[1]).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dqkv_pt[2] - dqkv_ref[2]).abs().max().item()}")
        print(f"dQKV Pytorch mean diff: {(dqkv_pt - dqkv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if (d <= MAX_HEADDIM_SM8x) or (is_sm80 or is_sm90):
        assert (dqkv - dqkv_ref).abs().max().item() <= 2 * (dqkv_pt - dqkv_ref).abs().max().item()

@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("bias_type", ["none", "static", "batch", "head", "cube"])
# @pytest.mark.parametrize("causal", [True])
#@pytest.mark.parametrize("d", [32, 40, 59, 64, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
#@pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize('d', [32, 64])
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
def test_flash_attn_output(
    seqlen_q, seqlen_k, d, bias_type, dtype
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

    q = torch.randn(
        batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    
    k = torch.randn(
        batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True
    )
    v = torch.randn(
        batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True
    )

    if bias_type == "none":
        bias = None
    elif bias_type == "static":
        bias = torch.randn(1,1,seqlen_q,seqlen_k, device=device, dtype=dtype, requires_grad=False)
    elif bias_type == "batch":
        bias = torch.randn(batch_size,1,seqlen_q,seqlen_k, device=device, dtype=dtype, requires_grad=False)
    elif bias_type == "head":
        bias = torch.randn(1,nheads,seqlen_q,seqlen_k, device=device, dtype=dtype, requires_grad=False)
    else: #cube
        bias = torch.randn(batch_size,nheads,seqlen_q,seqlen_k, device=device, dtype=dtype, requires_grad=False)
        
    out = flash_attn_func(q, k, v, bias, False, None)

    out_ref,_ = attention_ref(
        q,
        k,
        v,
        attn_bias=bias,
        upcast=True,
        reorder_ops=False,
    )
        
    out_pt,_ = attention_ref(
        q,
        k,
        v,
        attn_bias=bias,
        upcast=False,
        reorder_ops=True,
    )
    

    print(f"Output max diff: {(out - out_ref).abs().max().item()}")
    print(f"Output mean diff: {(out - out_ref).abs().mean().item()}")
    print(f"Pytorch max diff: {(out_pt - out_ref).abs().max().item()}")
    print(f"Pytorch mean diff: {(out_pt - out_ref).abs().mean().item()}")

    g = torch.randn_like(out)
    do_o = (g.float() * out.float()).sum(-1)
    if (d <= MAX_HEADDIM_SM8x) or (is_sm80 or is_sm90):
        (
            dq,
            dk,
            dv,
        ) = torch.autograd.grad(out, (q, k, v), g)
        (
            dq_ref,
            dk_ref,
            dv_ref,
        ) = torch.autograd.grad(out_ref, (q, k, v), g)
        (
            dq_pt,
            dk_pt,
            dv_pt,
        ) = torch.autograd.grad(out_pt, (q, k, v), g)
        print(f"dQ max diff: {(dq - dq_ref).abs().max().item()}")
        print(f"dK max diff: {(dk - dk_ref).abs().max().item()}")
        print(f"dV max diff: {(dv - dv_ref).abs().max().item()}")
        print(f"dQ mean diff: {(dq - dq_ref).abs().mean().item()}")
        print(f"dK mean diff: {(dk - dk_ref).abs().mean().item()}")
        print(f"dV mean diff: {(dv - dv_ref).abs().mean().item()}")
        print(f"dQ Pytorch max diff: {(dq_pt - dq_ref).abs().max().item()}")
        print(f"dK Pytorch max diff: {(dk_pt - dk_ref).abs().max().item()}")
        print(f"dV Pytorch max diff: {(dv_pt - dv_ref).abs().max().item()}")
        print(f"dQ Pytorch mean diff: {(dq_pt - dq_ref).abs().mean().item()}")
        print(f"dK Pytorch mean diff: {(dk_pt - dk_ref).abs().mean().item()}")
        print(f"dV Pytorch mean diff: {(dv_pt - dv_ref).abs().mean().item()}")

    # Check that FlashAttention's numerical error is at most twice the numerical error
    # of a Pytorch implementation.
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()

    if (d <= MAX_HEADDIM_SM8x) or (is_sm80 or is_sm90):
        assert (dq - dq_ref).abs().max().item() <= 3 * (dq_pt - dq_ref).abs().max().item()
        assert (dk - dk_ref).abs().max().item() <= 3 * (dk_pt - dk_ref).abs().max().item()
        assert (dv - dv_ref).abs().max().item() <= 3 * (dv_pt - dv_ref).abs().max().item()


# @pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("bias_type", ["none", "cube"])
# @pytest.mark.parametrize('causal', [True])
#@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 56, 64, 80, 96, 128])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192])
#@pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize('d', [32, 64])
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
# @pytest.mark.parametrize("dropout_p", [0.0])
def test_flash_attn_race_condition(seqlen_q, seqlen_k, d, bias_type, dtype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 60  # Sometimes we need large batch size for the race conditions to trigger
    nheads = 4
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    
    if bias_type == "none":
        bias = None
    else: #cube
        bias = torch.randn(batch_size,nheads,seqlen_q,seqlen_k, device=device, dtype=dtype, requires_grad=False)
        
    torch.random.manual_seed(42)

    out0, lse0 = flash_attn_func(q, k, v, bias, False, None, return_lse=True)
    
    g = torch.randn_like(out0)
    if (d <= MAX_HEADDIM_SM8x) or (is_sm80 or is_sm90):
        (
            dq0,
            dk0,
            dv0,
        ) = torch.autograd.grad(out0, (q, k, v), g)
        # Numerical error if we just do any arithmetic on dq
        dq_atol = 2 * ((dq0 + 0.3 - 0.3) - dq0).abs().max().item()

    for i in range(250):
        torch.random.manual_seed(42)
        out, lse = flash_attn_func(q, k, v, bias, False, None, return_lse=True)
        #out, lse, _ = flash_attn_func(q, k, v, dropout_p, causal=causal, return_attn_probs=True)
        assert torch.equal(out, out0)
        assert torch.equal(lse, lse0)

        if (d <= MAX_HEADDIM_SM8x or dropout_p == 0) or (is_sm80 or is_sm90):
            (
                dq,
                dk,
                dv,
            ) = torch.autograd.grad(out, (q, k, v), g)
            dq_equal = torch.allclose(dq, dq0, atol=dq_atol)
            if not dq_equal:
                print(f"Iter {i}, {dq_atol = }, dQ max diff: {(dq - dq0).abs().max().item()}")
            assert torch.equal(dv, dv0)
            assert torch.equal(dk, dk0)
            assert dq_equal


@pytest.mark.parametrize("dtype", [torch.float16])
@pytest.mark.parametrize("bias_type", ["none", "cube"])
# @pytest.mark.parametrize('causal', [False])
#@pytest.mark.parametrize("d", [16, 32, 64])
#@pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize('d', [32, 64])
@pytest.mark.parametrize("seqlen", [1, 2, 5, 17, 128])
# @pytest.mark.parametrize('seqlen', [2])
def test_flash_attn_bwd_overflow(seqlen, d, bias_type, dtype):
    """We previously had a bug where not masking elements beyond seqlen_k caused NaN in dQ,
    in the case where seqlen % 128 != 0.
    """
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 2
    nheads = 5
    q = torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda") * 5
    k, v = [
        torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda") * 3
        for _ in range(2)
    ]

    if bias_type == "none":
        bias = None
    else: #cube
        bias = torch.randn(batch_size,nheads,seqlen,seqlen, device=device, dtype=dtype, requires_grad=False)
        
    q.requires_grad_(True)
    k.requires_grad_(True)
    v.requires_grad_(True)

    out = flash_attn_func(q, k, v, bias, False, None, return_lse=False)
    g = torch.randn_like(out)
    out.backward(g)
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt,_ = attention_ref(q_pt, k_pt, v_pt, attn_bias=bias, upcast=False, reorder_ops=True)
    out_pt.backward(g)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, attn_bias=bias)
    out_ref.backward(g)
    print(f"dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(q_pt.grad - q_ref.grad).abs().max().item()}")
    print(f"dK Pytorch max diff: {(k_pt.grad - k_ref.grad).abs().max().item()}")
    print(f"dV Pytorch max diff: {(v_pt.grad - v_ref.grad).abs().max().item()}")
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    assert (q.grad - q_ref.grad).abs().max().item() <= 5 * (
        q_pt.grad - q_ref.grad
    ).abs().max().item() + 1e-3
    assert (k.grad - k_ref.grad).abs().max().item() <= 5 * (
        k_pt.grad - k_ref.grad
    ).abs().max().item() + 1e-3
    assert (v.grad - v_ref.grad).abs().max().item() <= 5 * (
        v_pt.grad - v_ref.grad
    ).abs().max().item() + 1e-3


@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize('dtype', [torch.bfloat16])
@pytest.mark.parametrize("bias_type", ["none", "cube"])
# @pytest.mark.parametrize('causal', [False])
#@pytest.mark.parametrize("d", [64, 128])
#@pytest.mark.parametrize('d', [64])
@pytest.mark.parametrize('d', [32, 64])
@pytest.mark.parametrize("seqlen", [97, 128, 200, 256])
# @pytest.mark.parametrize('seqlen', [128])
def test_flash_attn_bwd_transpose(seqlen, d, bias_type, dtype):
    """We previously had a bug where we were using the wrong strides of dout, which shows up
    when dout is not contiguous.
    """
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 5
    nheads = 2
    q, k, v = [
        torch.randn([batch_size, seqlen, nheads, d], dtype=dtype, device="cuda", requires_grad=True)
        for _ in range(3)
    ]

    if bias_type == "none":
        bias = None
    else: #cube
        bias = torch.randn(batch_size,nheads,seqlen,seqlen, device=device, dtype=dtype, requires_grad=False)

    out = rearrange(flash_attn_func(q, k, v, bias, False, None, return_lse=False), "b s ... -> s b ...")
    # So g is not contiguous
    g = torch.randn(seqlen, 2 * batch_size, nheads, d, dtype=dtype, device="cuda")[:, ::2]
    out.backward(g)
    q_pt = q.detach().clone().requires_grad_(True)
    k_pt = k.detach().clone().requires_grad_(True)
    v_pt = v.detach().clone().requires_grad_(True)
    out_pt, _ = attention_ref(q_pt, k_pt, v_pt, attn_bias=bias, upcast=False, reorder_ops=True)
    out_pt = rearrange(out_pt, "b s ... -> s b ...")
    out_pt.backward(g)
    q_ref = q.detach().clone().requires_grad_(True)
    k_ref = k.detach().clone().requires_grad_(True)
    v_ref = v.detach().clone().requires_grad_(True)
    out_ref, _ = attention_ref(q_ref, k_ref, v_ref, attn_bias=bias)
    out_ref = rearrange(out_ref, "b s ... -> s b ...")
    out_ref.backward(g)
    print(f"dQ max diff: {(q.grad - q_ref.grad).abs().max().item()}")
    print(f"dK max diff: {(k.grad - k_ref.grad).abs().max().item()}")
    print(f"dV max diff: {(v.grad - v_ref.grad).abs().max().item()}")
    print(f"dQ Pytorch max diff: {(q_pt.grad - q_ref.grad).abs().max().item()}")
    print(f"dK Pytorch max diff: {(k_pt.grad - k_ref.grad).abs().max().item()}")
    print(f"dV Pytorch max diff: {(v_pt.grad - v_ref.grad).abs().max().item()}")
    assert (out - out_ref).abs().max().item() <= 2 * (out_pt - out_ref).abs().max().item()
    assert (q.grad - q_ref.grad).abs().max().item() <= 2 * (
        q_pt.grad - q_ref.grad
    ).abs().max().item()
    assert (k.grad - k_ref.grad).abs().max().item() <= 2 * (
        k_pt.grad - k_ref.grad
    ).abs().max().item()
    assert (v.grad - v_ref.grad).abs().max().item() <= 2 * (
        v_pt.grad - v_ref.grad
    ).abs().max().item()


def compare_tensors(a, b, atol=7e-05, rtol=1e-5, name="tensor"):
    if a.shape != b.shape:
        raise ValueError(f"Shape mismatch for {name}: {a.shape} vs {b.shape}")
    
    diff = (a - b).abs()
    rel_diff = diff / (b.abs() + 1e-8)
    mask = ~(torch.isclose(a, b, atol=atol, rtol=rtol))

    num_mismatched = mask.sum().item()
    total = mask.numel()
    max_abs = diff.max().item()
    max_rel = rel_diff.max().item()

    # Optional: show where
    if num_mismatched > 0:
        
        print(f"\n{name} mismatch:")
        print(f"  Mismatched: {num_mismatched} / {total} ({100. * num_mismatched / total:.2f}%)")
        print(f"  Max absolute diff: {max_abs}")
        print(f"  Max relative diff: {max_rel}")


        idx = torch.nonzero(mask)
        i = idx[0].tolist()
        print(f"  First mismatch at index {i}: {a[tuple(i)].item()} vs {b[tuple(i)].item()}")

    return num_mismatched == 0

#@pytest.mark.focus
@pytest.mark.skip(reason="Determinism not implemented")
@pytest.mark.parametrize("dtype", ([torch.float16] if is_sm75 else [torch.float16, torch.bfloat16]))
# @pytest.mark.parametrize("dtype", [torch.bfloat16])
@pytest.mark.parametrize("bias_type", ["none", "cube"])
# @pytest.mark.parametrize("causal", [True])
#@pytest.mark.parametrize("d", [32, 40, 59, 64, 80, 96, 111, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize("d", [32, 64, 96, 128, 160, 192, 224, 256])
# @pytest.mark.parametrize('d', [32, 40, 64, 80, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [32, 64, 96, 128, 160, 192])
# @pytest.mark.parametrize('d', [56, 80])
#@pytest.mark.parametrize("d", [64])
@pytest.mark.parametrize('d', [32, 64])
@pytest.mark.parametrize("swap_sq_sk", [False, True])
# @pytest.mark.parametrize("swap_sq_sk", [False])
@pytest.mark.parametrize(
    "seqlen_q,seqlen_k",
    [
        (1, 239),
        (3, 799),
        (127, 512),
        (127, 513),
        (113, 203),
        (128, 217),
        (113, 211),
        (108, 256),
        (256, 512),
        (1023, 1024),
    ],
)
# @pytest.mark.parametrize('seqlen_q,seqlen_k', [(256, 128)])
def test_flash_attn_deterministic(seqlen_q, seqlen_k, swap_sq_sk, d, bias_type, dtype):
    if (
        max(seqlen_q, seqlen_k) >= 2048
        and torch.cuda.get_device_properties("cuda").total_memory <= 8 * 2**30
    ):
        pytest.skip()  # Reference implementation OOM
    if swap_sq_sk:
        seqlen_q, seqlen_k = seqlen_k, seqlen_q
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9
    q = torch.randn(batch_size, seqlen_q, nheads, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(batch_size, seqlen_k, nheads, d, device=device, dtype=dtype, requires_grad=True)

    if bias_type == "none":
        bias = None
    else: #cube
        bias = torch.randn(batch_size,nheads,seqlen_q,seqlen_k, device=device, dtype=dtype, requires_grad=False)

    out = flash_attn_func(q, k, v, bias, False, None, return_lse=False)
    #out = flash_attn_func(q, k, v, 0.0, causal=causal, window_size=window_size, deterministic=True)

    g = torch.randn_like(out)
    dq0, dk0, dv0 = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
    for _ in range(50):
        dq, dk, dv = torch.autograd.grad(out, (q, k, v), g, retain_graph=True)
        assert compare_tensors(dv, dv0, name='dv')
        assert compare_tensors(dk, dk0, name='dk')
        assert compare_tensors(dq, dq0, name='dq')

        
