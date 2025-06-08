import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from rgd_triton.triton.rope.head_rope_cached import triton_rope_fn 

def rotate_every_two(x):
    # apply in the shape of BHND instead of (BH)ND
    x1 = x[ :, :, :, ::2]
    x2 = x[ :, :, :, 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)  # in einsum notation: rearrange(x, '... d j -> ... (d j)')

def apply_rotary_pos_emb(x, sincos):
    return (x * sincos[1]) + (rotate_every_two(x) * sincos[0])


class Ref_Rotary(nn.Module):
    def __init__(self, seq_cutoff=None, fuse_ln=False, heads_second=True):
        self.seq_cutoff = seq_cutoff
        self.fuse_ln = fuse_ln
        self.heads_second = heads_second # (B,H,N,C) vs (B,N,H,C)

    def _apply_rotatary(self, q, k, pos_emb):

        # QK are in the shape (B, h, HW, C)
        # pos_emb is in the shape (s, H, W, R)
        s,H,W,R = pos_emb.shape

        # reshape pos_emb to match the sequence shape
        if self.heads_second:
            B,heads,HW,C = q.shape
            pos_emb = pos_emb.reshape(s,1,H*W,R)
            qlen = q.shape[2]
            klen = k.shape[2]
            pos_emb_q = pos_emb[:,:,:qlen]
            pos_emb_k = pos_emb[:,:,:klen]
            
        else:
            B,HW,heads,C = q.shape
            pos_emb = pos_emb.reshape(s,H*W,1,R)
            qlen = q.shape[1]
            klen = k.shape[1]
            pos_emb_q = pos_emb[:,:qlen]
            pos_emb_k = pos_emb[:,:klen]
        

        #print("==== applying roatery embedding ====")
        if R != C:
            # q,k,v are in the shape BHND
            q_rot = q[ :, :, :, : R]
            q_pas = q[ :, :, :, R :]

            k_rot = k[ :, :, :, : R]
            k_pas = k[ :, :, :, R :]

            q_rot = apply_rotary_pos_emb(q_rot, pos_emb_q)
            k_rot = apply_rotary_pos_emb(k_rot, pos_emb_k)

            q = torch.cat([q_rot, q_pas], dim=-1)
            k = torch.cat([k_rot, k_pas], dim=-1)
        else:
            q = apply_rotary_pos_emb(q, pos_emb_q)
            k = apply_rotary_pos_emb(k, pos_emb_k)

        # final outputs are of shape (B, h, HW, C)
        return q, k


    def forward(self, q, k, pos_emb, upcast=False):
        orig_dtype= q.dtype
        if upcast:
            q = q.float()
            k = k.float()
            
        if self.fuse_ln:
            q = F.layer_norm(q, eps=1e-6)
            k = F.layer_norm(k, eps=1e-6)

        # pos_emb is in the shape 2,h,w,c
        # split q and k based on the hedges - split along N
        if self.seq_cutoff is None:
            qi = q
            ki = k
            seq_cutoff_q = None
            seq_cutoff_k = None
        elif self.heads_second: #B,H,N,C
            seq_cutoff_q = self.seq_cutoff[0]
            seq_cutoff_k = self.seq_cutoff[1]
            catdim = 2
            qlen = q.shape[2]
            klen = k.shape[2]
            qi = q[:,:,:seq_cutoff_q,:]
            qv = q[:,:,seq_cutoff_q:,:]
            ki = k[:,:,:seq_cutoff_k,:]
            kv = k[:,:,seq_cutoff_k:,:]
        else: # B,N,H,C
            seq_cutoff_q = self.seq_cutoff[0]
            seq_cutoff_k = self.seq_cutoff[1]
            catdim = 1
            qlen = q.shape[1]
            klen = k.shape[1]
            qi = q[:,:seq_cutoff_q,:,:]
            qv = q[:,seq_cutoff_q:,:,:]
            ki = k[:,:seq_cutoff_k,:,:]
            kv = k[:,seq_cutoff_k:,:,:]        
        
        qi, ki = self._apply_rotatary(qi, ki, pos_emb=pos_emb)

        # combine the q and k again
        if (self.seq_cutoff is not None) and (seq_cutoff_q < qlen):
            q = torch.cat([qi,qv], dim=catdim)
        else:
            q = qi

        if (self.seq_cutoff is not None) and (seq_cutoff_k < klen):
            k = torch.cat([ki,kv], dim=catdim)
        else:
            k = ki

        return q.to(orig_dtype),k.to(orig_dtype)


           
class AxialRotaryEmbedding(nn.Module):
    def __init__(self, channels, max_freq = None, offset=0):
        super(AxialRotaryEmbedding, self).__init__()
        max_freq = 64 if max_freq is None else max_freq 
        
        self.dim = channels
        self.offset = offset
        #scales = torch.linspace(1., max_freq / 2, self.dim // 4)
        inv_freq = 1.0 / (max_freq ** (torch.arange(0, channels, 2).float() / channels))
        self.register_buffer("inv_freq", inv_freq)
        #self.register_buffer('scales', scales)

        
    def get_sincos(self, sin_inp):
        """
        Gets a base embedding for one dimension with sin and cos intertwined
        """
        
        return torch.stack([sin_inp.sin(), sin_inp.cos()])
     

    def forward(self, tensor):
        device, dtype, n = tensor.device, tensor.dtype, int(math.sqrt(tensor.shape[-2]))

        batch_size, y, x, orig_ch = tensor.shape
        pos_x = torch.arange(x, device=tensor.device).type(self.inv_freq.type())
        pos_y = torch.arange(y, device=tensor.device).type(self.inv_freq.type())
        sin_inp_x = torch.einsum("i,j->ij", pos_x, self.inv_freq) - self.offset
        sin_inp_y = torch.einsum("i,j->ij", pos_y, self.inv_freq) - self.offset
        
        #print(sin_inp_x.shape)
        
        emb_x = self.get_sincos(sin_inp_x)
        emb_y = self.get_sincos(sin_inp_y)
        
        #print("emb_x, emb_y", emb_x.shape, emb_y.shape)

        x_sinu = repeat(emb_x, 's i d -> s j i d', j = y)
        y_sinu = repeat(emb_y, 's j d -> s j i d', i = x)
        
        #print("x_sinu, ysinu",x_sinu.shape, y_sinu.shape)

        sin = torch.cat((x_sinu[0], y_sinu[0]), dim = -1)
        cos = torch.cat((x_sinu[1], y_sinu[1]), dim = -1)
        
        #print("sin,cos", sin.shape, cos.shape)

        sin, cos = map(lambda t: rearrange(t, 'i j d -> (i j) d'), (sin, cos))
        #print("sin,cos", sin.shape, cos.shape)
        sin, cos = map(lambda t: repeat(t, 'n d -> () n (d j)', j = 2), (sin, cos))
        #print("sin,cos", sin.shape, cos.shape)
        return torch.cat([sin, cos], dim=0)


def ref_rope_fn(query,key,pos_emb, seq_cutoff=None, heads_second=True, fused_ln=False):
    rr = Ref_Rotary(seq_cutoff=seq_cutoff, fuse_ln=fused_ln, heads_second=heads_second)
    rq, rk = rr.forward(query,key,pos_emb)
    return rq, rk

def make_inputs(B,H,NQ,NK,C,RDIM,EMB_LEN,ROT_FREQ=32, do_shift=False, heads_second=True, device="cuda", dtype=torch.float):
    penc = AxialRotaryEmbedding(RDIM//2, max_freq=ROT_FREQ)
    pos_emb = penc(torch.zeros(1,EMB_LEN,EMB_LEN,2)).to(device, dtype).detach()  
    pos_emb = rearrange(pos_emb, 'q (h w) b -> q h w b', h=EMB_LEN)

    if heads_second:
        q = torch.randn(B,H,NQ,C, device=device, dtype=dtype)
        k = torch.randn(B,H,NK,C, device=device, dtype=dtype)
    else:
        q = torch.randn(B,NQ,H,C, device=device, dtype=dtype)
        k = torch.randn(B,NK,H,C, device=device, dtype=dtype)      

    # enable gradients
    q = q.clone().detach().requires_grad_(True)
    k = k.clone().detach().requires_grad_(True)
    
    return q,k,pos_emb


is_sm75 = torch.cuda.get_device_capability("cuda") == (7, 5)
is_sm8x = torch.cuda.get_device_capability("cuda")[0] == 8
is_sm80 = torch.cuda.get_device_capability("cuda") == (8, 0)
is_sm90 = torch.cuda.get_device_capability("cuda") == (9, 0)

def get_tolerance(dtype):
    if dtype == torch.float32:
        return 1.3e-6, 1e-5 # from torch doc
        #return 1e-3, 1e-5
    elif dtype == torch.float16:
        #return 1e-3, 1e-5 # from torch doc
        return 1e-3, 1e-3 # accounting for multiplication
        #return 1e-2, 1e-3
    elif dtype == torch.bfloat16:
        return 1.6e-2, 1e-5 # from torch doc
        #return 1e-2, 1e-3
    else:
        raise ValueError(f"Unhandled dtype: {dtype}")
        


@pytest.mark.parametrize("dtype", ([torch.float32, torch.float16] if is_sm75 else [torch.float32, torch.float16, torch.bfloat16]))
@pytest.mark.parametrize(
    "d,rdim",
    [
        (32, 32),
        (64, 32),
        (64, 64),
        (128, 32),
        (128, 64),
        (128, 96),
        (128, 128),
    ],
)
@pytest.mark.parametrize('seq_cutoff', [0,0.5,0.75,1.0])
@pytest.mark.parametrize('heads_second', [True, False])
@pytest.mark.parametrize("seqlen", [14, 31, 97, 128, 200, 384, 768, 1024] )
def test_rope_head(seqlen, d, rdim, seq_cutoff, heads_second, dtype):
    if seqlen >= 2048 and torch.cuda.get_device_properties("cuda").total_memory <= 16 * 2**30:
        pytest.skip()  # Reference implementation OOM
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 4
    nheads = 9

    
    q,k,pos_emb = make_inputs(
                      B=batch_size,
                      H=nheads,
                      NQ=seqlen,
                      NK=seqlen,
                      C=d, 
                      RDIM=rdim,
                      EMB_LEN=seqlen,
                      ROT_FREQ=32,
                      do_shift=False, 
                      heads_second=heads_second, 
                      device=device, 
                      dtype=dtype
                     )

    seq_cutoff = int(seq_cutoff*seqlen)
    print(f"{q.shape=}, {k.shape=}, {pos_emb.shape=}, seq_cutoff={(seq_cutoff,seq_cutoff)}, {rdim=}, {heads_second=}, {dtype=}")
    qo, ko = triton_rope_fn(q, k, pos_emb.view(2,-1,pos_emb.shape[-1]), seq_cutoff=(seq_cutoff,seq_cutoff), heads_second=heads_second, fused_ln=False)
    
    qo_ref, ko_ref = ref_rope_fn(
        q,
        k,
        pos_emb,
        seq_cutoff=(seq_cutoff,seq_cutoff),
        heads_second=heads_second,
        fused_ln=False
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
    
    torch.testing.assert_close(qo, qo_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(ko, ko_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(dq, dq_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(dk, dk_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", ([torch.float32, torch.float16] if is_sm75 else [torch.float32, torch.float16, torch.bfloat16]))
@pytest.mark.parametrize('seq_cutoff', [0,0.5,0.75,1.0])
@pytest.mark.parametrize('heads_second', [True, False])
@pytest.mark.parametrize(
    "d,rdim",
    [
        (32, 32),
        (64, 32),
        (64, 64),
        (128, 32),
        (128, 64),
        (128, 96),
        (128, 128),
    ],
)
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
def test_rope_head_seqdiff(
    seqlen_q, seqlen_k, d, rdim, seq_cutoff, heads_second, dtype
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

    emb_len = max(seqlen_q, seqlen_k)
    q,k,pos_emb = make_inputs(
                      B=batch_size,
                      H=nheads,
                      NQ=seqlen_q,
                      NK=seqlen_k,
                      C=d, 
                      RDIM=rdim,
                      EMB_LEN=emb_len,
                      ROT_FREQ=32,
                      do_shift=False, 
                      heads_second=heads_second, 
                      device=device, 
                      dtype=dtype
                     )

    seq_cutoff_q = int(seqlen_q*seq_cutoff)
    seq_cutoff_k = int(seqlen_k*seq_cutoff)
    print(f"{q.shape=}, {k.shape=}, {pos_emb.shape=}, {emb_len=}, seq_cutoff={(seq_cutoff_q,seq_cutoff_k)}, {rdim=}, {heads_second=}, {dtype=}")
    qo, ko = triton_rope_fn(q, k, pos_emb.view(2,-1,pos_emb.shape[-1]), seq_cutoff=(seq_cutoff_q,seq_cutoff_k), heads_second=heads_second, fused_ln=False)
    
    qo_ref, ko_ref = ref_rope_fn(
        q,
        k,
        pos_emb,
        seq_cutoff=(seq_cutoff_q,seq_cutoff_k),
        heads_second=heads_second,
        fused_ln=False
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
    
    torch.testing.assert_close(qo, qo_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(ko, ko_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(dq, dq_ref, rtol=rtol, atol=atol)
    torch.testing.assert_close(dk, dk_ref, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", ([torch.float32, torch.float16] if is_sm75 else [torch.float32, torch.float16, torch.bfloat16]))
@pytest.mark.parametrize('heads_second', [True, False])
@pytest.mark.parametrize('seq_cutoff', [0,0.5,0.75,1.0])
@pytest.mark.parametrize(
    "d,rdim",
    [
        (32, 32),
        (64, 32),
        (64, 64),
        (128, 32),
        (128, 64),
        (128, 96),
        (128, 128),
    ],
)
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
def test_rope_head_race_condition(seqlen_q, seqlen_k, d, rdim, seq_cutoff, heads_second, dtype):
    device = "cuda"
    # set seed
    torch.random.manual_seed(0)
    batch_size = 60  # Sometimes we need large batch size for the race conditions to trigger
    nheads = 4

    emb_len = max(seqlen_q, seqlen_k)
    q,k,pos_emb = make_inputs(
                      B=batch_size,
                      H=nheads,
                      NQ=seqlen_q,
                      NK=seqlen_k,
                      C=d, 
                      RDIM=rdim,
                      EMB_LEN=emb_len,
                      ROT_FREQ=32,
                      do_shift=False, 
                      heads_second=heads_second, 
                      device=device, 
                      dtype=dtype
                     )

    seq_cutoff_q = int(seqlen_q*seq_cutoff)
    seq_cutoff_k = int(seqlen_k*seq_cutoff)
    print(f"{q.shape=}, {k.shape=}, {pos_emb.shape=}, {emb_len=}, seq_cutoff={(seq_cutoff_q,seq_cutoff_k)}, {rdim=}, {heads_second=}, {dtype=}")
    
    
    torch.random.manual_seed(42)

    qo0, ko0 = triton_rope_fn(q, k, pos_emb.view(2,-1,pos_emb.shape[-1]), seq_cutoff=(seq_cutoff_q,seq_cutoff_k), heads_second=heads_second, fused_ln=False)
    
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
        qo, ko = triton_rope_fn(q, k, pos_emb.view(2,-1,pos_emb.shape[-1]), seq_cutoff=(seq_cutoff_q,seq_cutoff_k), heads_second=heads_second, fused_ln=False)
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

        
