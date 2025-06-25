import torch

import triton
import triton.language as tl


@triton.jit
def _rope_transform_block(
        row_block, 
        sin_block,
        cos_block,
        seq_offsets,
        chan_offsets,
        MASK_LEN, 
        ROT_DIM,
        BLOCK_N : tl.constexpr,
        BLOCK_K : tl.constexpr,
        IS_BWD: tl.constexpr,
        ):
    # row_block: shape (BLOCK_N, BLOCK_K)
    # sin_cos_block: shape (2, BLOCK_M, R)
    # global_indices: shape (BLOCK_N,) of global sequence indices.
    # Create a mask for which rows should have RoPE applied.
    seq_mask = seq_offsets < MASK_LEN
    chan_mask = chan_offsets < ROT_DIM

    # Reshape row_block to shape (BLOCK_N, BLOCK_K//2, 2) without moving data,
    # effectively treating pairs of channels as a complex number.
    x_reshaped = tl.reshape(row_block, (BLOCK_N, BLOCK_K // 2, 2))

    # Perform the rotation:
    # rotated[..., 0] = -x_reshaped[..., 1]
    # rotated[..., 1] = x_reshaped[..., 0]
    x_even, x_odd = tl.split(x_reshaped)
    if not IS_BWD:
        rotated_even = -x_odd
        rotated_odd  = x_even
    else: # invert the transform
        rotated_even = x_odd
        rotated_odd  = -x_even
    rotated = tl.join(rotated_even, rotated_odd)  # shape (BLOCK_N, BLOCK_K//2, 2)

    # Flatten rotated back to shape (BLOCK_N, BLOCK_K)
    rotated_flat = tl.reshape(rotated, (BLOCK_N, BLOCK_K))

    # Apply the rotary position embedding:
    # out = x * cos + rotated_flat * sin
    out_block = row_block * cos_block + rotated_flat * sin_block

    # For rows where RoPE isn’t applied, keep original values.
    new_row_block = tl.where(seq_mask[:, None] & chan_mask[None,:], out_block, row_block)
    return new_row_block



@triton.jit
def _combined_rope_block_kernel(
        # -- inputs --
        q_ptr, 
        k_ptr, 
        sin_cos_ptr,
        # -- outputs --
        qo_ptr,
        ko_ptr,
        # -- parameters --
        q_seq_len,
        k_seq_len,
        p_seq_len,
        q_mask_len,
        k_mask_len,
        split_idx, # where q ends and k begins
        head_dim,
        rope_dim,
        # -- strides --
        stride_qb,
        stride_qh,
        stride_qn,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_oqb,
        stride_oqh,
        stride_oqn,
        stride_okb,
        stride_okh,
        stride_okn,
        stride_sincos_s,
        stride_sincos_n,
        # constants
        BLOCK_N: tl.constexpr, # how big each block in N is
        BLOCK_K: tl.constexpr, # how big each block in C is
        IS_BWD: tl.constexpr,
    ):
        
    # Launch grid: (total_blocks, B, H), where:
    #   total_blocks = cdiv(N1, BLOCK_M) + cdiv(N2, BLOCK_M)
    block_idx = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)

    # create the offset array
    seq_offsets = tl.arange(0, BLOCK_N)
    chan_offsets = tl.arange(0, BLOCK_K)

    # Determine which tensor this block processes.
    if block_idx < split_idx:
        # Processing q.
        total_seq = q_seq_len
        mask_len = q_mask_len
        seq_offsets = seq_offsets + block_idx * BLOCK_N
        rd_ptr = q_ptr + b*stride_qb + h*stride_qh + seq_offsets*stride_qn
        wr_ptr = qo_ptr + b*stride_oqb + h*stride_oqh + seq_offsets*stride_oqn
        
    else:
        # Processing k.
        local_block = block_idx - split_idx
        total_seq = k_seq_len
        mask_len = k_mask_len
        seq_offsets = seq_offsets + local_block * BLOCK_N
        rd_ptr = k_ptr + b*stride_kb + h*stride_kh + seq_offsets*stride_kn
        wr_ptr = ko_ptr + b*stride_okb + h*stride_okh + seq_offsets*stride_okn


    # load the sin/cos values
    sin_cos_ptr = sin_cos_ptr + seq_offsets*stride_sincos_n
    
    sin_block = tl.load(
              sin_cos_ptr[:,None] + chan_offsets[None,:],
              mask=(seq_offsets[:,None] < p_seq_len) & (chan_offsets[None,:] < rope_dim), 
              other=0.0
                )
    cos_block = tl.load(
              sin_cos_ptr[:,None] + chan_offsets[None,:] + stride_sincos_s,
              mask=(seq_offsets[:,None] < p_seq_len) & (chan_offsets[None,:] < rope_dim), 
              other=0.0
                )
    
    # Compute a mask to ensure we don't go beyond the total sequence length.
    valid_mask = (seq_offsets[:,None] < total_seq) & (chan_offsets[None,:] < head_dim)

    # Load the (BLOCK_M, BLOCK_K) tile from q or k using the valid mask.
    row_block = tl.load(rd_ptr[:,None]+chan_offsets[None,:], mask=valid_mask, other=0.0)
  
    # Apply the RoPE transformation to the block.
    new_row_block = _rope_transform_block(row_block, 
                                         sin_block, 
                                         cos_block,
                                         seq_offsets,
                                         chan_offsets,
                                         mask_len, 
                                         rope_dim,
                                         BLOCK_N,
                                         BLOCK_K,
                                         IS_BWD,
                                        )

    # Store the transformed block back using the valid mask.
    tl.store(wr_ptr[:,None]+chan_offsets[None,:], new_row_block, mask=valid_mask)

def combined_rope_fn(q, k, pos_emb, qo=None, ko=None, seq_cutoff=None, heads_second=True, is_bwd=False, kernel=None):
   
    assert q.ndim == k.ndim == 4
    assert pos_emb.ndim == 3
    assert q.dtype == k.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32], "Only support fp16, bf16, and fp32"
    assert q.is_cuda and k.is_cuda and pos_emb.is_cuda
    assert q.stride(-1) == k.stride(-1) == pos_emb.stride(-1) == 1

    if heads_second:
        bq,hq,q_seq_len,qdim = q.shape
        bk,hk,k_seq_len,kdim = k.shape
        # strides should be in B,H,N
        qstrides = (q.stride(0), q.stride(1), q.stride(2))
        kstrides = (k.stride(0), k.stride(1), k.stride(2))
        qostrides = (qo.stride(0), qo.stride(1), qo.stride(2))
        kostrides = (ko.stride(0), ko.stride(1), ko.stride(2))
    else:
        bq,q_seq_len,hq,qdim = q.shape
        bk,k_seq_len,hk,kdim = k.shape   
        # strides should be in B,H,N
        qstrides = (q.stride(0), q.stride(2), q.stride(1))
        kstrides = (k.stride(0), k.stride(2), k.stride(1))
        qostrides = (qo.stride(0), qo.stride(2), qo.stride(1))
        kostrides = (ko.stride(0), ko.stride(2), ko.stride(1))

    _,p_seq_len,rdim = pos_emb.shape

    assert q_seq_len < p_seq_len
    assert k_seq_len < p_seq_len
    assert (bq, hq, qdim) == (bk, hk, kdim)
    assert qdim <= 128, "Only supports head dimensions up to 128"

    if seq_cutoff is None:
        q_mask_len = q_seq_len
        k_mask_len = k_seq_len
    elif isinstance(seq_cutoff, (tuple, list)):
        q_mask_len = seq_cutoff[0]
        k_mask_len = seq_cutoff[1]
    else:
        q_mask_len = seq_cutoff
        k_mask_len = seq_cutoff


    BLOCK_K = (
        32
        if qdim <= 32
        else (64 if qdim <= 64 else (128 if qdim <= 128 else 256))
    )
    BLOCK_N = 4
    blocks_q = triton.cdiv(q_seq_len, BLOCK_N)
    blocks_k = triton.cdiv(k_seq_len, BLOCK_N)
    grid = lambda META: (blocks_q+blocks_k, bq, hq)  # noqa
    

    if kernel is not None:
        grid_tuple = grid({})
        kernel[grid_tuple](
            # -- inputs --
            q, 
            k, 
            pos_emb,
            # -- outputs --
            qo,
            ko,
            # -- parameters --
            q_seq_len,
            k_seq_len,
            p_seq_len,
            q_mask_len,
            k_mask_len,
            blocks_q, # where q ends and k begins
            qdim,
            rdim,
            # -- strides --
            *qstrides,
            *kstrides,
            *qostrides,
            *kostrides,
            pos_emb.stride(0),
            pos_emb.stride(1),
        )
        khandle = None
    else:
        khandle = _combined_rope_block_kernel[grid](
            # -- inputs --
            q, 
            k, 
            pos_emb,
            # -- outputs --
            qo,
            ko,
            # -- parameters --
            q_seq_len,
            k_seq_len,
            p_seq_len,
            q_mask_len,
            k_mask_len,
            blocks_q, # where q ends and k begins
            qdim,
            rdim,
            # -- strides --
            *qstrides,
            *kstrides,
            *qostrides,
            *kostrides,
            pos_emb.stride(0),
            pos_emb.stride(1),
            # constants
            BLOCK_N, # how big each block in N is
            BLOCK_K, # how big each block in C is
            is_bwd,
        )
        
    return qo, ko, khandle


class CombinedRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, pos_emb, seq_cutoff=None, heads_second=True, in_place=False, kernel_cache=None):
        """
        """
        # Make sure that the last dimension is contiguous
        q, k, pos_emb = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k, pos_emb]]
        if in_place:
            qo_i, qk_i = q, k
        else:
            qo_i, qk_i = torch.empty_like(q), torch.empty_like(k)
        qo, qk, kfwd = combined_rope_fn(q, k, pos_emb, qo=qo_i, ko=qk_i, seq_cutoff=seq_cutoff, heads_second=heads_second, is_bwd=False,
                                 kernel=(kernel_cache.fwd if kernel_cache is not None else None))
        if (kernel_cache is not None) and (kfwd is not None):
            kernel_cache.fwd = kfwd
            
        # Save the cache reference for the backward pass
        ctx.kernel_cache = kernel_cache
        # save the state for backward pass
        ctx.save_for_backward(pos_emb)
        ctx.seq_cutoff = seq_cutoff
        ctx.heads_second = heads_second
        return qo, qk

    @staticmethod
    def backward(ctx, doq, dok):
        pos_emb, = ctx.saved_tensors
        assert not ctx.needs_input_grad[2], "CombinedRoPE does not support gradients in position embeddings"
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(doq)
            dk = torch.empty_like(dok)
            _, _, kbwd = combined_rope_fn(
                doq, 
                dok, 
                pos_emb, 
                qo=dq, 
                ko=dk, 
                seq_cutoff=ctx.seq_cutoff, 
                heads_second=ctx.heads_second, 
                is_bwd=True,
                kernel=(ctx.kernel_cache.bwd if ctx.kernel_cache is not None else None),
            )
        if (ctx.kernel_cache is not None) and (kbwd is not None):
            kernel_cache.bwd = kbwd
        return dq, dk, None, None, None, None, None


def triton_rope_fn(query,key,pos_emb, seq_cutoff=None, heads_second=True, fused_ln=False, kernel_cache=None):
    return CombinedRoPEFunc.apply(query,key,pos_emb, seq_cutoff, heads_second, kernel_cache)
    
