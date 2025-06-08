import torch
import triton
import triton.language as tl


@triton.jit
def _fused_ln_qk_kernel_fwd(
        # -- inputs --
        q_ptr, 
        k_ptr, 
        # -- outputs --
        qo_ptr,
        ko_ptr,
        # -- norm buffers --
        norm_ptr,  # pointer to the mean
        # -- parameters --
        q_seq_len,
        k_seq_len,
        split_idx, # where q ends and k begins
        head_dim,
        norm_eps, # epsilon to avoid division by zero
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
        stride_onb,
        stride_onh,
        stride_onn,
        # constants
        BLOCK_N: tl.constexpr, # how big each block in N is
        BLOCK_K: tl.constexpr, # how big each block in C is
    ):
        
    # Launch grid: (total_blocks, B, H), where:
    #   total_blocks = cdiv(N1, BLOCK_M) + cdiv(N2, BLOCK_M)
    block_idx = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)

    # create the offset array
    seq_offsets = tl.arange(0, BLOCK_N)
    chan_offsets = tl.arange(0, BLOCK_K)

    # compute the mean and variance pointers based on the block
    seq_shift = block_idx * BLOCK_N + seq_offsets
    norm_ptr = norm_ptr + b*stride_onb + h*stride_onh + seq_shift*stride_onn
    
    # Determine which tensor this block processes.
    if block_idx < split_idx:
        # Processing q.
        total_seq = q_seq_len
        seq_offsets = seq_shift
        rd_ptr = q_ptr + b*stride_qb + h*stride_qh + seq_offsets*stride_qn
        wr_ptr = qo_ptr + b*stride_oqb + h*stride_oqh + seq_offsets*stride_oqn
        
    else:
        # Processing k.
        local_block = block_idx - split_idx
        total_seq = k_seq_len
        seq_offsets = seq_offsets + local_block * BLOCK_N
        rd_ptr = k_ptr + b*stride_kb + h*stride_kh + seq_offsets*stride_kn
        wr_ptr = ko_ptr + b*stride_okb + h*stride_okh + seq_offsets*stride_okn

    tl.debug_barrier()
    
    # Compute a mask to ensure we don't go beyond the total sequence length.
    valid_mask = (seq_offsets[:,None] < total_seq) & (chan_offsets[None,:] < head_dim)

    # Load the (BLOCK_M, BLOCK_K) tile from q or k using the valid mask. - make sure to promote to float32
    row_block = tl.load(rd_ptr[:,None]+chan_offsets[None,:], mask=valid_mask, other=0.0).to(tl.float32)

    # --- compute the norm ---
    # note that we don't have to do anything special here, because we assume the head dim will fit all at once
    
    # first compute the mean - want to apply over the channels which is axis=1
    mean = tl.sum(row_block, axis=1) / head_dim
    # then compute the variance
    row_shifted = tl.where(valid_mask, row_block - mean[:,None], 0.0)
    var = tl.sum(row_shifted*row_shifted, axis=1) / head_dim
    rstd = tl.rsqrt(var + norm_eps)

    # compute the norm
    new_row_block = row_shifted * rstd[:,None]
    
    # store the var and rstd
    norm_val = tl.join(mean, rstd)
    tl.store(norm_ptr[:,None] + tl.arange(0,2)[None,:], norm_val, mask=seq_offsets[:,None] < total_seq)
    
    # Store the transformed block back using the valid mask.
    tl.store(wr_ptr[:,None]+chan_offsets[None,:], new_row_block, mask=valid_mask)

@triton.jit
def _fused_ln_qk_kernel_bwd(
        # -- inputs --
        q_ptr,
        k_ptr,
        doq_ptr, 
        dok_ptr,         
        # -- norm buffers --
        norm_ptr,  # pointer to the mean
        # -- outputs --
        dq_ptr,
        dk_ptr,
        # -- parameters --
        q_seq_len,
        k_seq_len,
        split_idx, # where q ends and k begins
        head_dim,
        norm_eps, # epsilon to avoid division by zero
        # -- strides --
        stride_qb,
        stride_qh,
        stride_qn,
        stride_kb,
        stride_kh,
        stride_kn,
        stride_doqb,
        stride_doqh,
        stride_doqn,
        stride_dokb,
        stride_dokh,
        stride_dokn,
        stride_dqb,
        stride_dqh,
        stride_dqn,
        stride_dkb,
        stride_dkh,
        stride_dkn,
        stride_nb,
        stride_nh,
        stride_nn,
        # constants
        BLOCK_N: tl.constexpr, # how big each block in N is
        BLOCK_K: tl.constexpr, # how big each block in C is
    ):
        
    # Launch grid: (total_blocks, B, H), where:
    #   total_blocks = cdiv(N1, BLOCK_M) + cdiv(N2, BLOCK_M)
    block_idx = tl.program_id(0)
    b = tl.program_id(1)
    h = tl.program_id(2)

    # create the offset array
    seq_offsets = tl.arange(0, BLOCK_N)
    chan_offsets = tl.arange(0, BLOCK_K)

    # compute the mean and variance pointers based on the block
    seq_shift = block_idx * BLOCK_N + seq_offsets
    norm_ptr = norm_ptr + b*stride_nb + h*stride_nh + seq_shift*stride_nn

    # Determine which tensor this block processes.
    if block_idx < split_idx:
        # Processing q.
        total_seq = q_seq_len
        seq_offsets = seq_shift
        rd_ptr = q_ptr + b*stride_qb + h*stride_qh + seq_offsets*stride_qn
        drd_ptr = doq_ptr + b*stride_doqb + h*stride_doqh + seq_offsets*stride_doqn
        wr_ptr = dq_ptr + b*stride_dqb + h*stride_dqh + seq_offsets*stride_dqn
        
    else:
        # Processing k.
        local_block = block_idx - split_idx
        total_seq = k_seq_len
        seq_offsets = seq_offsets + local_block * BLOCK_N
        rd_ptr = k_ptr + b*stride_kb + h*stride_kh + seq_offsets*stride_kn
        drd_ptr = dok_ptr + b*stride_dokb + h*stride_dokh + seq_offsets*stride_dokn
        wr_ptr = dk_ptr + b*stride_dkb + h*stride_dkh + seq_offsets*stride_dkn

    tl.debug_barrier()
    
    # Compute a mask to ensure we don't go beyond the total sequence length.
    valid_mask = (seq_offsets[:,None] < total_seq) & (chan_offsets[None,:] < head_dim)

    # Load the (BLOCK_M, BLOCK_K) tile from q or k using the valid mask. - make sure to promote to float32
    row_block = tl.load(rd_ptr[:,None]+chan_offsets[None,:], mask=valid_mask, other=0.0).to(tl.float32)
    drow_block = tl.load(drd_ptr[:,None]+chan_offsets[None,:], mask=valid_mask, other=0.0).to(tl.float32)

    # load the mean and rstd buffers
    norm_val = tl.load(norm_ptr[:,None] + tl.arange(0,2)[None,:], mask=seq_offsets[:,None] < total_seq, other=0.0).to(tl.float32)
    mean, rstd = tl.split(norm_val)
    
    # --- compute the norm ---
    # we have to compute the norm again from the buffers to get row_block_hat
    row_block_hat = tl.where(valid_mask, row_block - mean[:,None], 0.0)*rstd[:,None]

    # nor we can use a simplified form of the VDP since the weight is essentially 1
    # dx = 1/sigma*(dy - (1/N*x<dot>dy)*x)
    inner_dg = tl.sum(row_block_hat * drow_block, axis=1, keep_dims=True) / head_dim
    mean_dg = tl.sum(drow_block, axis=1, keep_dims=True)/head_dim
    drow_block_new = (drow_block - row_block_hat*inner_dg - mean_dg) * rstd[:,None]
    
    # Store the transformed block back using the valid mask.
    tl.store(wr_ptr[:,None]+chan_offsets[None,:], drow_block_new, mask=valid_mask)

def _fused_ln_qk_fwd(q, k, heads_second=True, norm_eps=1e-6, kernel_cache=None):

    assert q.ndim == k.ndim == 4
    assert q.dtype == k.dtype, "All tensors must have the same type"
    assert q.dtype in [torch.float16, torch.bfloat16, torch.float32], "Only support fp16, bf16, and fp32"
    assert q.is_cuda and k.is_cuda
    assert q.stride(-1) == k.stride(-1) == 1
    
    # allocate the outputs
    qo = torch.empty_like(q)
    ko = torch.empty_like(k)
    
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

    assert (bq, hq, qdim) == (bk, hk, kdim)
    assert qdim <= 256, "Only supports head dimensions up to 256"
    
    MULT=1
    BLOCK_K, BLOCK_N = (
        (32,16*MULT)
        if qdim <= 32
        else ((64,8*MULT) if qdim <= 64 else ((128,4*MULT) if qdim <= 128 else (256,2*MULT)))
    )
    blocks_q = triton.cdiv(q_seq_len, BLOCK_N)
    blocks_k = triton.cdiv(k_seq_len, BLOCK_N)

    # allocate the buffers
    buff_size = (blocks_q+blocks_k)*BLOCK_N
    norm_buff = torch.empty((bq, hq, buff_size, 2), device=q.device, dtype=torch.float32)
    
    grid = lambda META: (blocks_q+blocks_k, bq, hq)  # noqa

    if (kernel_cache is not None) and (kernel_cache.fwd is not None):
        grid_tuple = grid({})
        kernel_cache.fwd[grid_tuple](
            # -- inputs --
            q, 
            k, 
            # -- outputs --
            qo,
            ko,
            # -- norm buffers --
            norm_buff,  # pointer to the mean
            # -- parameters --
            q_seq_len,
            k_seq_len,
            blocks_q, # where q ends and k begins
            qdim,
            norm_eps,
            # -- strides --
            *qstrides,
            *kstrides,
            *qostrides,
            *kostrides,
            norm_buff.stride(0),
            norm_buff.stride(1),
            norm_buff.stride(2),    
        )
    else:
        kfwd = _fused_ln_qk_kernel_fwd[grid](
            # -- inputs --
            q, 
            k, 
            # -- outputs --
            qo,
            ko,
            # -- norm buffers --
            norm_buff,  # pointer to the mean
            # -- parameters --
            q_seq_len,
            k_seq_len,
            blocks_q, # where q ends and k begins
            qdim,
            norm_eps,
            # -- strides --
            *qstrides,
            *kstrides,
            *qostrides,
            *kostrides,
            norm_buff.stride(0),
            norm_buff.stride(1),
            norm_buff.stride(2),
            # constants
            BLOCK_N, # how big each block in N is
            BLOCK_K, # how big each block in C is
            num_warps=1,
        )
        if kernel_cache is not None:
            kernel_cache.fwd = kfwd
            
    return qo, ko, norm_buff

def _fused_ln_qk_bwd(q, k, doq, dok, norm_buff, dq, dk, heads_second=True, norm_eps=1e-6, kernel_cache=None):

    assert q.stride(-1) == k.stride(-1) == doq.stride(-1) == dok.stride(-1) == 1
    assert dq.stride(-1) == dk.stride(-1) == 1

    if heads_second:
        bq,hq,q_seq_len,qdim = doq.shape
        bk,hk,k_seq_len,kdim = dok.shape
        # strides should be in B,H,N
        qstrides = (q.stride(0), q.stride(1), q.stride(2))
        kstrides = (k.stride(0), k.stride(1), k.stride(2))
        doqstrides = (doq.stride(0), doq.stride(1), doq.stride(2))
        dokstrides = (dok.stride(0), dok.stride(1), dok.stride(2))
        dqstrides = (dq.stride(0), dq.stride(1), dq.stride(2))
        dkstrides = (dk.stride(0), dk.stride(1), dk.stride(2))
    else:
        bq,q_seq_len,hq,qdim = doq.shape
        bk,k_seq_len,hk,kdim = dok.shape   
        # strides should be in B,H,N
        qstrides = (q.stride(0), q.stride(2), q.stride(1))
        kstrides = (k.stride(0), k.stride(2), k.stride(1))
        doqstrides = (doq.stride(0), doq.stride(2), doq.stride(1))
        dokstrides = (dok.stride(0), dok.stride(2), dok.stride(1))
        dqstrides = (dq.stride(0), dq.stride(2), dq.stride(1))
        dkstrides = (dk.stride(0), dk.stride(2), dk.stride(1))

    MULT=1
    BLOCK_K, BLOCK_N = (
        (32,16*MULT)
        if qdim <= 32
        else ((64,8*MULT) if qdim <= 64 else ((128,4*MULT) if qdim <= 128 else (256,2*MULT)))
    )
    blocks_q = triton.cdiv(q_seq_len, BLOCK_N)
    blocks_k = triton.cdiv(k_seq_len, BLOCK_N)
    
    grid = lambda META: (blocks_q+blocks_k, bq, hq)  # noqa

    if (kernel_cache is not None) and (kernel_cache.bwd is not None):
        grid_tuple = grid({})
        kernel_cache.bwd[grid_tuple](
            # -- inputs --
            q,
            k,
            doq, 
            dok, 
            # -- norm buffers --
            norm_buff,  # pointer to the mean
            # -- outputs --
            dq,
            dk,
            # -- parameters --
            q_seq_len,
            k_seq_len,
            blocks_q, # where q ends and k begins
            qdim,
            norm_eps,
            # -- strides --
            *qstrides,
            *kstrides,
            *doqstrides,
            *dokstrides,
            *dqstrides,
            *dkstrides,
            norm_buff.stride(0),
            norm_buff.stride(1),
            norm_buff.stride(2),
        )
    else:
        kbwd = _fused_ln_qk_kernel_bwd[grid](
            # -- inputs --
            q,
            k,
            doq, 
            dok, 
            # -- norm buffers --
            norm_buff,  # pointer to the mean
            # -- outputs --
            dq,
            dk,
            # -- parameters --
            q_seq_len,
            k_seq_len,
            blocks_q, # where q ends and k begins
            qdim,
            norm_eps,
            # -- strides --
            *qstrides,
            *kstrides,
            *doqstrides,
            *dokstrides,
            *dqstrides,
            *dkstrides,
            norm_buff.stride(0),
            norm_buff.stride(1),
            norm_buff.stride(2),
            # constants
            BLOCK_N, # how big each block in N is
            BLOCK_K, # how big each block in C is
            num_warps=1,
        )
        if kernel_cache is not None:
            kernel_cache.bwd = kbwd
    return 


class Fused_LNQK_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, heads_second=True, norm_eps=1e-6, kernel_cache=None):
        """
        """
        # Make sure that the last dimension is contiguous
        q, k = [x if x.stride(-1) == 1 else x.contiguous() for x in [q, k]]
        qo, qk, norm_buff = _fused_ln_qk_fwd(q, k, heads_second=heads_second, norm_eps=norm_eps,
                                            kernel_cache=kernel_cache)
        # Save the cache reference for the backward pass
        ctx.kernel_cache = kernel_cache
        # save the state for backward pass
        ctx.save_for_backward(q,k,norm_buff)
        ctx.heads_second = heads_second
        ctx.norm_eps = norm_eps
        return qo, qk

    @staticmethod
    def backward(ctx, doq, dok):
        q, k, norm_buff = ctx.saved_tensors
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dq = torch.empty_like(doq)
            dk = torch.empty_like(dok)
            _fused_ln_qk_bwd(
                q,
                k,
                doq, 
                dok, 
                norm_buff,
                dq, 
                dk, 
                heads_second=ctx.heads_second, 
                norm_eps=ctx.norm_eps,
                kernel_cache=ctx.kernel_cache,
            )
        return dq, dk, None, None, None


def triton_fused_ln_func_fn(query,key,heads_second=True, norm_eps=1e-6, kernel_cache=None):
    return Fused_LNQK_Func.apply(query,key,heads_second, norm_eps, kernel_cache)
