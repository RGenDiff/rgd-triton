import torch
import triton
import triton.language as tl


@triton.jit
def _fused_norm_kernel_fwd(
        # -- inputs --
        x1_ptr, 
        x2_ptr, 
        # -- outputs --
        x1o_ptr,
        x2o_ptr,
        # -- norm buffers --
        norm_ptr,  # pointer to the length
        # -- parameters --
        seq_len_x1,
        seq_len_x2,
        split_idx, # where q ends and k begins
        norm_eps, # epsilon to avoid division by zero
        # -- strides --
        stride_x1b,
        stride_x1n,
        stride_x2b,
        stride_x2n,
        stride_ox1b,
        stride_ox1n,
        stride_ox2b,
        stride_ox2n,
        stride_onb,
        # constants
        DIM_X1: tl.constexpr,
        DIM_X2: tl.constexpr,
        BLOCK_N: tl.constexpr, # how big each block in N is
        BLOCK_K: tl.constexpr, # how big each block in C is
    ):
        
    # Launch grid: (total_blocks, B), where:
    #   total_blocks = cdiv(N1, BLOCK_M) + cdiv(N2, BLOCK_M)
    block_idx = tl.program_id(0)
    bidx = tl.program_id(1)

    # create the offset array
    seq_offsets = tl.arange(0, BLOCK_N)
    chan_offsets = tl.arange(0, BLOCK_K)

    # compute the mean and variance pointers based on the block
    seq_shift = block_idx * BLOCK_N + seq_offsets
    norm_ptr = norm_ptr + bidx*stride_onb + seq_shift

    # Determine which tensor this block processes.
    if block_idx < split_idx:
        # Processing q.
        total_seq = seq_len_x1
        chan_dim = DIM_X1
        seq_offsets = seq_shift
        rd_ptr = x1_ptr + bidx*stride_x1b + seq_offsets*stride_x1n
        wr_ptr = x1o_ptr + bidx*stride_ox1b + seq_offsets*stride_ox1n
        
    else:
        # Processing k.
        local_block = block_idx - split_idx
        total_seq = seq_len_x2
        chan_dim = DIM_X2
        seq_offsets = seq_offsets + local_block * BLOCK_N
        rd_ptr = x2_ptr + bidx*stride_x2b +  seq_offsets*stride_x2n
        wr_ptr = x2o_ptr + bidx*stride_ox2b + seq_offsets*stride_ox2n

    # -- compute the rms mean --
    len_buffer = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    tl.debug_barrier()
    
    # loop over the k-blocks in the channel dim
    for i in range(0, chan_dim, BLOCK_K):
        channels = i + chan_offsets
        # Compute a mask to ensure we don't go beyond the total sequence length.
        valid_mask = (seq_offsets[:,None] < total_seq) & (channels < chan_dim)

        # Load the (BLOCK_M, BLOCK_K) tile from q or k using the valid mask. - make sure to promote to float32
        row_block = tl.load(rd_ptr[:,None] + channels[None,:], mask=valid_mask, other=0.0).to(tl.float32)

        # compute the square of the x input
        row_block2 = row_block*row_block
        len_buffer += row_block2

    tl.debug_barrier()
    
    # now reduce over the block and take the sqrt
    # compute over the channel dim (axis=1)
    len_val = tl.sum(len_buffer, axis=1)
    len_val = tl.sqrt(len_val)
    rgate = (len_val < norm_eps)
    rlen = tl.where(rgate, 1.0/norm_eps, 1.0/len_val)
    
    # store the rlen value
    tl.store(norm_ptr, rlen, mask=seq_offsets < total_seq)

    # -- compute the normalized (and scaled) rms output
    tl.debug_barrier()
    
    # loop over the k-blocks in the channel dim
    for i in range(0, chan_dim, BLOCK_K):
        channels = i + chan_offsets
        # Compute a mask to ensure we don't go beyond the total sequence length.
        valid_mask = (seq_offsets[:,None] < total_seq) & (channels < chan_dim)

        # Load the (BLOCK_M, BLOCK_K) tile from q or k using the valid mask. - make sure to promote to float32
        row_block = tl.load(rd_ptr[:,None] + channels[None,:], mask=valid_mask, other=0.0).to(tl.float32)
    
        # compute the norm
        new_row_block = row_block * rlen[:,None]
    
        # Store the transformed block back using the valid mask.
        tl.store(wr_ptr[:,None] + channels[None,:], new_row_block, mask=valid_mask)

    # end of kernel



@triton.jit
def _fused_norm_kernel_bwd(
        # -- inputs --
        x1_ptr,
        x2_ptr,
        dox1_ptr, 
        dox2_ptr,         
        # -- norm buffers --
        norm_ptr,  # pointer to the mean
        # -- outputs --
        dx1_ptr,
        dx2_ptr,
        # -- parameters --
        seq_len_x1,
        seq_len_x2,
        blocks_x1,
        blocks_x2,
        split_idx, # where q ends and k begins
        dim_x1,
        dim_x2,
        norm_eps, # epsilon to avoid division by zero
        # -- strides --
        stride_x1b,
        stride_x1n,
        stride_x2b,
        stride_x2n,
        stride_dox1b,
        stride_dox1n,
        stride_dox2b,
        stride_dox2n,
        stride_dx1b,
        stride_dx1n,
        stride_dx2b,
        stride_dx2n,
        stride_nb,
        # constants
        BLOCK_N: tl.constexpr, # how big each block in N is
        BLOCK_K: tl.constexpr, # how big each block in C is
        NUM_K_BLOCKS: tl.constexpr, # how many blocks in the C dim
    ):
        
    # Launch grid: (total_blocks, B, blocks_k), where:
    #   total_blocks = cdiv(N1, BLOCK_M) + cdiv(N2, BLOCK_M)
    #   blocks_k is dim/BLOCK_K
    block_idx = tl.program_id(0)
    bidx = tl.program_id(1)
    cidx = tl.program_id(2)

    # create the offset array
    seq_offsets = tl.arange(0, BLOCK_N)
    chan_offsets = tl.arange(0, BLOCK_K)

    # compute the mean and variance pointers based on the block
    seq_shift = block_idx * BLOCK_N + seq_offsets
    norm_ptr = norm_ptr + bidx*stride_nb + seq_shift

    # Determine which tensor this block processes.
    if block_idx < split_idx:
        # Processing q.
        total_seq = seq_len_x1
        seq_offsets = seq_shift
        chan_dim = dim_x1
        rd_ptr = x1_ptr + bidx*stride_x1b + seq_offsets*stride_x1n
        drd_ptr = dox1_ptr + bidx*stride_dox1b + seq_offsets*stride_dox1n
        wr_ptr = dx1_ptr + bidx*stride_dx1b + seq_offsets*stride_dx1n
        
    else:
        # Processing k.
        local_block = block_idx - split_idx
        total_seq = seq_len_x2
        seq_offsets = seq_offsets + local_block * BLOCK_N
        chan_dim = dim_x2
        rd_ptr = x2_ptr + bidx*stride_x2b  + seq_offsets*stride_x2n
        drd_ptr = dox2_ptr + bidx*stride_dox2b + seq_offsets*stride_dox2n
        wr_ptr = dx2_ptr + bidx*stride_dx2b  + seq_offsets*stride_dx2n

    tl.debug_barrier()
    # load the mean and rstd buffers
    rlen_val = tl.load(norm_ptr, mask=seq_offsets < total_seq, other=0.0).to(tl.float32)

    
    # -- create a few buffers --
    
    # save states to avoid multiple loads
    row_block_hat = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    drow_block = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    # buffer to accumulate for the derivative
    inner_dg_buffer = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

    # now compute the back-prop for dx

    # the form is similar to VJP with inner = dy*w*x/N
    # dx = w*(dy - x*inner*rms*rms)*rms
    tl.debug_barrier()

    for i in range(0, NUM_K_BLOCKS):
        channels = i*BLOCK_K + chan_offsets
            
        # Compute a mask to ensure we don't go beyond the total sequence length.
        valid_mask = (seq_offsets[:,None] < total_seq) & (channels[None,:] < chan_dim)

        # Load the (BLOCK_M, BLOCK_K) tile from q or k using the valid mask. - make sure to promote to float32
        _drow_block = tl.load(drd_ptr[:,None]+channels[None,:], mask=valid_mask, other=0.0).to(tl.float32)
        _row_block = tl.load(rd_ptr[:,None]+channels[None,:], mask=valid_mask, other=0.0).to(tl.float32)
        
        # recompute the norm
        _row_block_hat = _row_block*rlen_val[:,None]

        inner_dg_buffer += _row_block_hat * _drow_block

        if i == cidx:
            row_block_hat = _row_block_hat
            drow_block = _drow_block

    tl.debug_barrier()
    # now we can offset based on the K block
    chan_offsets = chan_offsets + cidx*BLOCK_K
    valid_mask = (seq_offsets[:,None] < total_seq) & (chan_offsets[None,:] < chan_dim)
    
    # now we can reduce the inner_dg_buffer over the channel dim
    inner_dg = tl.sum(inner_dg_buffer, axis=1, keep_dims=True) 
    
    drow_block_new = (drow_block - row_block_hat*inner_dg) * rlen_val[:,None]
    
    # Store the transformed block back using the valid mask.
    tl.store(wr_ptr[:,None]+chan_offsets[None,:], drow_block_new, mask=valid_mask)
    
        
    
def _fused_norm_fwd(x1, x2, norm_eps=1e-6, kernel_cache=None):

    assert x1.ndim == x2.ndim == 3
    assert x1.dtype == x2.dtype, "All tensors must have the same type"
    assert x1.dtype in [torch.float16, torch.bfloat16, torch.float32], "Only support fp16, bf16, and fp32"
    assert x1.is_cuda and x2.is_cuda
    assert x1.stride(-1) == x2.stride(-1) == 1
    assert x1.shape[0] == x2.shape[0]
    
    # allocate the outputs
    x1o = torch.empty_like(x1)
    x2o = torch.empty_like(x2)

    bx1, seq_len_x1, dim_x1 = x1.shape
    bx2, seq_len_x2, dim_x2 = x2.shape

    dim = max(dim_x1, dim_x2)
    
    MULT=1
    BLOCK_K, BLOCK_N  = (
        (32, 16*MULT) if dim <= 32 else (
            (64, 8*MULT) if dim <= 64 else (
                (128, 4*MULT) if dim <= 128 else
                   (256, 2*MULT)
                
            )
        )
    )
    blocks_x1 = triton.cdiv(seq_len_x1, BLOCK_N)
    blocks_x2 = triton.cdiv(seq_len_x2, BLOCK_N)
    blocks_k = triton.cdiv(dim, BLOCK_K)

    # allocate the buffers
    buff_size = (blocks_x1 + blocks_x2)*BLOCK_N
    # add two entries, because we also need to set the gate
    norm_buff = torch.empty((bx1, buff_size), device=x1.device, dtype=torch.float32)
    
    grid = lambda META: (blocks_x1 + blocks_x2, bx1, 1) # grid must be 3D for caching 
    if (kernel_cache is not None) and (kernel_cache.fwd is not None):
        grid_tuple = grid({})
        kernel_cache.fwd[grid_tuple](
            # -- inputs --
            x1, 
            x2, 
            # -- outputs --
            x1o,
            x2o,
            # -- norm buffers --
            norm_buff,  # pointer to the rms
            # -- parameters --
            seq_len_x1,
            seq_len_x2,
            blocks_x1, # where q ends and k begins
            norm_eps,
            # -- strides --
            x1.stride(0), #batch
            x1.stride(1), #seq
            x2.stride(0), #batch
            x2.stride(1), #seq
            x1o.stride(0),
            x1o.stride(1),
            x2o.stride(0),
            x2o.stride(1),
            norm_buff.stride(0), #batch
        )
    else:
        kfwd = _fused_norm_kernel_fwd[grid](
            # -- inputs --
            x1, 
            x2, 
            # -- outputs --
            x1o,
            x2o,
            # -- norm buffers --
            norm_buff,  # pointer to the rms
            # -- parameters --
            seq_len_x1,
            seq_len_x2,
            blocks_x1, # where q ends and k begins
            norm_eps,
            # -- strides --
            x1.stride(0), #batch
            x1.stride(1), #seq
            x2.stride(0), #batch
            x2.stride(1), #seq
            x1o.stride(0),
            x1o.stride(1),
            x2o.stride(0),
            x2o.stride(1),
            norm_buff.stride(0), #batch
            # constants
            dim_x1,
            dim_x2,
            BLOCK_N, # how big each block in N is
            BLOCK_K, # how big each block in C is
            num_warps=1,
        )
        if kernel_cache is not None:
            kernel_cache.fwd = kfwd
            
    return x1o, x2o, norm_buff


def _fused_norm_bwd(dox1, dox2, x1, x2, norm_buff, dx1, dx2, norm_eps=1e-6, kernel_cache=None):

    assert x1.stride(-1) == x2.stride(-1) == dox1.stride(-1) == dox2.stride(-1) == 1, f"{dox1.stride(-1)=}, {dox2.stride(-1)=}"
    assert dx1.stride(-1) == dx2.stride(-1) == 1
    
    bx1, seq_len_x1, dim_x1 = x1.shape
    bx2, seq_len_x2, dim_x2 = x2.shape

    dim = max(dim_x1, dim_x2)    
    
    MULT=1
    BLOCK_K, BLOCK_N  = (
        (32, 16*MULT) if dim <= 32 else (
            (64, 8*MULT) if dim <= 64 else (
                (128, 4*MULT) if dim <= 128 else
                   (256, 2*MULT)
                
            )
        )
    )
    blocks_x1 = triton.cdiv(seq_len_x1, BLOCK_N)
    blocks_x2 = triton.cdiv(seq_len_x2, BLOCK_N)
    total_blocks = blocks_x1 + blocks_x2
    blocks_k = triton.cdiv(dim, BLOCK_K)
    
    grid = lambda META: (total_blocks, bx1, blocks_k)  # noqa

    if (kernel_cache is not None) and (kernel_cache.bwd is not None):
        grid_tuple = grid({})
        kernel_cache.bwd[grid_tuple](
            # -- inputs --
            x1,
            x2,
            dox1, 
            dox2, 
            # -- norm buffers --
            norm_buff,  # pointer to the mean
            # -- outputs --
            dx1,
            dx2,
            # -- parameters --
            seq_len_x1,
            seq_len_x2,
            blocks_x1,
            blocks_x2,
            blocks_x1, # where q ends and k begins
            dim_x1,
            dim_x2,
            norm_eps,
            # -- strides --
            x1.stride(0), # batch
            x1.stride(1), # sequence
            x2.stride(0), # batch
            x2.stride(1), # batch
            dox1.stride(0), # batch
            dox1.stride(1), 
            dox2.stride(0),
            dox2.stride(1),
            dx1.stride(0),
            dx1.stride(1),
            dx2.stride(0),
            dx2.stride(1),
            norm_buff.stride(0), # batch
        )
    else:
        kbwd = _fused_norm_kernel_bwd[grid](
            # -- inputs --
            x1,
            x2,
            dox1, 
            dox2, 
            # -- norm buffers --
            norm_buff,  # pointer to the mean
            # -- outputs --
            dx1,
            dx2,
            # -- parameters --
            seq_len_x1,
            seq_len_x2,
            blocks_x1,
            blocks_x2,
            blocks_x1, # where q ends and k begins
            dim_x1,
            dim_x2,
            norm_eps,
            # -- strides --
            x1.stride(0), # batch
            x1.stride(1), # sequence
            x2.stride(0), # batch
            x2.stride(1), # batch
            dox1.stride(0), # batch
            dox1.stride(1), 
            dox2.stride(0),
            dox2.stride(1),
            dx1.stride(0),
            dx1.stride(1),
            dx2.stride(0),
            dx2.stride(1),
            norm_buff.stride(0), # batch
            # constants
            BLOCK_N, # how big each block in N is
            BLOCK_K, # how big each block in C is
            blocks_k,
            num_warps=1,
        )
        if kernel_cache is not None:
            kernel_cache.bwd = kbwd
    return 


class Fused_Norm_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1,x2, norm_eps=1e-6, kernel_cache=None):
        """
        """
        # Make sure that the last dimension is contiguous
        x1, x2 = [x if x.stride(-1) == 1 else x.contiguous() for x in [x1, x2]]
        ox1, ox2, norm_buff = _fused_norm_fwd(x1, x2, norm_eps=norm_eps,
                                             kernel_cache=kernel_cache)
        # Save the cache reference for the backward pass
        ctx.kernel_cache = kernel_cache
        # save the state for backward pass
        ctx.save_for_backward(x1, x2, norm_buff)
        ctx.norm_eps = norm_eps
        return ox1, ox2

    @staticmethod
    def backward(ctx, dox1, dox2):

        # it seems that if we follow the fused norm with a transpose, it makes the gradient nolonger contiguous
        dox1, dox2 = [x if x.stride(-1) == 1 else x.contiguous() for x in [dox1, dox2]]
        
        x1, x2, norm_buff = ctx.saved_tensors
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dx1 = torch.empty_like(x1)
            dx2 = torch.empty_like(x2)
            _fused_norm_bwd(
                dox1,
                dox2,
                x1,
                x2,
                norm_buff,
                dx1, 
                dx2, 
                kernel_cache=ctx.kernel_cache,
            )
        # must clone because we may have changed the expected structure of dox1,dox2 with the contiguous
        return dx1.clone(), dx2.clone(), None, None


def triton_fused_norm_func_fn(x1, x2, norm_eps=1e-6, kernel_cache=None):
    return Fused_Norm_Func.apply(x1, x2, norm_eps, kernel_cache)
