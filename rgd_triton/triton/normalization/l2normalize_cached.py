import torch
import triton
import triton.language as tl

@triton.jit
def _norm_kernel_fwd(
        # -- inputs --
        x_ptr, 
        # -- outputs --
        xo_ptr,
        # -- norm buffers --
        norm_ptr,  # pointer to the length
        # -- parameters --
        seq_len,
        norm_eps, # epsilon to avoid division by zero
        # -- strides --
        stride_xb,
        stride_xn,
        stride_oxb,
        stride_oxn,
        stride_onb,
        # constants
        DIM_X: tl.constexpr,
        BLOCK_N: tl.constexpr, # how big each block in N is
        BLOCK_K: tl.constexpr, # how big each block in C is
        USE_PTX: tl.constexpr,
    ):
        
    # Launch grid: (total_blocks, B), where:
    #   total_blocks = cdiv(N, BLOCK_M)
    block_idx = tl.program_id(0)
    bidx = tl.program_id(1)

    # create the offset array
    seq_offsets = tl.arange(0, BLOCK_N)
    chan_offsets = tl.arange(0, BLOCK_K)

    # compute the mean and variance pointers based on the block
    seq_shift = block_idx * BLOCK_N + seq_offsets
    norm_ptr = norm_ptr + bidx*stride_onb + seq_shift
    #rstd_ptr = rstd_ptr + b*stride_ovb + h*stride_ovh + seq_shift

    # Processing q.
    total_seq = seq_len
    chan_dim = DIM_X
    seq_offsets = seq_shift
    rd_ptr = x_ptr + bidx*stride_xb + seq_offsets*stride_xn
    wr_ptr = xo_ptr + bidx*stride_oxb + seq_offsets*stride_oxn
        
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

    if not USE_PTX:
        rgate = (len_val < norm_eps)
        rlen = tl.where(rgate, 1.0/norm_eps, 1.0/len_val)

    else:
        # For each (rlen) in (len_val), perform the following:
        # check to see if it's less than norm_eps
        # Do the above 4 elements at a time.
        rlen = tl.inline_asm_elementwise(
            asm="""
            max.f32 $0, $1, $2;
            rcp.approx.f32 $0, $0;
            """,
            constraints=(
                # one output
                "=r,"
                # 2 inputs
                "r,r"
                ),
            args=[len_val, norm_eps], # these are the inputs
            dtype=(tl.float32,), # note: this is the output dtype
            is_pure=True,
            pack=1,
        )
    
    
    #rgate = rgate.to(tl.float32)
    #rlen = tl.rsqrt(len_val + norm_eps)
    #norm_val = tl.join(rlen, rgate)
    
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
def _norm_kernel_bwd(
        # -- inputs --
        x_ptr,
        dox_ptr,     
        # -- norm buffers --
        norm_ptr,  # pointer to the mean
        # -- outputs --
        dx_ptr,
        # -- parameters --
        seq_len,
        blocks_x,
        dim_x,
        norm_eps, # epsilon to avoid division by zero
        # -- strides --
        stride_xb,
        stride_xn,
        stride_doxb,
        stride_doxn,
        stride_dxb,
        stride_dxn,
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

    total_seq = seq_len
    seq_offsets = seq_shift
    chan_dim = dim_x
    rd_ptr = x_ptr + bidx*stride_xb + seq_offsets*stride_xn
    drd_ptr = dox_ptr + bidx*stride_doxb + seq_offsets*stride_doxn
    wr_ptr = dx_ptr + bidx*stride_dxb + seq_offsets*stride_dxn
        

    tl.debug_barrier()
    
    # load the mean and rstd buffers
    rlen_val = tl.load(norm_ptr, mask=seq_offsets < total_seq, other=0.0).to(tl.float32)

    #rlen_val, rgate = norm_val.split()
    
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
    #drow_block_new = tl.where(rgate[:,None] != 0.0, drow_block/norm_eps, drow_block_new)
    
    # Store the transformed block back using the valid mask.
    tl.store(wr_ptr[:,None]+chan_offsets[None,:], drow_block_new, mask=valid_mask)
    
        
    
def _norm_fwd(x, norm_eps=1e-6, use_ptx=False, kernel_cache=None):

    assert x.ndim == 3
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Only support fp16, bf16, and fp32"
    assert x.is_cuda 
    assert x.stride(-1) == 1
    
    # allocate the outputs
    xo = torch.empty_like(x)

    bx, seq_len_x, dim = x.shape
    
    MULT=1
    BLOCK_K, BLOCK_N  = (
        (32, 16*MULT) if dim <= 32 else (
            (64, 8*MULT) if dim <= 64 else (
                (128, 4*MULT) if dim <= 128 else
                    (256, 2*MULT)
            )
        )
    )

    blocks_x = triton.cdiv(seq_len_x, BLOCK_N)
    blocks_k = triton.cdiv(dim, BLOCK_K)

    # allocate the buffers
    buff_size = blocks_x*BLOCK_N
    # add two entries, because we also need to set the gate
    norm_buff = torch.empty((bx, buff_size), device=x.device, dtype=torch.float32)
    #rstd_buff = torch.empty((bq, hq, buff_size), device=q.device, dtype=torch.float32)
    
    grid = lambda META: (blocks_x, bx, 1) # grid must be 3D for caching 

    if (kernel_cache is not None) and (kernel_cache.fwd is not None):
        grid_tuple = grid({})
        kernel_cache.fwd[grid_tuple](
            # -- inputs --
            x, 
            # -- outputs --
            xo,
            # -- norm buffers --
            norm_buff,  # pointer to the rms
            # -- parameters --
            seq_len_x,
            norm_eps,
            # -- strides --
            x.stride(0), #batch
            x.stride(1), #seq
            xo.stride(0),
            xo.stride(1),
            norm_buff.stride(0), #batch
            # constants
        )
    else:
        kfwd = _norm_kernel_fwd[grid](
            # -- inputs --
            x, 
            # -- outputs --
            xo,
            # -- norm buffers --
            norm_buff,  # pointer to the rms
            # -- parameters --
            seq_len_x,
            norm_eps,
            # -- strides --
            x.stride(0), #batch
            x.stride(1), #seq
            xo.stride(0),
            xo.stride(1),
            norm_buff.stride(0), #batch
            # constants
            dim,
            BLOCK_N, # how big each block in N is
            BLOCK_K, # how big each block in C is
            use_ptx,
            num_warps=1,
        )
        if kernel_cache is not None:
            kernel_cache.fwd = kfwd
            
    return xo, norm_buff


def _norm_bwd(dox, x, norm_buff, dx, norm_eps=1e-6, kernel_cache=None):

    assert x.stride(-1) == dox.stride(-1) == dx.stride(-1) == 1
    
    bx, seq_len_x, dim = x.shape
    
    MULT=1
    BLOCK_K, BLOCK_N  = (
        (32, 16*MULT) if dim <= 32 else (
            (64, 8*MULT) if dim <= 64 else (
                (128, 4*MULT) if dim <= 128 else
                   (256, 2*MULT)
                
            )
        )
    )
    blocks_x = triton.cdiv(seq_len_x, BLOCK_N)
    blocks_k = triton.cdiv(dim, BLOCK_K)
    
    grid = lambda META: (blocks_x, bx, blocks_k)  # noqa
    
    if (kernel_cache is not None) and (kernel_cache.bwd is not None):
        grid_tuple = grid({})
        kernel_cache.bwd[grid_tuple](
            # -- inputs --
            x,
            dox, 
            # -- norm buffers --
            norm_buff,  # pointer to the mean
            #rstd_buff,  # pointer to the 1/std
            # -- outputs --
            dx,
            # -- parameters --
            seq_len_x,
            blocks_x,
            dim,
            norm_eps,
            # -- strides --
            x.stride(0), # batch
            x.stride(1), # sequence
            dox.stride(0), # batch
            dox.stride(1), 
            dx.stride(0),
            dx.stride(1),
            norm_buff.stride(0), # batch
        )
    else:
        kbwd = _norm_kernel_bwd[grid](
            # -- inputs --
            x,
            dox, 
            # -- norm buffers --
            norm_buff,  # pointer to the mean
            #rstd_buff,  # pointer to the 1/std
            # -- outputs --
            dx,
            # -- parameters --
            seq_len_x,
            blocks_x,
            dim,
            norm_eps,
            # -- strides --
            x.stride(0), # batch
            x.stride(1), # sequence
            dox.stride(0), # batch
            dox.stride(1), 
            dx.stride(0),
            dx.stride(1),
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


class Norm_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, norm_eps=1e-6,use_ptx=False, kernel_cache=None):
        """
        """
        # Make sure that the last dimension is contiguous
        x = x if x.stride(-1) == 1 else x.contiguous()
        ox, norm_buff = _norm_fwd(x, norm_eps=norm_eps,use_ptx=use_ptx,
                                 kernel_cache=kernel_cache)
        # Save the cache reference for the backward pass
        ctx.kernel_cache = kernel_cache
        # save the state for backward pass
        ctx.save_for_backward(x, norm_buff)
        ctx.norm_eps = norm_eps
        return ox

    @staticmethod
    def backward(ctx, dox):

        x, norm_buff = ctx.saved_tensors
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dx = torch.empty_like(x)
            _norm_bwd(
                dox,
                x,
                norm_buff,
                dx, 
                kernel_cache=ctx.kernel_cache,
            )
        return dx, None, None, None


def triton_norm_func_fn(x,norm_eps=1e-6,use_ptx=False, kernel_cache=None):
    return Norm_Func.apply(x, norm_eps, use_ptx, kernel_cache)
    
    