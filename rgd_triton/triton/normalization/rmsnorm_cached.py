import torch
import triton
import triton.language as tl


@triton.jit
def _rms_kernel_fwd(
        # -- inputs --
        x_ptr, 
        w_ptr,
        # -- outputs --
        xo_ptr,
        # -- norm buffers --
        norm_ptr,  # pointer to the rms
        # -- parameters --
        seq_len_x,
        norm_eps, # epsilon to avoid division by zero
        # -- strides --
        stride_xb,
        stride_xn,
        stride_wb,
        stride_oxb,
        stride_oxn,
        stride_onb,
        # constants
        DIM_X: tl.constexpr,
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
    total_seq = seq_len_x
    chan_dim = DIM_X
    seq_offsets = seq_shift
    sc_ptr = w_ptr + bidx*stride_wb
    rd_ptr = x_ptr + bidx*stride_xb + seq_offsets*stride_xn
    wr_ptr = xo_ptr + bidx*stride_oxb + seq_offsets*stride_oxn
        

    # -- compute the rms mean --
    mean_buffer = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)

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
        mean_buffer += row_block2

    tl.debug_barrier()
    
    # now reduce over the block and take the sqrt
    # compute over the channel dim (axis=1)
    mean = tl.sum(mean_buffer, axis=1) / chan_dim
    rrms = tl.rsqrt(mean + norm_eps)

    # store the rrms value
    tl.store(norm_ptr, rrms, mask=seq_offsets < total_seq)

    # -- compute the normalized (and scaled) rms output

    tl.debug_barrier()
    
    # loop over the k-blocks in the channel dim
    for i in range(0, chan_dim, BLOCK_K):
        channels = i + chan_offsets
        # Compute a mask to ensure we don't go beyond the total sequence length.
        valid_mask = (seq_offsets[:,None] < total_seq) & (channels < chan_dim)

        # Load the (BLOCK_M, BLOCK_K) tile from q or k using the valid mask. - make sure to promote to float32
        row_block = tl.load(rd_ptr[:,None] + channels[None,:], mask=valid_mask, other=0.0).to(tl.float32)
    
        # load the weight
        row_scale = tl.load(sc_ptr + channels, channels < chan_dim, other=0.0).to(tl.float32)
    
        # compute the norm
        new_row_block = row_block * rrms[:,None] * row_scale[None, :]
    
        # Store the transformed block back using the valid mask.
        tl.store(wr_ptr[:,None] + channels[None,:], new_row_block, mask=valid_mask)

    # end of kernel


@triton.jit
def _rms_kernel_bwd(
        # -- inputs --
        x_ptr,
        w_ptr,
        dox_ptr, 
        # -- norm buffers --
        norm_ptr,  # pointer to the mean
        # -- outputs --
        dx_ptr,
        dwg_ptr,
        lock_ptr,
        # -- parameters --
        seq_len_x,
        blocks_x,
        dim_x,
        norm_eps, # epsilon to avoid division by zero
        # -- strides --
        stride_xb,
        stride_xn,
        stride_wb,
        stride_doxb,
        stride_doxn,
        stride_dxb,
        stride_dxn,
        stride_dwgb,
        stride_dwgn,
        stride_locksb,
        stride_locksk,
        stride_nb,
        # constants
        BLOCK_N: tl.constexpr, # how big each block in N is
        BLOCK_K: tl.constexpr, # how big each block in C is
        GROUP_SIZE: tl.constexpr, # how many reduction targets there are for dw
        NUM_K_BLOCKS: tl.constexpr, # how many blocks in the C dim
        DBG_KCOUNT: tl.constexpr, # write the lock count
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
    lock_ptr = lock_ptr + bidx*stride_locksb + cidx*stride_locksk
    
    # Determine which tensor this block processes.
    total_seq = seq_len_x
    seq_offsets = seq_shift
    sc_ptr = w_ptr + bidx*stride_wb
    group_id = (block_idx)%GROUP_SIZE
    dsc_ptr = dwg_ptr + bidx*stride_dwgb + group_id*stride_dwgn
    lock_ptr = lock_ptr + group_id
    chan_dim = dim_x
    rd_ptr = x_ptr + bidx*stride_xb + seq_offsets*stride_xn
    drd_ptr = dox_ptr + bidx*stride_doxb + seq_offsets*stride_doxn
    wr_ptr = dx_ptr + bidx*stride_dxb + seq_offsets*stride_dxn
        

    # in all cases, compute the count pointer by an offset of 2x group size for 2 and 3
    count_ptr = lock_ptr + GROUP_SIZE

    tl.debug_barrier()
    
    # load the mean and rstd buffers
    rrms_val = tl.load(norm_ptr, mask=seq_offsets < total_seq, other=0.0).to(tl.float32)

    # -- create a few buffers --
    
    # save states to avoid multiple loads
    row_block = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    row_block_hat = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    drow_block = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    row_scale = tl.zeros((BLOCK_K,), dtype=tl.float32)

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
        _row_scale = tl.load(sc_ptr + channels, channels < chan_dim, other=0.0).to(tl.float32)
        

        # recompute the norm
        _row_block_hat = _row_block*rrms_val[:,None]

        inner_dg_buffer += _row_block_hat * _drow_block * _row_scale[None,:]

        if i == cidx:
            row_block_hat = _row_block_hat
            row_block = _row_block
            drow_block = _drow_block
            row_scale = _row_scale

    tl.debug_barrier()
    
    # now we can offset based on the K block
    chan_offsets = chan_offsets + cidx*BLOCK_K
    valid_mask = (seq_offsets[:,None] < total_seq) & (chan_offsets[None,:] < chan_dim)
    
    # now we can reduce the inner_dg_buffer over the channel dim
    inner_dg = tl.sum(inner_dg_buffer, axis=1, keep_dims=True) / chan_dim
    
    drow_block_new = (drow_block*row_scale[None,:] - row_block_hat*inner_dg) * rrms_val[:,None]
    
    # Store the transformed block back using the valid mask.
    tl.store(wr_ptr[:,None]+chan_offsets[None,:], drow_block_new, mask=valid_mask)

    # compute the dw derivative
    drow_scale = tl.sum(row_block_hat * drow_block, axis=0) 
    
    row_scale_mask = chan_offsets < chan_dim
    dsc_ptr = dsc_ptr + chan_offsets
    
    # make sure all threads get to this point
    tl.debug_barrier()
    
    # finally, we need to write back the drow_scale
    while tl.atomic_cas(lock_ptr, 0, 1) == 1:
        pass
        
    count = tl.load(count_ptr)
    # if it's a first store, we don't need to load anything
    if count == 0:
        tl.atomic_xchg(count_ptr, 1)
    else:
        if DBG_KCOUNT:
            tl.atomic_add(count_ptr, 1)
            
        drow_scale += tl.load(dsc_ptr, mask=row_scale_mask, other=0)
        
    tl.store(dsc_ptr, drow_scale, mask=row_scale_mask)

    # make sure all threads get to this point
    tl.debug_barrier()

    # Release the lock
    tl.atomic_xchg(lock_ptr, 0)



    
@triton.jit
def _dw_kernel_bwd(
                dwg_ptr,
                dw_ptr,
                # -- parameters
                dim_x,
                total_groups,
                stride_dwb,
                stride_dwgb,
                stride_dwgn,
                # -- constants
                BLOCK_G: tl.constexpr, # how many blocks in the group dim
                BLOCK_C: tl.constexpr, # how many blocks in the channel dim
                GROUP_SIZE: tl.constexpr, # how many reduction targets there are for dw
                ):
    # the pid will be in [blocks_c, batch]
    block_cidx = tl.program_id(0)
    bidx = tl.program_id(1)

    chan_dim = dim_x
    total_groups = total_groups
    rd_stride = stride_dwgn
    rd_ptr = dwg_ptr + bidx*stride_dwgb
    wr_ptr = dw_ptr + bidx*stride_dwb

    
    
    # now we reduce over the blocks
    chan_offs = block_cidx*BLOCK_C + tl.arange(0,BLOCK_C)
    group_offs = tl.arange(0,BLOCK_G)
    dw_buffer = tl.zeros((BLOCK_G,BLOCK_C), dtype=tl.float32)
    
    tl.debug_barrier()
    
    for i in range(0, total_groups, BLOCK_G):
        offs = i + group_offs
        dw_buffer += tl.load(rd_ptr + offs[:,None]*rd_stride + chan_offs[None,:],
                             mask = (offs[:,None] < total_groups) & (chan_offs[None,:] < chan_dim),
                             other = 0.0)

    tl.debug_barrier()
    
    # finally reduce over block_G and write back
    reduced_dw = tl.sum(dw_buffer, axis=0)
    tl.store(wr_ptr + chan_offs, reduced_dw, mask = (chan_offs < chan_dim))
        
    
        
    

def _rms_fwd(x, w, norm_eps=1e-6, kernel_cache=None):

    assert x.ndim == w.ndim == 3, f"{x.shape=}, {w.shape=}"
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32], "Only support fp16, bf16, and fp32"
    assert x.is_cuda and w.is_cuda
    assert x.stride(-1) == w.stride(-1) == 1
    assert x.shape[0] == w.shape[0] 
    assert w.shape[-1] == x.shape[2]

    
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
    norm_buff = torch.empty((bx, buff_size), device=x.device, dtype=torch.float32)
    
    grid = lambda META: (blocks_x, bx, 1) # grid must be 3D for caching

    if (kernel_cache is not None) and (kernel_cache.fwd is not None):
        grid_tuple = grid({})
        kernel_cache.fwd[grid_tuple](
            # -- inputs --
            x, 
            w,
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
            w.stride(0),
            xo.stride(0),
            xo.stride(1),
            norm_buff.stride(0), #batch
        )
    else:
        kfwd = _rms_kernel_fwd[grid](
            # -- inputs --
            x, 
            w,
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
            w.stride(0),
            xo.stride(0),
            xo.stride(1),
            norm_buff.stride(0), #batch
            # constants
            dim,
            BLOCK_N, # how big each block in N is
            BLOCK_K, # how big each block in C is
            num_warps=1,
        )
        if kernel_cache is not None:
            kernel_cache.fwd = kfwd
        
    return xo, norm_buff


def _rms_bwd(dox, x, w, norm_buff, dx, dw, norm_eps=1e-6, kernel_cache=None):

    assert x.stride(-1) == dox.stride(-1) == w.stride(-1) == 1
    assert dw.stride(-1) == dx.stride(-1) == 1
    
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
    total_blocks = blocks_x
    blocks_k = triton.cdiv(dim, BLOCK_K)

    # have to compute the total groups from the BLOCK_N size and block_x1, blocks_x2

    num_programs = bx*total_blocks
    # create a set of writeback groups for reducing the weights
    GROUP_SIZE = (
        256 if num_programs <= 1024 else (
            128 if num_programs <= 4096 else (
                96 if num_programs <= 8192 else 
                    64
            )
        )
    )

    # finally, we can compute the total groups
    total_groups = blocks_x%GROUP_SIZE if blocks_x < GROUP_SIZE else GROUP_SIZE
    total_passes = triton.cdiv(blocks_x, GROUP_SIZE)
    

    # allocate the dw array
    dwg = torch.empty((bx, GROUP_SIZE, dim), dtype=torch.float32, device=x.device)
    locks = torch.zeros((bx, blocks_k, 2*GROUP_SIZE), dtype=torch.int32, device=x.device)

    
    grid = lambda META: (total_blocks, bx, blocks_k)  # noqa

    
    if (kernel_cache is not None) and (kernel_cache.bwd is not None):
        grid_tuple = grid({})
        kernel_cache.bwd[grid_tuple](
            # -- inputs --
            x,
            w,
            dox, 
            # -- norm buffers --
            norm_buff,  # pointer to the mean
            # -- outputs --
            dx,
            dwg,
            locks,
            # -- parameters --
            seq_len_x,
            blocks_x,
            dim,
            norm_eps,
            # -- strides --
            x.stride(0), # batch
            x.stride(1), # sequence
            w.stride(0),
            dox.stride(0), # batch
            dox.stride(1), 
            dx.stride(0),
            dx.stride(1),
            dwg.stride(0),
            dwg.stride(1),
            locks.stride(0),
            locks.stride(1),
            norm_buff.stride(0), # batch
        )
    else:
        kbwd = _rms_kernel_bwd[grid](
            # -- inputs --
            x,
            w,
            dox, 
            # -- norm buffers --
            norm_buff,  # pointer to the mean
            # -- outputs --
            dx,
            dwg,
            locks,
            # -- parameters --
            seq_len_x,
            blocks_x,
            dim,
            norm_eps,
            # -- strides --
            x.stride(0), # batch
            x.stride(1), # sequence
            w.stride(0),
            dox.stride(0), # batch
            dox.stride(1), 
            dx.stride(0),
            dx.stride(1),
            dwg.stride(0),
            dwg.stride(1),
            locks.stride(0),
            locks.stride(1),
            norm_buff.stride(0), # batch
            # constants
            BLOCK_N, # how big each block in N is
            BLOCK_K, # how big each block in C is
            GROUP_SIZE, # how many reduction targets there are for dw
            blocks_k,
            False,
            num_warps=1,
        )
        if kernel_cache is not None:
            kernel_cache.bwd = kbwd
    
    # do the final reduction
    BLOCK_C = 64
    BLOCK_G = 32

    total_c_blocks = triton.cdiv(dim, BLOCK_C)

    # grid is in [blocks_c, 2]
    grid = lambda META: (total_c_blocks, bx, 1)  # grid must be 3D for caching

    if (kernel_cache is not None) and (kernel_cache.bwd_dw is not None):
        grid_tuple = grid({})
        kernel_cache.bwd_dw[grid_tuple](
            # -- inputs --
            dwg,
            dw,
            # -- parameters
            dim,
            total_groups,
            dw.stride(0),
            dwg.stride(0),
            dwg.stride(1),
        )
    else:
        kbdw = _dw_kernel_bwd[grid](
            # -- inputs --
            dwg,
            dw,
            # -- parameters
            dim,
            total_groups,
            dw.stride(0),
            dwg.stride(0),
            dwg.stride(1),
            # -- constants
            BLOCK_G, # how many blocks in the group dim
            BLOCK_C, # how many blocks in the channel dim
            GROUP_SIZE, # how many reduction targets there are for dw
            num_warps=1,
        )
        if kernel_cache is not None:
            kernel_cache.bwd_dw = kbdw
    
    return 


class RMS_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, norm_eps=1e-6, kernel_cache=None):
        """
        """
        # Make sure that the last dimension is contiguous
        x, w = [x if x.stride(-1) == 1 else x.contiguous() for x in [x, w]]

        ctx.wshape = w.shape
        
        if w.ndim == 1:
            w = w.view(1,1, x.shape[-1]).expand(x.shape[0], -1, -1)
            reduce_w = True
        elif w.shape[0] == 1:
            w = w.expand(x.shape[0], -1, -1) 
            reduce_w = True
        else:
            reduce_w = False
            
        ox, norm_buff = _rms_fwd(x, w, norm_eps=norm_eps,
                                kernel_cache=kernel_cache)
        # Save the cache reference for the backward pass
        ctx.kernel_cache = kernel_cache
        # save the state for backward pass
        ctx.save_for_backward(x, w, norm_buff)
        ctx.norm_eps = norm_eps
        ctx.reduce_w = reduce_w
        return ox

    @staticmethod
    def backward(ctx, dox):

        dox = dox if dox.stride(-1) == 1 else dox.contiguous()
        
        x, w, norm_buff = ctx.saved_tensors
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dx = torch.empty_like(x)
            dw = torch.empty_like(w)
            _rms_bwd(
                dox,
                x,
                w,
                norm_buff,
                dx, 
                dw,
                kernel_cache=ctx.kernel_cache,
            )


        if ctx.reduce_w:
            dw = dw.sum(dim=0)
        return dx, dw.view(ctx.wshape), None, None


def triton_rms_func_fn(x, w, norm_eps=1e-6, kernel_cache=None):
    return RMS_Func.apply(x, w, norm_eps, kernel_cache)
    
    
    
