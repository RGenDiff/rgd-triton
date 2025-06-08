import torch
import triton
import triton.language as tl


@triton.jit
def _fused_rms_kernel_fwd(
        # -- inputs --
        x1_ptr, 
        x2_ptr, 
        w1_ptr,
        w2_ptr,
        # -- outputs --
        x1o_ptr,
        x2o_ptr,
        # -- norm buffers --
        norm_ptr,  # pointer to the rms
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
        stride_w1b,
        stride_w2b,
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
        sc_ptr = w1_ptr + bidx*stride_w1b
        rd_ptr = x1_ptr + bidx*stride_x1b + seq_offsets*stride_x1n
        wr_ptr = x1o_ptr + bidx*stride_ox1b + seq_offsets*stride_ox1n
        
    else:
        # Processing k.
        local_block = block_idx - split_idx
        total_seq = seq_len_x2
        chan_dim = DIM_X2
        seq_offsets = seq_offsets + local_block * BLOCK_N
        sc_ptr = w2_ptr + bidx*stride_w2b
        rd_ptr = x2_ptr + bidx*stride_x2b +  seq_offsets*stride_x2n
        wr_ptr = x2o_ptr + bidx*stride_ox2b + seq_offsets*stride_ox2n

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
def _fused_rms_kernel_bwd(
        # -- inputs --
        x1_ptr,
        x2_ptr,
        w1_ptr,
        w2_ptr,
        dox1_ptr, 
        dox2_ptr,         
        # -- norm buffers --
        norm_ptr,  # pointer to the mean
        # -- outputs --
        dx1_ptr,
        dx2_ptr,
        dw1g_ptr,
        dw2g_ptr,
        lock_ptr,
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
        stride_w1b,
        stride_w2b,
        stride_dox1b,
        stride_dox1n,
        stride_dox2b,
        stride_dox2n,
        stride_dx1b,
        stride_dx1n,
        stride_dx2b,
        stride_dx2n,
        stride_dw1gb,
        stride_dw1gn,
        stride_dw2gb,
        stride_dw2gn,
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
    if block_idx < split_idx:
        # Processing q.
        total_seq = seq_len_x1
        seq_offsets = seq_shift
        sc_ptr = w1_ptr + bidx*stride_w1b
        group_id = (block_idx)%GROUP_SIZE
        dsc_ptr = dw1g_ptr + bidx*stride_dw1gb + group_id*stride_dw1gn
        lock_ptr = lock_ptr + group_id
        chan_dim = dim_x1
        rd_ptr = x1_ptr + bidx*stride_x1b + seq_offsets*stride_x1n
        drd_ptr = dox1_ptr + bidx*stride_dox1b + seq_offsets*stride_dox1n
        wr_ptr = dx1_ptr + bidx*stride_dx1b + seq_offsets*stride_dx1n
        
    else:
        # Processing k.
        local_block = block_idx - split_idx
        total_seq = seq_len_x2
        seq_offsets = seq_offsets + local_block * BLOCK_N
        sc_ptr = w2_ptr + bidx*stride_w2b
        group_id = (local_block)%GROUP_SIZE
        dsc_ptr = dw2g_ptr + bidx*stride_dw2gb + group_id*stride_dw2gn
        lock_ptr = lock_ptr + group_id + GROUP_SIZE
        chan_dim = dim_x2
        rd_ptr = x2_ptr + bidx*stride_x2b  + seq_offsets*stride_x2n
        drd_ptr = dox2_ptr + bidx*stride_dox2b + seq_offsets*stride_dox2n
        wr_ptr = dx2_ptr + bidx*stride_dx2b  + seq_offsets*stride_dx2n

    # in all cases, compute the count pointer by an offset of 2x group size for 2 and 3
    count_ptr = lock_ptr + 2*GROUP_SIZE

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
def _fused_dw_kernel_bwd(
                dw1g_ptr,
                dw2g_ptr,
                dw1_ptr,
                dw2_ptr,
                # -- parameters
                dim_x1,
                dim_x2,
                total_groups1,
                total_groups2,
                stride_dw1b,
                stride_dw2b,
                stride_dw1gb,
                stride_dw1gn,
                stride_dw2gb,
                stride_dw2gn,
                # -- constants
                BLOCK_G: tl.constexpr, # how many blocks in the group dim
                BLOCK_C: tl.constexpr, # how many blocks in the channel dim
                GROUP_SIZE: tl.constexpr, # how many reduction targets there are for dw
                ):
    # the pid will be in [blocks_c, batch, 2]
    block_cidx = tl.program_id(0)
    bidx = tl.program_id(1)
    tensor_id = tl.program_id(2)

    if tensor_id == 0:
        chan_dim = dim_x1
        total_groups = total_groups1
        rd_stride = stride_dw1gn
        rd_ptr = dw1g_ptr + bidx*stride_dw1gb
        wr_ptr = dw1_ptr + bidx*stride_dw1b
    else:
        chan_dim = dim_x2
        total_groups = total_groups2
        rd_stride = stride_dw2gn
        rd_ptr = dw2g_ptr  + bidx*stride_dw2gb
        wr_ptr = dw2_ptr  + bidx*stride_dw2b       

    
    
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
        
    
        
    

def _fused_rms_fwd(x1, x2, w1, w2, norm_eps=1e-6, kernel_cache=None):

    assert x1.ndim == x2.ndim == w1.ndim == w2.ndim == 3
    assert x1.dtype == x2.dtype, "All tensors must have the same type"
    assert x1.dtype in [torch.float16, torch.bfloat16, torch.float32], "Only support fp16, bf16, and fp32"
    assert x1.is_cuda and x2.is_cuda and w1.is_cuda and w2.is_cuda
    assert x1.stride(-1) == x2.stride(-1) == w1.stride(-1) == w2.stride(-1) == 1
    assert w1.shape[1] == w2.shape[1] == 1 # don't support sequence based norms
    assert x1.shape[0] == x2.shape[0] == w1.shape[0] == w1.shape[0]
    assert w1.shape[-1] == x1.shape[2]
    assert w2.shape[-1] == x2.shape[2]

    
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
    norm_buff = torch.empty((bx1, buff_size), device=x1.device, dtype=torch.float32)
    
    grid = lambda META: (blocks_x1 + blocks_x2, bx1, 1) # grid must be 3D for caching
    
    if (kernel_cache is not None) and (kernel_cache.fwd is not None):
        grid_tuple = grid({})
        kernel_cache.fwd[grid_tuple](
            # -- inputs --
            x1, 
            x2, 
            w1,
            w2,
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
            w1.stride(0),
            w2.stride(0),
            x1o.stride(0),
            x1o.stride(1),
            x2o.stride(0),
            x2o.stride(1),
            norm_buff.stride(0), #batch
        )
    else:
        kfwd = _fused_rms_kernel_fwd[grid](
            # -- inputs --
            x1, 
            x2, 
            w1,
            w2,
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
            w1.stride(0),
            w2.stride(0),
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


def _fused_rms_bwd(dox1, dox2, x1, x2, w1, w2, norm_buff, dx1, dx2, dw1, dw2, norm_eps=1e-6, kernel_cache=None):

    assert x1.stride(-1) == x2.stride(-1) == dox1.stride(-1) == dox2.stride(-1) == 1
    assert w1.stride(-1) == w2.stride(-1) == 1
    assert dw1.stride(-1) == dw2.stride(-1) == dx1.stride(-1) == dx2.stride(-1) == 1
    
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

    # have to compute the total groups from the BLOCK_N size and block_x1, blocks_x2

    num_programs = bx1*total_blocks
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
    total_groups1 = blocks_x1%GROUP_SIZE if blocks_x1 < GROUP_SIZE else GROUP_SIZE
    total_groups2 = blocks_x2%GROUP_SIZE if blocks_x2 < GROUP_SIZE else GROUP_SIZE
    total_passes1 = triton.cdiv(blocks_x1, GROUP_SIZE)
    total_passes2 = triton.cdiv(blocks_x2, GROUP_SIZE)
    

    # allocate the dw array
    dw1g = torch.empty((bx1, GROUP_SIZE, dim_x1), dtype=torch.float32, device=x1.device)
    dw2g = torch.empty((bx1, GROUP_SIZE, dim_x2), dtype=torch.float32, device=x1.device)
    locks = torch.zeros((bx1, blocks_k, 4*GROUP_SIZE), dtype=torch.int32, device=x1.device)

    
    grid = lambda META: (total_blocks, bx1, blocks_k)  # noqa

    if (kernel_cache is not None) and (kernel_cache.bwd is not None):
        grid_tuple = grid({})
        kernel_cache.bwd[grid_tuple](
            # -- inputs --
            x1,
            x2,
            w1,
            w2,
            dox1, 
            dox2, 
            # -- norm buffers --
            norm_buff,  # pointer to the mean
            # -- outputs --
            dx1,
            dx2,
            dw1g,
            dw2g,
            locks,
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
            w1.stride(0),
            w2.stride(0),
            dox1.stride(0), # batch
            dox1.stride(1), 
            dox2.stride(0),
            dox2.stride(1),
            dx1.stride(0),
            dx1.stride(1),
            dx2.stride(0),
            dx2.stride(1),
            dw1g.stride(0),
            dw1g.stride(1),
            dw2g.stride(0),
            dw2g.stride(1),
            locks.stride(0),
            locks.stride(1),
            norm_buff.stride(0), # batch
        )
    else:
        kbwd = _fused_rms_kernel_bwd[grid](
            # -- inputs --
            x1,
            x2,
            w1,
            w2,
            dox1, 
            dox2, 
            # -- norm buffers --
            norm_buff,  # pointer to the mean
            # -- outputs --
            dx1,
            dx2,
            dw1g,
            dw2g,
            locks,
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
            w1.stride(0),
            w2.stride(0),
            dox1.stride(0), # batch
            dox1.stride(1), 
            dox2.stride(0),
            dox2.stride(1),
            dx1.stride(0),
            dx1.stride(1),
            dx2.stride(0),
            dx2.stride(1),
            dw1g.stride(0),
            dw1g.stride(1),
            dw2g.stride(0),
            dw2g.stride(1),
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
    grid = lambda META: (total_c_blocks, bx1, 2)  

    if (kernel_cache is not None) and (kernel_cache.bwd_dw is not None):
        grid_tuple = grid({})
        kernel_cache.bwd_dw[grid_tuple](
            dw1g,
            dw2g,
            dw1,
            dw2,
            # -- parameters
            dim_x1,
            dim_x2,
            total_groups1,
            total_groups2,
            dw1.stride(0),
            dw2.stride(0),
            dw1g.stride(0),
            dw1g.stride(1),
            dw2g.stride(0),
            dw2g.stride(1),    
        )
    else:
        kbdw = _fused_dw_kernel_bwd[grid](
            dw1g,
            dw2g,
            dw1,
            dw2,
            # -- parameters
            dim_x1,
            dim_x2,
            total_groups1,
            total_groups2,
            dw1.stride(0),
            dw2.stride(0),
            dw1g.stride(0),
            dw1g.stride(1),
            dw2g.stride(0),
            dw2g.stride(1),
            # -- constants
            BLOCK_G, # how many blocks in the group dim
            BLOCK_C, # how many blocks in the channel dim
            GROUP_SIZE, # how many reduction targets there are for dw
            num_warps=1,
        )
        if kernel_cache is not None:
            kernel_cache.bwd_dw = kbdw
    
    return 


class Fused_RMS_Func(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x1, x2, w1, w2, norm_eps=1e-6, kernel_cache=None):
        """
        """
        # Make sure that the last dimension is contiguous
        x1, x2, w1, w2 = [x if x.stride(-1) == 1 else x.contiguous() for x in [x1, x2, w1, w2]]

        ctx.w1shape = w1.shape
        ctx.w2shape = w2.shape
        
        if w1.ndim == 1:
            w1 = w1.view(1,1, x1.shape[-1]).expand(x1.shape[0], -1, -1)
            reduce_w1 = True
        elif w1.shape[0] == 1:
            w1 = w1.expand(x1.shape[0], -1, -1) 
            reduce_w1 = True
        else:
            reduce_w1 = False
            
        if w2.ndim == 1:
            w2 = w2.view(1,1, x2.shape[-1]).expand(x2.shape[0], -1, -1) 
            reduce_w2 = True
        elif w2.shape[0] == 1:
            w2 = w2.expand(x2.shape[0], -1, -1) 
            reduce_w2 = True
        else:
            reduce_w2 = False
        
        ox1, ox2, norm_buff = _fused_rms_fwd(x1, x2, w1, w2, norm_eps=norm_eps,
                                            kernel_cache=kernel_cache)
        # Save the cache reference for the backward pass
        ctx.kernel_cache = kernel_cache
        # save the state for backward pass
        ctx.save_for_backward(x1, x2, w1, w2, norm_buff)
        ctx.norm_eps = norm_eps
        ctx.reduce_w1 = reduce_w1
        ctx.reduce_w2 = reduce_w2
        return ox1, ox2

    @staticmethod
    def backward(ctx, dox1, dox2):

        dox1, dox2 = [x if x.stride(-1) == 1 else x.contiguous() for x in [dox1, dox2]]
        
        x1, x2, w1, w2, norm_buff = ctx.saved_tensors
        # Triton's autotune causes the Tensor._version to change, and so Pytorch autograd
        # does a memcpy. To avoid this we run in inference_mode, which doesn't track the version.
        with torch.inference_mode():
            dx1 = torch.empty_like(x1)
            dx2 = torch.empty_like(x2)
            dw1 = torch.empty_like(w1)
            dw2 = torch.empty_like(w2)
            _fused_rms_bwd(
                dox1,
                dox2,
                x1,
                x2,
                w1,
                w2,
                norm_buff,
                dx1, 
                dx2, 
                dw1,
                dw2,
                kernel_cache=ctx.kernel_cache,
            )


        if ctx.reduce_w1:
            dw1 = dw1.sum(dim=0)
        if ctx.reduce_w2:
            dw2 = dw2.sum(dim=0)
        return dx1.clone(), dx2.clone(), dw1.view(ctx.w1shape), dw2.view(ctx.w2shape), None, None


def triton_fused_rms_func_fn(x1,x2, w1, w2, norm_eps=1e-6, kernel_cache=None):
    return Fused_RMS_Func.apply(x1, x2, w1, w2, norm_eps, kernel_cache)
    
    
    
