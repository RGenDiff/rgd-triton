import torch
import torch.nn as nn

##################################
##    Cacheable Module 
##################################

# create a cachable module based on nn.Module
# this supports dynamic cache creation based on provided fields
# it also handles the creation and destruction of the cache objects
# based on the enable_cache() function
class CachableModule(nn.Module):
    def __init__(self, cache_fields=None):
        super().__init__()
        self._cache_fields = cache_fields or []
        self.cache = self._make_cache()

    def _make_cache(self):
        # Create a dynamic class with fields like .fwd, .bwd, etc.
        fields = {name: None for name in self._cache_fields}
        return type("DynamicCache", (), fields)()

    def enable_cache(self, enable=True):
        self.cache = self._make_cache() if enable else None


##################################
##    Attention Operations
##################################

class FlashAttention(CachableModule):
    def __init__(self):
        # setup the cache
        super().__init__(cache_fields=("fwd", "bwd_odot", "bwd"))

        # setup the triton function
        from .triton.flash_attn.flash_attn_triton_cached1 import flash_attn_func
        self._triton_fn = flash_attn_func

    def forward(self, q, k, v, bias=None, causal=False, softmax_scale=None):
        return self._triton_fn(q=q,
                               k=k,
                               v=v,
                               bias=bias,
                               causal=causal,
                               softmax_scale=softmax_scale,
                               kernel_cache=self.cache
                            )

##################################
##    Basic Normalization Operations
##################################

class L2Normalize(CachableModule):
    def __init__(self):
        # setup the cache
        super().__init__(cache_fields=("fwd", "bwd"))

        # setup the triton function
        from .triton.normalization.l2normalize_cached import triton_norm_func_fn
        self._triton_fn = triton_norm_func_fn

    def forward(self, x, eps=1e-6):
        return self._triton_fn(x=x,
                               norm_eps=eps,
                               kernel_cache=self.cache
                            )

class FusedL2Normalize2(CachableModule):
    def __init__(self):
        # setup the cache
        super().__init__(cache_fields=("fwd", "bwd"))

        # setup the triton function
        from .triton.normalization.fused_l2normalize_cached import triton_fused_norm_func_fn
        self._triton_fn = triton_fused_norm_func_fn

    def forward(self, x1, x2, eps=1e-6):
        return self._triton_fn(x1=x1,
                               x2=x2,
                               norm_eps=eps,
                               kernel_cache=self.cache
                            )

class FusedHeadLayerNorm(CachableModule):
    def __init__(self):
        # setup the cache
        super().__init__(cache_fields=("fwd", "bwd"))

        # setup the triton function
        from .triton.normalization.head_layernorm_cached import triton_fused_ln_func_fn
        self._triton_fn = triton_fused_ln_func_fn

    def forward(self, query, key, heads_second=True, eps=1e-6):
        return self._triton_fn(query=query,
                               key=key,
                               heads_second=heads_second,
                               norm_eps=eps,
                               kernel_cache=self.cache
                            )

##################################
##    RMS Normalization Operations with Affine
##################################


class FusedRMSNorm(CachableModule):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True,
                device=None, dtype=None,):
        # setup the cache
        super().__init__(cache_fields=("fwd", "bwd", "bwd_dw"))

        assert isinstance(normalized_shape,int), "Fused RMSNorm only supports int dims"
        self.normalized_shape = (normalized_shape,)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # setup the weights
        if self.elementwise_affine:
            factory_kwargs = {"device": device, "dtype": dtype}
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

        # setup the triton function
        from .triton.normalization.rmsnorm_cached import triton_rms_func_fn
        self._triton_fn = triton_rms_func_fn

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight)
    
    def forward(self, x, weight=None):
        
        if self.elementwise_affine:
            weight = self.weight # ignore the input
        else:
            assert weight is not None, "Must pass in a weight tensor if not using element-wise affine"
    
        return self._triton_fn(x=x, 
                               w=weight, 
                               norm_eps=self.eps,
                               kernel_cache=self.cache
                            )

class FusedRMSNorm2(CachableModule):
    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True,
                device=None, dtype=None,):
        # setup the cache
        super().__init__(cache_fields=("fwd", "bwd", "bwd_dw"))

        assert isinstance(normalized_shape, (tuple, list)) and len(normalized_shape) == 2, "Fused RMSNorm2 requires 2 int dims"
        
        self.normalized_shape1 = (normalized_shape[0],)
        self.normalized_shape2 = (normalized_shape[1],)
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        
        # setup the weights
        if self.elementwise_affine:
            factory_kwargs = {"device": device, "dtype": dtype}
            self.weight1 = nn.Parameter(
                torch.empty(self.normalized_shape1, **factory_kwargs)
            )
            self.weight2 = nn.Parameter(
                torch.empty(self.normalized_shape2, **factory_kwargs)
            )
        else:
            self.register_parameter("weight1", None)
            self.register_parameter("weight2", None)
        self.reset_parameters()

        # setup the triton function
        from .triton.normalization.fused_rmsnorm_cached import triton_fused_rms_func_fn
        self._triton_fn = triton_fused_rms_func_fn

    def reset_parameters(self) -> None:
        """
        Resets parameters based on their initialization used in __init__.
        """
        if self.elementwise_affine:
            torch.nn.init.ones_(self.weight1)
            torch.nn.init.ones_(self.weight2)
    
    def forward(self, x1, x2, weight1=None, weight2=None):
        
        if self.elementwise_affine:
            weight1 = self.weight1 # ignore the input
            weight2 = self.weight2 # ignore the input
        else:
            assert (weight1 is not None) and (weight2 is not None), "Must pass in a weight tensor if not using element-wise affine"
    
        return self._triton_fn(x1=x1,
                               x2=x2, 
                               w1=weight1, 
                               w2=weight2, 
                               norm_eps=self.eps,
                               kernel_cache=self.cache
                            )


##################################
##    RoPE Funcitons
##################################


class RoPE(CachableModule):
    def __init__(self):
        # setup the cache
        super().__init__(cache_fields=("fwd", "bwd"))

        # setup the triton function
        from .triton.rope.head_rope_cached import triton_rope_fn
        self._triton_fn = triton_rope_fn

    def forward(self, query, key, pos_emb, seq_cutoff=None, heads_second=True):
        return self._triton_fn(query=query,
                               key=key,
                               pos_emb=pos_emb,
                               seq_cutoff=seq_cutoff,
                               heads_second=heads_second,
                               kernel_cache=self.cache
                            )


class FusedLNRoPE(CachableModule):
    def __init__(self):
        # setup the cache
        super().__init__(cache_fields=("fwd", "bwd"))

        # setup the triton function
        from .triton.rope.fused_ln_rope_cached import triton_fused_ln_rope_fn
        self._triton_fn = triton_fused_ln_rope_fn

    def forward(self, query, key, pos_emb, seq_cutoff=None, heads_second=True, eps=1e-6):
        return self._triton_fn(query=query,
                               key=key,
                               pos_emb=pos_emb,
                               seq_cutoff=seq_cutoff,
                               heads_second=heads_second,
                               norm_eps=eps,
                               kernel_cache=self.cache
                            )


       
