import torch
from torch import nn
import triton
import triton.language as tl
# from vllm import _custom_ops as ops
import time
@triton.jit
def _fwd_rmsnorm(
    input_and_output: torch.Tensor,	# [num_tokens, hidden_size], contiguous
    weight: torch.Tensor,			# [hidden_size]
    eps: float,
    hidden_size: tl.constexpr,
    N: tl.constexpr,
):
    # grid shape: [num_tokens]
    my_token_id = tl.program_id(0)
    input_and_output += my_token_id * hidden_size	# [hidden_size]
    offs = tl.arange(0, N)
    mask = offs < hidden_size
    x = tl.load(input_and_output+offs,mask = mask)
    x = x.to(tl.float32)
    variance = tl.sum(x*x, axis=0) / hidden_size
    rstd = 1 / tl.sqrt(variance + eps)

    w = tl.load(weight+offs,mask = mask).to(tl.float32)
    x = x*rstd*w
    tl.store(input_and_output+offs, x.to(tl.bfloat16),mask = mask)

def rmsnorm_inplace(
    input_and_output: torch.Tensor,	# [num_tokens, hidden_size]
    weight: torch.Tensor,
    eps: float
):
    # assert input_and_output.is_contiguous()
    assert weight.is_contiguous()
    grid = (input_and_output.shape[0], )
    N = triton.next_power_of_2 (input_and_output.shape[1])
    _fwd_rmsnorm[grid](
        input_and_output,
        weight,
        eps,
        input_and_output.shape[1],
        N
    )

@triton.jit
def _fwd_fused_add_rmsnorm(
    input_and_output: torch.Tensor,	# [num_tokens, hidden_size], contiguous
    residual_io: torch.Tensor,		# [num_tokens, hidden_size], contiguous
    weight: torch.Tensor,			# [hidden_size]
    eps: float,

    hidden_size: tl.constexpr,
    N: tl.constexpr,
    ):
    # grid shape: [num_tokens]
    my_token_id = tl.program_id(0)
    input_and_output += my_token_id * hidden_size	# [hidden_size]
    residual_io += my_token_id * hidden_size

    offs = tl.arange(0, N)
    mask = offs < hidden_size
    x = tl.load(input_and_output+offs,mask = mask)
    r = tl.load(residual_io+offs,mask = mask)
    x += r
    tl.store(residual_io+offs, x.to(tl.bfloat16),mask = mask)

    x = x.to(tl.float32)
    variance = tl.sum(x*x, axis=0) / hidden_size
    rstd = 1 / tl.sqrt(variance + eps)

    w = tl.load(weight+offs,mask = mask).to(tl.float32)
    x = x*rstd*w
    tl.store(input_and_output+offs, x.to(tl.bfloat16),mask = mask)

def fused_add_rmsnorm_inplace(
	input_and_output: torch.Tensor,	# [num_tokens, hidden_size]
	residual_io: torch.Tensor,
	weight: torch.Tensor,
	eps: float
):
    """
    Perform fused add & rmsnorm

    This function accepts input_and_output (x), residual_io (r), and weight(w)
    as inputs, set r = x+r, and x = rms_norm(x+r, w)
    """
    assert input_and_output.is_contiguous()
    assert residual_io.is_contiguous()
    assert weight.is_contiguous()
    grid = (input_and_output.shape[0], )
    N = triton.next_power_of_2 (input_and_output.shape[1])
    _fwd_fused_add_rmsnorm[grid](
        input_and_output,
        residual_io,
        weight,
        eps,
        input_and_output.shape[1],
        N
    )


class RMSNorm(nn.Module):

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    @torch.compile
    def rms_forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # fused_add_rmsnorm_inplace(x,torch.zeros_like(x),self.weight.data,self.eps)
        # x = x.contiguous()
        # rmsnorm_inplace(x,self.weight.data,self.eps)
        # return x 
        # t1 = time.perf_counter()
        orig_dtype = x.dtype
        x = x.to(torch.float32)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        # print(time.perf_counter()-t1)
        return x
        # print('x',x)
        # print('x1',x1)
        
        # out = torch.empty_like(x)
        # ops.rms_norm(
        #     out,
        #     x,
        #     self.weight.data,
        #     self.eps,
        # )
        
        # return out

    @torch.compile
    def add_rms_forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # fused_add_rmsnorm_inplace(x,residual,self.weight.data,self.eps)
        # return x,residual
        orig_dtype = x.dtype
        x = x.to(torch.float32).add_(residual.to(torch.float32))
        residual = x.to(orig_dtype)
        var = x.pow(2).mean(dim=-1, keepdim=True)
        x.mul_(torch.rsqrt(var + self.eps))
        x = x.to(orig_dtype).mul_(self.weight)
        return x, residual

    def forward(
        self,
        x: torch.Tensor,
        residual: torch.Tensor | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            return self.rms_forward(x)
        else:
            return self.add_rms_forward(x, residual)
