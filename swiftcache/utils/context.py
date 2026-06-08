from dataclasses import dataclass, field
import torch

@dataclass
class Context:
    is_prefill: bool = False
    cu_seqlens_q: torch.Tensor | None = None
    cu_seqlens_k: torch.Tensor | None = None
    max_seqlen_q: int = 0
    max_seqlen_k: int = 0
    slot_mapping: torch.Tensor | None = None
    context_lens: torch.Tensor | None = None
    block_tables: torch.Tensor | None = None
    block_id_to_fix: torch.Tensor | None = None
    cpu_kv_cache_manager = None
    controller = None
    num_hidden_layers:int = 0
    layer_elem:int = 0
    boundary: int = 0
    default_stream = torch.cuda.default_stream()
    transfer_stream: torch.cuda.Stream = field(default_factory=torch.cuda.Stream)
    # transfer_stream: torch.cuda.Stream = torch.cuda.default_stream()
    h2d_finished_event: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    attn_event: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    write_event: torch.cuda.Event = field(default_factory=torch.cuda.Event)
    role:str = 'master'
    external_num_blocks:int = 0
    local_num_blocks: int = 0
    block_size:int  = 256
    temp_k = None
    temp_v = None
    tp_group = None
    first_rank = 0

# 全局唯一实例，在模块加载时创建一次
_CONTEXT = Context()

def get_context():
    return _CONTEXT

def set_context(is_prefill, cu_seqlens_q=None, cu_seqlens_k=None, max_seqlen_q=0, max_seqlen_k=0,
                slot_mapping=None, context_lens=None, block_tables=None, block_id_to_fix = None, cpu_kv_cache_manager=None):
    ctx = _CONTEXT  # 不重新实例化，修改已有对象
    ctx.is_prefill = is_prefill
    ctx.cu_seqlens_q = cu_seqlens_q
    ctx.cu_seqlens_k = cu_seqlens_k
    ctx.max_seqlen_q = max_seqlen_q
    ctx.max_seqlen_k = max_seqlen_k
    ctx.slot_mapping = slot_mapping
    ctx.context_lens = context_lens
    ctx.block_tables = block_tables
    ctx.cpu_kv_cache_manager = cpu_kv_cache_manager
    ctx.block_id_to_fix = block_id_to_fix

def reset_context():
    ctx = _CONTEXT
    ctx.is_prefill = False
    ctx.cu_seqlens_q = None
    ctx.cu_seqlens_k = None
    ctx.max_seqlen_q = 0
    ctx.max_seqlen_k = 0
    ctx.slot_mapping = None
    ctx.context_lens = None
    ctx.block_tables = None
    ctx.cpu_kv_cache_manager = None
    ctx.controller = None
    # ctx.temp_k = None
    # ctx.temp_v = None
    # ctx.tp_group = None

def set_tp_group(tp_group,first_rank):
    _CONTEXT.tp_group = tp_group
    _CONTEXT.first_rank = first_rank

def set_external_kvcache(controller):
    _CONTEXT.controller = controller

def set_role(role:str):
    _CONTEXT.role = role

def set_external_local_num_blocks(external_num_blocks, local_num_blocks):
    _CONTEXT.external_num_blocks = external_num_blocks
    _CONTEXT.local_num_blocks = local_num_blocks

