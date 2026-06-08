import torch
from torch import nn
import triton
import triton.language as tl
from torch.cuda.nvtx import range_push,range_pop
from flash_attn import flash_attn_varlen_func, flash_attn_with_kvcache
from swiftcache.utils.context import get_context
import time
@triton.jit
def fix_block_id_kernel(
    block_table:torch.Tensor, #[num_seq,max_len]
    block_id_to_fix:torch.Tensor, #[len(block_id_to_fix), 2]
    block_table_stride,
    local_num_blocks:tl.constexpr,
):
    idx = tl.program_id(0)
    seq_idx = tl.load(block_id_to_fix + idx*2)
    block_idx = tl.load(block_id_to_fix + idx*2 + 1)
    block_id = tl.load(block_table + seq_idx*block_table_stride + block_idx)
    block_id += local_num_blocks
    tl.store(block_table + seq_idx*block_table_stride + block_idx, block_id)

def fix_block_id(block_table:torch.Tensor, block_id_to_fix:torch.Tensor, local_num_blocks):
    fix_block_id_kernel[(block_id_to_fix.shape[0],)](block_table, block_id_to_fix, block_table.stride(0),local_num_blocks)

@triton.jit
def elastic_kvcache_adjust_block_id_kernel(
    block_table:torch.Tensor,
    new_block_table:torch.Tensor,
    layer_id,
    num_blocks,
    num_hidden_layers:tl.constexpr,
    TILE_SIZE: tl.constexpr,
):

    tile_idx = tl.program_id(0)
    offset = tile_idx*TILE_SIZE + tl.arange(0,TILE_SIZE)
    mask = offset < num_blocks
    blocks = tl.load(block_table + offset,mask = mask)
    blocks = blocks * num_hidden_layers + layer_id
    tl.store(new_block_table + offset,blocks,mask = mask)

def elastic_kvcache_adjust_block_id(
    block_table:torch.Tensor,
    layer_id:int,
    num_hidden_layers:int,
):
    TILE_SIZE = 1024
    N = block_table.numel()
    new_block_table = torch.empty_like(block_table)
    elastic_kvcache_adjust_block_id_kernel[((N + TILE_SIZE - 1)//TILE_SIZE,)](block_table, new_block_table,layer_id,N,num_hidden_layers,TILE_SIZE)
    return new_block_table

@triton.jit
def write_kvcache_with_slot_fix_kernel(
    key_ptr,
    key_stride, #self.num_heads * self.head_dim
    value_ptr,
    value_stride, #self.num_heads * self.head_dim
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    layer_id,
    layer_elem: tl.constexpr,
    boundary: tl.constexpr,
    D: tl.constexpr,
    D_next_power_of_2: tl.constexpr,
):
    idx = tl.program_id(0)
    mask = tl.arange(0, D_next_power_of_2) < D
    key_offsets = idx * key_stride + tl.arange(0, D_next_power_of_2)
    value_offsets = idx * value_stride + tl.arange(0, D_next_power_of_2)
    key = tl.load(key_ptr + key_offsets,mask = mask)
    value = tl.load(value_ptr + value_offsets,mask = mask)
    slot = tl.load(slot_mapping_ptr + idx)
    # local block 需要矫正 block_id
    if slot >= boundary:
        slot += layer_id * layer_elem
    # slot = 2654784
    slot_i64 = slot.to(tl.int64)
    cache_offsets = slot_i64 * D + tl.arange(0, D_next_power_of_2)
    # print('a',slot * D)
    tl.store(k_cache_ptr + cache_offsets, key,mask = mask)
    tl.store(v_cache_ptr + cache_offsets, value,mask = mask)


def write_kvcache_with_slot_fix(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor,layer_id:int,layer_elem:int, boundary):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # print(f' k_cache.stride(1):{ k_cache.stride(1)},D:{D}')
    D_next_power_of_2 = triton.next_power_of_2(D)
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    write_kvcache_with_slot_fix_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, layer_id, layer_elem, boundary, D,D_next_power_of_2)

@triton.jit
def write_kvcache_kernel(
    key_ptr,
    key_stride, #self.num_heads * self.head_dim
    value_ptr,
    value_stride, #self.num_heads * self.head_dim
    k_cache_ptr,
    v_cache_ptr,
    slot_mapping_ptr,
    slot_offset,
    D: tl.constexpr,
    D_next_power_of_2: tl.constexpr,
):
    idx = tl.program_id(0)
    mask = tl.arange(0, D_next_power_of_2) < D
    key_offsets = idx * key_stride + tl.arange(0, D_next_power_of_2)
    value_offsets = idx * value_stride + tl.arange(0, D_next_power_of_2)
    key = tl.load(key_ptr + key_offsets,mask = mask)
    value = tl.load(value_ptr + value_offsets,mask = mask)
    slot = tl.load(slot_mapping_ptr + idx).to(tl.int64)+slot_offset
    cache_offsets = slot * D + tl.arange(0, D_next_power_of_2)
    tl.store(k_cache_ptr + cache_offsets, key,mask = mask)
    tl.store(v_cache_ptr + cache_offsets, value,mask = mask)


def write_kvcache(key: torch.Tensor, value: torch.Tensor, k_cache: torch.Tensor, v_cache: torch.Tensor, slot_mapping: torch.Tensor,slot_offset:int):
    N, num_heads, head_dim = key.shape
    D = num_heads * head_dim
    # print(f' k_cache.stride(1):{ k_cache.stride(1)},D:{D}')
    D_next_power_of_2 = triton.next_power_of_2(D)
    assert key.stride(-1) == 1 and value.stride(-1) == 1
    assert key.stride(1) == head_dim and value.stride(1) == head_dim
    assert k_cache.stride(1) == D and v_cache.stride(1) == D
    assert slot_mapping.numel() == N
    write_kvcache_kernel[(N,)](key, key.stride(0), value, value.stride(0), k_cache, v_cache, slot_mapping, slot_offset, D,D_next_power_of_2)

class Attention2(nn.Module):

    def __init__(
        self,
        layer_id,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])
        #[num_kvcache_blocks, self.block_size, num_kv_heads, head_dim]

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
                # 恢复kv cache
            if context.controller.need_load:
                # print("xxxx")
                with torch.cuda.stream(context.transfer_stream):
                    context.transfer_stream.wait_event(context.attn_event)
                    self.load_kvcache(context,self.layer_id)
                    context.h2d_finished_event.record()
    
                if context.controller.kv_cache_initialized:
                    context.default_stream.wait_event(context.h2d_finished_event)
            # write_kvcache(k, v, k_cache, v_cache, context.slot_mapping)
            # print(self.layer_id, context.layer_elem, context.boundary,context.slot_mapping)
            write_kvcache_with_slot_fix(k, v, k_cache, v_cache, context.slot_mapping,self.layer_id, context.layer_elem, context.boundary)
            context.write_event.record()


                    
        if context.is_prefill:
            if context.block_tables is not None:    # prefix cache
                k, v = k_cache, v_cache
                if self.layer_id > 0:
                    fix_block_id(context.block_tables, context.block_id_to_fix, context.local_num_blocks)
            range_push('attention')
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=context.block_tables)
            context.attn_event.record()
            range_pop()

        else:    # decode
            if self.layer_id > 0:
                fix_block_id(context.block_tables, context.block_id_to_fix, context.local_num_blocks)
            # print(context.block_tables, context.block_id_to_fix)
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=context.block_tables, 
                                        softmax_scale=self.scale, causal=True)
            context.attn_event.record()
        
        if context.controller is not None and context.controller.kv_cache_initialized and context.controller.need_store:
            with torch.cuda.stream(context.transfer_stream):
                
                context.transfer_stream.wait_event(context.write_event)
                self.store_kvcache(context,self.layer_id)

        o = o.view(-1, self.num_heads * self.head_dim)
        return o

    def store_kvcache(self,context,layer_id:int):
        c = context.controller
        c.store_kvcache(self.k_cache,self.v_cache,layer_id)

    def load_kvcache(self,context,layer_id:int):
        c = context.controller
        c.load_kvcache(self.k_cache,self.v_cache,layer_id)

    def wait_last_event(self,context):
        context.controller.wait_last_event()



class Attention(nn.Module):

    def __init__(
        self,
        layer_id,
        num_heads,
        head_dim,
        scale,
        num_kv_heads,
    ):
        super().__init__()
        self.layer_id = layer_id
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.scale = scale
        self.num_kv_heads = num_kv_heads
        self.k_cache = self.v_cache = torch.tensor([])

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor):
        o: torch.Tensor
        q = q.view(-1, self.num_heads, self.head_dim)
        k = k.view(-1, self.num_kv_heads, self.head_dim)
        v = v.view(-1, self.num_kv_heads, self.head_dim)
        context = get_context()
        k_cache, v_cache = self.k_cache, self.v_cache
        if k_cache.numel() and v_cache.numel():
            # if self.layer_id==0:
            #     if context.cpu_kv_cache_manager is not None:
            #         # 恢复kv cache
            #             with torch.cuda.stream(context.transfer_stream):
            #                 self.load_kvcache(context,self.layer_id)
            #                 h2d_finished_event = torch.cuda.Event()
            #                 h2d_finished_event.record()
            #                 context.h2d_finished_event = h2d_finished_event
   
            # if context.cpu_kv_cache_manager is not None: 
            #     torch.cuda.default_stream().wait_event(context.h2d_finished_event)


            
            write_kvcache(k, v, k_cache, v_cache, context.slot_mapping,self.layer_id*context.block_size)
            # write_event = torch.cuda.Event()
            # write_event.record()

            # if self.layer_id == 13:
            #     print(k_cache[3])
                    
        if context.is_prefill:
            new_block_tables = None
            if context.block_tables is not None:    # prefix cache
                new_block_tables = elastic_kvcache_adjust_block_id(context.block_tables,self.layer_id, context.num_hidden_layers)
                k, v = k_cache, v_cache
            range_push('attention')
            o = flash_attn_varlen_func(q, k, v,
                                       max_seqlen_q=context.max_seqlen_q, cu_seqlens_q=context.cu_seqlens_q,
                                       max_seqlen_k=context.max_seqlen_k, cu_seqlens_k=context.cu_seqlens_k,
                                       softmax_scale=self.scale, causal=True, block_table=new_block_tables)
            # attn_event = torch.cuda.Event()
            # attn_event.record()
            # range_pop()

        else:    # decode
            new_block_tables = elastic_kvcache_adjust_block_id(context.block_tables,self.layer_id, context.num_hidden_layers)
            o = flash_attn_with_kvcache(q.unsqueeze(1), k_cache, v_cache,
                                        cache_seqlens=context.context_lens, block_table=new_block_tables, 
                                        softmax_scale=self.scale, causal=True)
            # attn_event = torch.cuda.Event()
            # attn_event.record()

        # if context.cpu_kv_cache_manager is not None:
        #     with torch.cuda.stream(context.transfer_stream):
        #         context.transfer_stream.wait_event(write_event)
        #         # write_event.synchronize()
        #         self.store_kvcache(context,self.layer_id)
        # if context.cpu_kv_cache_manager is not None and self.layer_id < context.cpu_kv_cache_manager.num_hidden_layers - 1:
            
        #         # 恢复下一次层kv cache
        #             with torch.cuda.stream(context.transfer_stream):
        #                 context.transfer_stream.wait_event(attn_event)
        #                 # attn_event.synchronize()
        #                 self.load_kvcache(context,self.layer_id+1)
        #                 h2d_finished_event = torch.cuda.Event()
        #                 h2d_finished_event.record()
        #                 context.h2d_finished_event = h2d_finished_event
        o = o.view(-1, self.num_heads * self.head_dim)
        return o

    def store_kvcache(self,context,layer_id:int):
        context.cpu_kv_cache_manager.store_kvcache(context.cpu_kv_cache_manager.non_cached_block_table_grouped, self.k_cache,'k',layer_id)
        context.cpu_kv_cache_manager.store_kvcache(context.cpu_kv_cache_manager.non_cached_block_table_grouped, self.v_cache,'v',layer_id)

    def load_kvcache(self,context,layer_id:int):
        context.cpu_kv_cache_manager.load_kvcache(context.cpu_kv_cache_manager.cached_block_table_grouped, self.k_cache,'k',layer_id)
        context.cpu_kv_cache_manager.load_kvcache(context.cpu_kv_cache_manager.cached_block_table_grouped, self.v_cache,'v',layer_id)
    
