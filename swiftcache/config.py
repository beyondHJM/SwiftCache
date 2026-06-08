import os
from dataclasses import dataclass, field
from transformers import AutoConfig

from swiftcache.global_config import global_config
from multiprocessing.synchronize import Event
@dataclass
class ExternalKVCacheConfig:
    num_blocks_start_end:list[int] = field(default_factory=list)
    num_external_kvcache = 0


@dataclass
class Config:
    model: str
    max_num_batched_tokens: int = 81920//2
    max_num_seqs: int = 1
    max_model_len: int = 81920//2
    gpu_memory_utilization: float = 0.97
    tensor_parallel_size: int = 1
    enforce_eager: bool = True
    hf_config: AutoConfig | None = None
    eos: int = -1
    kvcache_block_size: int = global_config.get('kvcache_block_size')
    num_kvcache_blocks: int = -1
    local_num_blocks:int = -1
    rank: int = 0
    role: str = 'master'
    dist_port:int = 2334
    external_kvcache_config: ExternalKVCacheConfig = ExternalKVCacheConfig()
    master_list:list = field(default_factory = list)
    slave_list:list = field(default_factory = list)
    slave_event:dict | Event = field(default_factory = dict)
    slave_ready_event:dict | Event = field(default_factory = dict)
    tp_group : list = field(default_factory = list)
    master_minimum_scaling_block_count:int = 1
    slave_minimum_scaling_block_count:int = 1





    def __post_init__(self):
        assert os.path.isdir(self.model)
        assert self.kvcache_block_size % 256 == 0
        assert 1 <= self.tensor_parallel_size <= 8
        self.hf_config = AutoConfig.from_pretrained(self.model)
        max_position_embeddings = getattr(self.hf_config, 'max_position_embeddings', None)
        max_position_embeddings = max_position_embeddings if max_position_embeddings is not None else self.hf_config.language_config.max_position_embeddings
        self.max_model_len = min(self.max_model_len, max_position_embeddings )
        assert self.max_num_batched_tokens >= self.max_model_len

