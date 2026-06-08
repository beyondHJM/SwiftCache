from copy import copy
from enum import Enum, auto
from itertools import count
from typing import Optional
import time

import torch
from swiftcache.sampling_params import SamplingParams
from swiftcache.utils.io import load_pil_images
from swiftcache.global_config import global_config


class SequenceStatus(Enum):
    WAITING = auto()
    RUNNING = auto()
    FINISHED = auto()

class VisionStatus(Enum):
    WAITING = auto()
    ENCODING = auto()
    COMPLETED = auto()

class RequestType(Enum):
    LANGUAGE_ONLY = auto()
    VISION_LANGUAGE = auto()

class Sequence:
    block_size = global_config.get('kvcache_block_size')
    counter = count()

    def __init__(self, token_ids: list[int], sampling_params = SamplingParams(), input_embeds: Optional[torch.Tensor] = None,seq_id = None):
        self.seq_id = seq_id if seq_id is not None else next(Sequence.counter)
        self.status = SequenceStatus.WAITING
        self.input_embeds = None

        if input_embeds is not None:
            self.input_embeds = input_embeds
        self.token_ids = copy(token_ids)
        self.last_token = self.token_ids[-1]
        self.num_tokens = len(self.token_ids)
        self.num_prompt_tokens = len(self.token_ids)
        self.num_cached_tokens = 0
        self.block_table = []
        self.local_block_table = []
        self.local_cached_blocks = []
        self.local_uncached_blocks = []
        self.num_blocks_per_slave = [] #len(self.num_blocks_per_slave) = len(slave_list) 用来计算local_block_id的偏移量的
        self.cum_blocks_per_slave = [] #len = len(slave_list)+1 
        self.block_belong_to_slave = [] #len = len(self.block_table)
        self.temperature = sampling_params.temperature
        self.max_tokens = sampling_params.max_tokens
        self.ignore_eos = sampling_params.ignore_eos
        self.time_usage = []

        #used for online inference
        self.request_id = None
        self.finised_event = None
        self.extra_info = {}

    def __len__(self):
        return self.num_tokens

    def __getitem__(self, key):
        return self.token_ids[key]

    @property
    def is_finished(self):
        return self.status == SequenceStatus.FINISHED

    @property
    def num_completion_tokens(self):
        return self.num_tokens - self.num_prompt_tokens

    @property
    def prompt_token_ids(self):
        return self.token_ids[:self.num_prompt_tokens]

# 这行有问题吗，明天看一下–
    @property
    def completion_token_ids(self):
        return self.token_ids[self.num_prompt_tokens:]

    @property
    def num_cached_blocks(self):
        return self.num_cached_tokens // self.block_size

    @property
    def num_blocks(self):
        return (self.num_tokens + self.block_size - 1) // self.block_size

    @property
    def last_block_num_tokens(self):
        return self.num_tokens - (self.num_blocks - 1) * self.block_size

    def block(self, i):
        assert 0 <= i < self.num_blocks
        return self.token_ids[i*self.block_size: (i+1)*self.block_size]

    def append_token(self, token_id: int):
        self.token_ids.append(token_id)
        self.last_token = token_id
        self.num_tokens += 1
    
    def time_usage_append(self,t:float):
        self.time_usage.append(t)

    def __getstate__(self):
        return (self.seq_id, self.input_embeds, self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table,
                self.token_ids if self.num_completion_tokens == 0 else self.last_token)

    def __setstate__(self, state):
        self.seq_id, self.input_embeds, self.num_tokens, self.num_prompt_tokens, self.num_cached_tokens, self.block_table = state[:-1]
        if self.num_completion_tokens == 0:
            self.token_ids = state[-1]
        else:
            self.last_token = state[-1]

class VisionSequence:
    counter = count()
    def __init__(self, raw_input=None, sampling_params = SamplingParams(), status:VisionStatus = VisionStatus.WAITING, prepare_inputs = None):
        # assert isinstance(raw_input, dict), f"raw_input must be dict, got {type(raw_input).__name__}"
        self.seq_id = next(Sequence.counter)
        if status is None or status is VisionStatus.WAITING:
            self.status = VisionStatus.WAITING
            self.conversation = raw_input
            self.pil_images = load_pil_images(raw_input)
            self.prepare_inputs: Optional[Any] = None 

        if status is VisionStatus.ENCODING:
            self.status = VisionStatus.ENCODING
            self.prepare_inputs = prepare_inputs

        self.time_usage=[]
        self.sampling_params = sampling_params


def sequence_cpp_to_py(seqs_cpp):
    seqs = []
    for seq_cpp in seqs_cpp:
        sampling_params = SamplingParams(seq_cpp.temperature,seq_cpp.max_tokens,seq_cpp.ignore_eos)
        seq = Sequence(seq_cpp.token_ids,sampling_params,seq_id = seq_cpp.seq_id)
        seq.block_table = seq_cpp.block_table
        seq.num_cached_tokens = seq_cpp.num_cached_tokens
        seqs.append(seq)
    return seqs