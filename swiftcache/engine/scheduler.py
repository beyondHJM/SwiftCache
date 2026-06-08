from collections import deque
from enum import Enum, auto
from typing import Union

from swiftcache.config import Config
from swiftcache.engine.sequence import Sequence, SequenceStatus, RequestType, VisionSequence, VisionStatus
from swiftcache.engine.block_manager import create_block_manager


class SchedulingType(Enum):
    # IMAGE_PROCESSING = auto()   
    # VISION_ENCODING = auto() 
    PREFILL = "prefill" 
    DECODING = "decoding"

class Scheduler:

    def __init__(self, config: Config):
        self.cnt = 0
        self.config = config
        self.role = config.role
        self.max_num_seqs = config.max_num_seqs
        self.max_num_batched_tokens = config.max_num_batched_tokens
        self.eos = config.eos
        if config.role == 'master':
            assert config.num_kvcache_blocks - config.external_kvcache_config.num_blocks_start_end[-1] == config.local_num_blocks
        self.block_manager = create_block_manager(config)
        # self.vision_waiting: deque[Sequence] = deque()
        # self.vision_running: deque[Sequence] = deque()
        self.waiting: deque[Sequence] = deque()
        self.running: deque[Sequence] = deque()
    
    def check_kvcache_change(self):
        pass

    def is_finished(self):
        return not self.waiting and not self.running 
    # def is_finished(self):
    #     return not self.waiting and not self.running and not self.vision_waiting and not self.vision_running
    def add(self, seq: Union[Sequence, VisionSequence]):
        if isinstance(seq, Sequence):
            self.waiting.append(seq)
        # if isinstance(seq, VisionSequence):
        #     if seq.status == VisionStatus.WAITING:
        #         self.vision_waiting.append(seq)
        #     elif seq.status == VisionStatus.ENCODING:
        #         self.vision_running.append(seq)
                
    def schedule(self) -> tuple[list[Sequence], SchedulingType]:
        scheduled_seqs = {
            # 'image_process_seqs':[],
            # 'vision_encode_seqs':[],
            'prefill_seqs':[],
            'decoding_seqs':[]
        }
        #image processing and vision encoding
        # while self.vision_waiting:
        #     seq = self.vision_waiting[0]
        #     self.vision_waiting.popleft()
        #     seq.status = VisionStatus.ENCODING
        #     scheduled_seqs['image_process_seqs'].append(seq)
        #     return scheduled_seqs['image_process_seqs'],SchedulingType.IMAGE_PROCESSING

        # while self.vision_running:
        #     seq = self.vision_running[0]
        #     self.vision_running.popleft()
        #     seq.status = VisionStatus.COMPLETED
        #     scheduled_seqs['vision_encode_seqs'].append(seq)
        #     return scheduled_seqs['vision_encode_seqs'],SchedulingType.VISION_ENCODING

        if self.config.role == 'master':
            self.block_manager.free_block_ids.sync_num_free_blocks()
            self.block_manager.master_check_blocks_update()
            self.cnt +=1
            if self.cnt % 100 == 0:
                print("===---------")
                self.block_manager.print_block_info()

        # prefill
        num_seqs = 0
        num_batched_tokens = 0
        while self.waiting and num_seqs < self.max_num_seqs:
            seq = self.waiting[0]
            if num_batched_tokens + len(seq) > self.max_num_batched_tokens or not self.block_manager.can_allocate(seq):
                break
            num_seqs += 1
    
            self.block_manager.allocate(seq)
            num_batched_tokens += len(seq) - seq.num_cached_tokens
            seq.status = SequenceStatus.RUNNING
            self.waiting.popleft()
            self.running.append(seq)
            scheduled_seqs['prefill_seqs'].append(seq)
        usage_rate = self.block_manager.usage_rate()
        

        if scheduled_seqs['prefill_seqs']:
            # print(f'KV Cache 使用率为{usage_rate*100:.2f}%')
            return scheduled_seqs['prefill_seqs'], "prefill"

        # decode
        while self.running and num_seqs < self.max_num_seqs:
            seq = self.running.popleft()
            while not self.block_manager.can_append(seq):
                if self.running:
                    self.preempt(self.running.pop())
                else:
                    self.preempt(seq)
                    break
            else:
                num_seqs += 1
                self.block_manager.may_append(seq)
                scheduled_seqs['decoding_seqs'].append(seq)
        assert scheduled_seqs['decoding_seqs']
        self.running.extendleft(reversed(scheduled_seqs['decoding_seqs']))
        # print(f'KV Cache 使用率为{usage_rate*100:.2f}%')
        return scheduled_seqs['decoding_seqs'], "decoding"

    def preempt(self, seq: Sequence):
        seq.status = SequenceStatus.WAITING
        self.block_manager.deallocate(seq)
        self.waiting.appendleft(seq)

    def postprocess(self, seqs: list[Sequence], token_ids: list[int]) -> list[bool]:
        for seq, token_id in zip(seqs, token_ids):
            seq.append_token(token_id)
            if (not seq.ignore_eos and token_id == self.eos) or seq.num_completion_tokens == seq.max_tokens:
                seq.status = SequenceStatus.FINISHED
                self.block_manager.deallocate(seq)
                self.running.remove(seq)
    
    def create_add(self,prompt,sampling_params):
        seq = Sequence(prompt, sampling_params)
        self.add(seq)
