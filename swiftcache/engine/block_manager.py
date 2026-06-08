from collections import deque
import heapq
import xxhash
import numpy as np
from typing import Union, Optional
from swiftcache.global_config import global_config
from swiftcache.engine.sequence import Sequence
from swiftcache.config import Config
from swiftcache.utils.zmq_wrapper import ZMQServer, ZMQClient
import time
class Block:

    def __init__(self, block_id,compare_mode="asc",external_blocks_num = 0):
        self.block_id = block_id
        self.ref_count = 0
        self.hash = -1
        self.token_ids = []
        self.hit_count = 0
        self.prefix_position = 0
        self.prefix_depth = 0
        # self.block_hash
        self.external_blocks_num = external_blocks_num
        assert compare_mode in ['asc','desc','local_first']
        self.compare_mode = compare_mode
    def update(self, _hash: int, token_ids: list[int]):
        self.hash = _hash
        self.token_ids = token_ids

    def reset(self):
        self.ref_count = 1
        self.hash = -1
        self.token_ids = []
    
    @property
    def dist(self):
        return self.prefix_depth - self.prefix_position

    def __lt__(self, other: "Block") -> bool:
        # 规则1：hit_count 小的优先级高
        if self.hash == -1 and other.hash != -1:
            return True
        if self.hash != -1 and other.hash == -1:
            return False
        if self.hit_count != other.hit_count:
            return self.hit_count < other.hit_count
        # if self.prefix_position != other.prefix_position:
        #     return self.prefix_position > other.prefix_position
        # if self.prefix_depth != other.prefix_depth:
        #     return self.prefix_depth < other.prefix_depth
        if self.dist != other.dist:
            return self.dist < other.dist

        if self.compare_mode == 'asc':
            return self.block_id < other.block_id
        if self.compare_mode == 'desc':
            return self.block_id > other.block_id
        if self.compare_mode == 'local_first':
            if self.block_id < self.external_blocks_num and other.block_id < self.external_blocks_num :
                return self.block_id < other.block_id
            if self.block_id >= self.external_blocks_num and other.block_id < self.external_blocks_num :
                return True
            if self.block_id < self.external_blocks_num and other.block_id >= self.external_blocks_num :
                return False
            if self.block_id >= self.external_blocks_num and other.block_id >= self.external_blocks_num :
                return self.block_id < other.block_id


class FreeBlockIds:
    def __init__(self, blocks: list[Block],offset:int = 0,config:Config = None, minimum_scaling_block_count = 0):
        self.config = config
        self.blocks = blocks       # Block 对象列表
        self.offset = offset
        if config.role == 'slave':
            self.init_block_num = config.slave_minimum_scaling_block_count
            self.minimum_scaling_block_count = config.slave_minimum_scaling_block_count
            self._heap: list[Block] = list(blocks[:self.init_block_num])
        else:
            self.init_block_num = minimum_scaling_block_count
            self.minimum_scaling_block_count = minimum_scaling_block_count
            self._heap: list[Block] = list(blocks[self.init_block_num:])     # 复制所有 Block 对象
            print(f"self.init_block_num:{self.init_block_num}")
        
            # print(f'slave{config.rank}:{[block.block_id for block in self._heap[:20]]}')
        heapq.heapify(self._heap)                   # 一次性转化成最小堆

    def push(self, block_id: int):
        """将指定 block_id 加入堆"""
        block = self.blocks[block_id]
        heapq.heappush(self._heap, block)
    
    def append(self, block_id: int):
        self.push(block_id)

    def pop(self) -> Optional[int]:
        """弹出优先级最高的 block id"""
        if not self._heap:
            return None
        block = heapq.heappop(self._heap)
        return block.block_id

    def peek(self) -> Optional[int]:
        """获得堆顶元素的 block id，但不删除"""
        if not self._heap:
            return None
        return self._heap[0].block_id
    
    def remove(self, block_id: int) -> bool:
        """根据 block_id 删除堆中对应的 Block"""
        block_id += self.offset
        for i, block in enumerate(self._heap):
            if block.block_id == block_id:
                # 删除该元素
                # print(f"即将删除{block_id}")
                self._heap.pop(i)
                # 重新建堆
                heapq.heapify(self._heap)
                return True
        return False  # 没找到
    
    def remove_batch_with_global_id(self, global_block_ids:list[int]):
        global_block_ids_set = set(global_block_ids)
        self._heap = [block for block in self._heap if block.block_id not in global_block_ids_set]
        heapq.heapify(self._heap)
    
    def push_batch_with_global_id(self, block_ids):
        new_blocks = [self.blocks[b_id-self.offset] for b_id in block_ids]
        self._heap.extend(new_blocks)
        heapq.heapify(self._heap)

    def slave_scale_up_once(self):
        self._heap.extend(list(self.blocks[self.init_block_num:self.init_block_num + self.minimum_scaling_block_count]))  
        heapq.heapify(self._heap)   
        self.init_block_num += self.minimum_scaling_block_count

    def slave_scale_up_many(self, n:int):
        self._heap.extend(list(self.blocks[self.init_block_num:self.init_block_num + self.minimum_scaling_block_count * n]))  
        heapq.heapify(self._heap)   
        self.init_block_num += self.minimum_scaling_block_count * n
        print(f'current init_block_num:{self.init_block_num}')
    
    def slave_scale_up_with_specific_blocks(self,num_blocks:int):
        n = (num_blocks + self.minimum_scaling_block_count - 1) // self.minimum_scaling_block_count
        self.slave_scale_up_many(n)
    
    def master_scale_down_many(self, n:int):
        num_blocks = self.minimum_scaling_block_count * n
        global_block_ids = [block.block_id for block in self.blocks[self.init_block_num: self.init_block_num + num_blocks]]
        print(f'即将被缩减的global_block_ids:{global_block_ids}')
        t = time.perf_counter()
        self.remove_batch_with_global_id(global_block_ids)
        print(time.perf_counter()-t)
        self.init_block_num += num_blocks

    def master_scale_down_with_specific_blocks(self,num_blocks:int):
        pass
    


    @property
    def num_blocks_lent(self):
        return len(self.blocks) - self.init_block_num
        
    def __len__(self):
        return len(self._heap)
    
    def __getitem__(self, index: int):
        """支持索引访问，返回 Block 对象"""
        return self._heap[index].block_id


# def occupy_block_ids(free_block_ids: FreeBlockIds, occupied_block_ids:list[int]):
#     free_block_ids.remove_batch(occupied_block_ids)

class MultiFreeBlockIds:
    def __init__(self,blocks: list[Block], num_blocks_start_end: list[int],local_num_blocks,config:Config,minimum_scaling_block_count:int):
        # assert len(blocks) == num_blocks_start_end[-1]
        self.blocks = blocks
        self.minimum_scaling_block_count = minimum_scaling_block_count
        self.n_group = len(num_blocks_start_end)-1
        self.num_blocks_start_end = num_blocks_start_end
        self.local_num_blocks = len(blocks) - num_blocks_start_end[-1]
        self.n_blocks = len(blocks)
        self.multi_free_block_ids = []
        self.n_free_blocks = self.n_blocks
        for i in range(self.n_group):
            free_blocks_ids = FreeBlockIds(blocks[num_blocks_start_end[i]:num_blocks_start_end[i+1]],num_blocks_start_end[i],config,minimum_scaling_block_count[i])
            self.multi_free_block_ids.append(free_blocks_ids)
        self.local_free_block_ids = FreeBlockIds(blocks[num_blocks_start_end[-1]:], num_blocks_start_end[-1], config)
        self.counter = 0
        print(f'n_free_blocks init:{self.n_free_blocks}')
        self.sync_num_free_blocks()
    def get_group_idx(self, block_id):
        for i in range(self.n_group):
            if block_id >= self.num_blocks_start_end[i] and block_id < self.num_blocks_start_end[i+1]:
                return i
        
        return -1
        

    def append(self,block_id):
        self.n_free_blocks += 1
        group_id = self.get_group_idx(block_id)
        block_id_in_group = block_id - self.num_blocks_start_end[group_id] # -offset
        if group_id != -1:
            self.multi_free_block_ids[group_id].append(block_id_in_group)
        else:
            self.local_free_block_ids.append(block_id_in_group)

    def remove(self, block_id):
        self.n_free_blocks -= 1
        group_id = self.get_group_idx(block_id)
        block_id_in_group = block_id - self.num_blocks_start_end[group_id] # -offset
        if group_id != -1:
            self.multi_free_block_ids[group_id].remove(block_id_in_group)
        else:
            self.local_free_block_ids.remove(block_id_in_group)

    def __len__(self):
        return self.n_free_blocks

    def __getitem__(self, index: int):
        # 如果所有 external group 都空，直接处理
        if all(len(g) == 0 for g in self.multi_free_block_ids):
            if len(self.local_free_block_ids) == 0:
                raise ValueError("external 和 local 的 block 都空了")
            # external 都空 → 返回 local
            return self.local_free_block_ids._heap[index].block_id

        # 找到第一个非空的 external group
        for _ in range(self.n_group):
            if len(self.multi_free_block_ids[self.counter]) > 0:
                break
            print(f"group_id:{self.counter} 的 block 已经全部用完")
            self.counter = (self.counter + 1) % self.n_group

        external_block = self.multi_free_block_ids[self.counter]._heap[index]

        # 如果 local 为空，直接返回 external
        if len(self.local_free_block_ids) == 0:
            return external_block.block_id

        local_block = self.local_free_block_ids._heap[index]
        # print(f'{external_block.block_id},{local_block.block_id}')
        if external_block < local_block:
            self.counter = (self.counter + 1) % self.n_group
            return external_block.block_id
        else:
            return local_block.block_id
    
    def master_scale_down_many(self,slave_idx:int, n:int):
        self.multi_free_block_ids[slave_idx].master_scale_down_many(n)
        num_blocks = self.minimum_scaling_block_count[slave_idx] * n
        self.n_free_blocks - num_blocks
    
    def sync_num_free_blocks(self):
        self.n_free_blocks = 0
        for free_blocks_ids  in self.multi_free_block_ids:
            self.n_free_blocks += len(free_blocks_ids)
        self.n_free_blocks += len(self.local_free_block_ids)
        # print(f'n_free_blocks sync:{self.n_free_blocks}')

class BlockManager:

    def __init__(self, config):
        self.config = config
        num_blocks = config.num_kvcache_blocks
        assert num_blocks > 0
        self.role = config.role
        self.block_size = config.kvcache_block_size
        num_blocks_start_end = config.external_kvcache_config.num_blocks_start_end
        # self.blocks: list[Block] = [Block(i) for i in range(num_blocks)]
        local_num_blocks = config.local_num_blocks
        self.hash_to_block_id: dict[int, int] = dict()
        self.block_id_to_hash: dict[int, int] = dict()
        
        if self.role == 'master': #master

            self.slave_rank_to_idx = {}
            for i,rank in enumerate(config.slave_list):
                self.slave_rank_to_idx[f'slave{rank}'] = i
            self.minimum_scaling_block_count = [0 for _ in range(len(config.slave_list))]
            self.zmq_server = ZMQServer()
            self.collect_minimum_scaling_block_count_from_slave()
            self.blocks= [Block(i,compare_mode = 'local_first', external_blocks_num = num_blocks_start_end[-1]) for i in range(num_blocks)]
            self.free_block_ids= MultiFreeBlockIds(self.blocks, num_blocks_start_end, local_num_blocks,config, self.minimum_scaling_block_count)
            self.can_allocate = self.master_can_allocate
            self.broadcast_ready_to_slaves()
        elif global_config['use_priority_queue']: #slave
             self.zmq_client = ZMQClient(name=f'slave{config.rank}')
             self.blocks = [Block(i) for i in range(num_blocks)]
             self.free_block_ids = FreeBlockIds(blocks = self.blocks,offset = 0,config = config, minimum_scaling_block_count=config.slave_minimum_scaling_block_count) 
             self.can_allocate = self.slave_can_allocate
             self.notify_master_minimum_scaling_block_count()
             self.wait_for_master_ready()
        else:
            self.blocks = [Block(i) for i in range(num_blocks)]
            self.free_block_ids:deque[int] = deque(range(num_blocks))
            self.can_allocate = self.slave_can_allocate
        self.used_block_ids: set[int] = set()
        self.occupied_block_ids: set[int] = set()

    @classmethod
    def compute_hash(cls, token_ids: list[Union[int,list]], prefix: int = -1):
        for i in range(len(token_ids)):
            if isinstance(token_ids[i],list):
                token_ids[i] = token_ids[i][0]
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        h1 = h.intdigest()
        # h2 = token_ids[-1]
        # print(h1,h2)
        # print(f"h.intdigest():{h.intdigest()}")
        return h1 

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return self.blocks[block_id]

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    # def can_allocate(self, seq: Sequence) -> bool:
    #     return len(self.free_block_ids) >= seq.num_blocks

    def print_block_info(self):
        hit_count  = [block.hit_count for block in self.blocks[:1024]]
        print('hit_count:',hit_count)
    
    def master_can_allocate(self,seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks
    
    def slave_can_allocate(self, seq: Sequence) -> bool:
        if len(self.free_block_ids) >= seq.num_blocks:
            return True
        else:
            can_used_block_num = len(self.free_block_ids) + self.free_block_ids.num_blocks_lent
            if can_used_block_num >= seq.num_blocks:
                num_blocks_to_scale_up = seq.num_blocks - len(self.free_block_ids)
                print(f'xxx num_blocks_to_scale_up:{num_blocks_to_scale_up}, seq.num_blocks:{seq.num_blocks}, len(self.free_block_ids):{len(self.free_block_ids)}')
                times_to_scale_up = (num_blocks_to_scale_up + self.config.slave_minimum_scaling_block_count - 1) // self.config.slave_minimum_scaling_block_count
                self.zmq_client.send_dict({'num_blocks_to_scale_up':num_blocks_to_scale_up,'times_to_scale_up':times_to_scale_up})
                self.free_block_ids.slave_scale_up_with_specific_blocks(num_blocks_to_scale_up)
                return True
            else:
                return False
    
    def master_check_blocks_update(self):
        messages = self.zmq_server.recv_all_dict_nonblock()
        if len(messages) > 0:
            print('find blocks_update!') 
            print(messages)
        
            for slave_name, info in messages:
                slave_idx = self.slave_rank_to_idx[slave_name.decode('utf-8')]
                free_block_ids = self.free_block_ids.multi_free_block_ids[slave_idx]
                free_block_ids.master_scale_down_many(info['times_to_scale_up'])
            self.free_block_ids.sync_num_free_blocks()


    def allocate(self, seq: Sequence):
        # 只针对prefill阶段
        assert not seq.block_table
        h = -1
        cache_miss = False
        last_full_block_idx = 0
        # print("ccccc")
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1:
            # if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]

                # print(f"新分配：{block_id}")
                block = self._allocate_block(block_id)
                block.hit_count = 0
                block.prefix_position = i
                block.prefix_depth = 0
            else:
                # cache 命中
                seq.num_cached_tokens += self.block_size
                if block_id in self.used_block_ids:
                    block = self.blocks[block_id]
                    block.ref_count += 1
                else:
                    # 命中释放的block
                    block = self._allocate_block(block_id)
                # print(f'position:{block.prefix_position},{i}')
            
            if h != -1:
                # h ！=-1表示当前token_ids长度刚好等于block_size,这个block满了
                last_full_block_idx = max(last_full_block_idx, i)
                block.hit_count += 1
                block.update(h, token_ids)
                # self.hash_to_block_id[h] = block_id
                old_hash = self.block_id_to_hash.get(block_id,-1)
                if old_hash != -1:
                    self.hash_to_block_id.pop(old_hash)
                self.hash_to_block_id[h] = block_id
                self.block_id_to_hash[block_id] = h
            else:
                #擦除已有的缓存
                old_hash = self.block_id_to_hash.get(block_id,-1)
                if old_hash != -1:
                    # print(f"old_hash:{old_hash}")
                    self.hash_to_block_id.pop(old_hash)
                    self.block_id_to_hash.pop(block_id)
            seq.block_table.append(block_id)

        for i in range(last_full_block_idx):
            block_id = seq.block_table[i]
            block = self.blocks[block_id]
            block.prefix_depth = max(block.prefix_depth, last_full_block_idx)
        seq.extra_info['prefix_cached'] = seq.num_cached_tokens

    def deallocate(self, seq: Sequence):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= (len(seq) % self.block_size == 1)

    def may_append(self, seq: Sequence):
        block_table = seq.block_table
        last_block = self.blocks[block_table[-1]]
        if len(seq) % self.block_size == 1:
            assert last_block.hash != -1
            block_id = self.free_block_ids[0]
            self._allocate_block(block_id)
            block_table.append(block_id)
        elif len(seq) % self.block_size == 0:
            assert last_block.hash == -1
            token_ids = seq.block(seq.num_blocks-1)
            prefix = self.blocks[block_table[-2]].hash if len(block_table) > 1 else -1
            h = self.compute_hash(token_ids, prefix)
            last_block.update(h, token_ids)
            self.hash_to_block_id[h] = last_block.block_id
        else:
            assert last_block.hash == -1
    
    def usage_rate(self) -> float:
        """返回当前 block 使用率 (0.0 ~ 1.0)."""
        total_blocks = len(self.blocks)
        used_blocks = len(self.used_block_ids)
        return used_blocks / total_blocks if total_blocks > 0 else 0.0
    
    def collect_minimum_scaling_block_count_from_slave(self):
        n = len(self.config.slave_list)
        while n>0:
            ident, result = self.zmq_server.recv_dict()
            print('result',result,ident.decode('utf-8'))
            idx = self.slave_rank_to_idx[ident.decode('utf-8')]
            self.minimum_scaling_block_count[idx] = result['minimum_scaling_block_count']
            n -= 1
        
    
    def notify_master_minimum_scaling_block_count(self):
        self.zmq_client.send_dict({'minimum_scaling_block_count':self.config.master_minimum_scaling_block_count})

    
    def broadcast_ready_to_slaves(self):
        for slave_id in self.slave_rank_to_idx.keys():
            self.zmq_server.send_dict(slave_id.encode('utf-8'), {'message': "ready"})
    
    def wait_for_master_ready(self):
        result = self.zmq_client.recv_dict()
        assert result['message'] == 'ready'
        
            
