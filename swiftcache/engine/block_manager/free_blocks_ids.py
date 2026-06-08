import heapq
import time
from typing import Optional
from swiftcache.engine.block_manager.block import Block
from swiftcache.config import Config
class FreeBlockIdsBase:
    def __init__(self, blocks, offset: int = 0, config=None, minimum_scaling_block_count=0):
        self.config = config
        self.blocks = blocks
        self.offset = offset
        self.minimum_scaling_block_count = minimum_scaling_block_count
        self._heap = []

    # ======== Interface-like methods ========
    def scale_up(self, *args, **kwargs):
        """Expand resources (should be overridden by subclass)"""
        raise NotImplementedError("scale_up() is not implemented in this class.")

    def scale_down(self, *args, **kwargs):
        """Shrink resources (should be overridden by subclass)"""
        raise NotImplementedError("scale_down() is not implemented in this class.")

    # ======== Shared methods ========
    def push(self, block_id: int):
        block = self.blocks[block_id]
        heapq.heappush(self._heap, block)

    def append(self, block_id: int):
        self.push(block_id)

    def pop(self) -> Optional[int]:
        if not self._heap:
            return None
        block = heapq.heappop(self._heap)
        return block.block_id

    def peek(self) -> Optional[int]:
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

    def remove_batch_with_global_id(self, global_block_ids: list[int]):
        global_block_ids_set = set(global_block_ids)
        self._heap = [block for block in self._heap if block.block_id not in global_block_ids_set]
        heapq.heapify(self._heap)

    def push_batch_with_global_id(self, block_ids: list[int]):
        new_blocks = [self.blocks[b_id - self.offset] for b_id in block_ids]
        self._heap.extend(new_blocks)
        heapq.heapify(self._heap)

    @property
    def num_blocks_lent(self):
        return len(self.blocks) - len(self._heap)

    def __len__(self):
        return len(self._heap)

    def __getitem__(self, index: int):
        return self._heap[index].block_id


class MasterFreeBlockIds(FreeBlockIdsBase):
    def __init__(self, blocks, offset=0, config=None, minimum_scaling_block_count=0):
        super().__init__(blocks, offset, config, minimum_scaling_block_count)
        # Master keeps all blocks except initial reserved ones
        self.init_block_num = minimum_scaling_block_count
        self._heap = list(blocks[self.init_block_num:])
        heapq.heapify(self._heap)

    def scale_down(self, n: int):
        """Master-specific shrink"""
        num_blocks = self.minimum_scaling_block_count * n
        global_block_ids = [
            block.block_id for block in
            self.blocks[self.init_block_num:self.init_block_num + num_blocks]
        ]
        print(f"[Master] Global block IDs to remove: {global_block_ids}")
        t = time.perf_counter()
        self.remove_batch_with_global_id(global_block_ids)
        print(f"[Master] scale_down took {time.perf_counter() - t:.6f} sec")
        self.init_block_num += num_blocks

    def master_scale_down_many(self, n:int):
        num_blocks = self.minimum_scaling_block_count * n
        global_block_ids = [block.block_id for block in self.blocks[self.init_block_num: self.init_block_num + num_blocks]]
        print(f'即将被缩减的global_block_ids:{global_block_ids}')
        t = time.perf_counter()
        self.remove_batch_with_global_id(global_block_ids)
        print(time.perf_counter()-t)
        self.init_block_num += num_blocks


class SlaveFreeBlockIds(FreeBlockIdsBase):
    def __init__(self, blocks, offset=0, config=None, minimum_scaling_block_count = 1):
        super().__init__(blocks, offset, config, minimum_scaling_block_count)
        # Slave gets initial N blocks
        self.init_block_num = minimum_scaling_block_count
        self._heap = list(blocks[:self.init_block_num])
        heapq.heapify(self._heap)

    def scale_up(self, n: int = 1):
        """Slave-specific expansion"""
        self._heap.extend(
            self.blocks[self.init_block_num:
                        self.init_block_num + self.minimum_scaling_block_count * n]
        )
        heapq.heapify(self._heap)
        self.init_block_num += self.minimum_scaling_block_count * n
        print(f"[Slave] init_block_num after scale_up: {self.init_block_num}")
    
    def scale_up_with_specific_blocks(self,num_blocks:int):
        n = (num_blocks + self.minimum_scaling_block_count - 1) // self.minimum_scaling_block_count
        self.scale_up(n)


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
            free_blocks_ids = MasterFreeBlockIds(blocks[num_blocks_start_end[i]:num_blocks_start_end[i+1]],num_blocks_start_end[i],config,minimum_scaling_block_count[i])
            self.multi_free_block_ids.append(free_blocks_ids)
        self.local_free_block_ids = MasterFreeBlockIds(blocks[num_blocks_start_end[-1]:], num_blocks_start_end[-1], config)
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