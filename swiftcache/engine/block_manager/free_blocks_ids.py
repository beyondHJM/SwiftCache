"""Free lists with explicit ownership; all public IDs are global IDs.

Changing a boundary never moves KV tensors. Active blocks are drained by the
coordinator before a boundary can be handed to the other model.
"""
import heapq


class FreeBlockIdsBase:
    def __init__(self, blocks, offset=0, config=None, minimum_scaling_block_count=1):
        self.blocks = blocks
        self.offset = offset
        self.config = config
        self.minimum_scaling_block_count = max(1, minimum_scaling_block_count)
        self.by_id = {b.block_id: b for b in blocks}
        self._free = set()
        self._heap_cache=[]
        self._heap_dirty=True

    @property
    def _heap(self):
        if self._heap_dirty:
            self._heap_cache=[self.by_id[i] for i in self._free]
            heapq.heapify(self._heap_cache)
            self._heap_dirty=False
        return self._heap_cache

    def push(self, block_id):
        if not self.owns(block_id):
            return
        if self.by_id[block_id].ref_count:
            raise RuntimeError('Cannot free a referenced block')
        if block_id in self._free:
            raise RuntimeError('Double free')
        self._free.add(block_id)
        self._heap_dirty=True

    append = push

    def remove(self, block_id):
        if block_id not in self._free:
            raise RuntimeError(f'Block {block_id} is not free/owned')
        self._free.remove(block_id)
        if not self._heap_dirty and self._heap_cache and self._heap_cache[0].block_id==block_id:
            heapq.heappop(self._heap_cache)
        else:
            self._heap_dirty=True
        return True

    def remove_batch_with_global_id(self, ids):
        self._free.difference_update(ids)
        self._heap_dirty=True

    def __len__(self):
        return len(self._free)

    def __getitem__(self, index):
        return self._heap[index].block_id


class MasterFreeBlockIds(FreeBlockIdsBase):
    def __init__(self, blocks, offset=0, config=None, minimum_scaling_block_count=1, reserved=None):
        super().__init__(blocks, offset, config, minimum_scaling_block_count)
        self.init_block_num = minimum_scaling_block_count if reserved is None else reserved
        if not 0 <= self.init_block_num <= len(blocks):
            raise ValueError('Invalid initial reserved boundary')
        self.draining = set()
        self._free = {b.block_id for b in blocks[self.init_block_num:]}

    def owns(self, block_id):
        return (block_id in self.by_id and
                block_id >= self.offset + self.init_block_num and
                block_id not in self.draining)

    def drain(self, target):
        if not self.init_block_num <= target <= len(self.blocks):
            raise ValueError('Invalid reclaim boundary')
        ids = set(range(self.offset + self.init_block_num, self.offset + target))
        self.draining.update(ids)
        self._free.difference_update(ids)
        self._heap_dirty=True
        return ids

    def set_reserved(self, target):
        if not 0 <= target <= len(self.blocks):
            raise ValueError('Invalid reserved boundary')
        old = self.init_block_num
        if target > old:
            ids = set(range(self.offset + old, self.offset + target))
            if any(self.by_id[i].ref_count for i in ids):
                raise RuntimeError('Attempted to hand off active KV')
            self._free.difference_update(ids)
            self.draining.difference_update(ids)
        else:
            ids = set(range(self.offset + target, self.offset + old))
            if any(self.by_id[i].ref_count for i in ids):
                raise RuntimeError('Returned memory is still referenced')
            self._free.update(ids)
        self.init_block_num = target
        self._heap_dirty=True
        return ids

    def scale_down(self, n):
        return self.set_reserved(self.init_block_num + n*self.minimum_scaling_block_count)


class SlaveFreeBlockIds(FreeBlockIdsBase):
    def __init__(self, blocks, offset=0, config=None, minimum_scaling_block_count=1):
        super().__init__(blocks, offset, config, minimum_scaling_block_count)
        initial = getattr(config, 'elastic_initial_blocks', 1)
        unit = self.minimum_scaling_block_count
        self.init_block_num = min(len(blocks), ((initial+unit-1)//unit)*unit)
        self._free = {b.block_id for b in blocks[:self.init_block_num]}

    @property
    def num_blocks_lent(self):
        return len(self.blocks) - self.init_block_num

    def owns(self, block_id):
        return block_id in self.by_id and block_id < self.init_block_num

    def set_reserved(self, target):
        if not 0 <= target <= len(self.blocks):
            raise ValueError('Invalid worker boundary')
        old = self.init_block_num
        ids = set(range(min(old,target), max(old,target)))
        if any(self.by_id[i].ref_count for i in ids):
            raise RuntimeError('Attempted to return active worker KV')
        if target > old:
            self._free.update(ids)
        else:
            self._free.difference_update(ids)
        self.init_block_num = target
        self._heap_dirty=True
        return ids


class MultiFreeBlockIds:
    def __init__(self, blocks, num_blocks_start_end, local_num_blocks, config, minimum_scaling_block_count):
        self.blocks = blocks
        self.num_blocks_start_end = num_blocks_start_end
        self.n_group = len(num_blocks_start_end)-1
        self.multi_free_block_ids = []
        for i in range(self.n_group):
            lo,hi = num_blocks_start_end[i:i+2]
            self.multi_free_block_ids.append(MasterFreeBlockIds(
                blocks[lo:hi], lo, config, minimum_scaling_block_count[i],
                config.initial_reserved_master[i]))
        lo = num_blocks_start_end[-1]
        self.local_free_block_ids = MasterFreeBlockIds(blocks[lo:],lo,config,reserved=0)

    @property
    def groups(self):
        return self.multi_free_block_ids+[self.local_free_block_ids]

    def get_group_idx(self, block_id):
        for i,(lo,hi) in enumerate(zip(self.num_blocks_start_end,self.num_blocks_start_end[1:])):
            if lo <= block_id < hi:
                return i
        return -1

    def group(self, block_id):
        idx=self.get_group_idx(block_id)
        return self.multi_free_block_ids[idx] if idx >= 0 else self.local_free_block_ids

    def owns(self, block_id):
        return self.group(block_id).owns(block_id)

    def append(self, block_id):
        self.group(block_id).append(block_id)

    def remove(self, block_id):
        return self.group(block_id).remove(block_id)

    def __len__(self):
        return sum(len(g) for g in self.groups)

    def __getitem__(self, index):
        if index != 0:
            raise IndexError('Only the next candidate is supported')
        return min((g._heap[0] for g in self.groups if len(g))).block_id

    def sync_num_free_blocks(self):
        return len(self)
