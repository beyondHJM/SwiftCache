import xxhash
import numpy as np
from swiftcache.engine.sequence import Sequence
from swiftcache.engine.block_manager.block import Block
from swiftcache.engine.block_manager.free_blocks_ids import MultiFreeBlockIds,SlaveFreeBlockIds
from swiftcache.config import Config
from swiftcache.utils.zmq_wrapper import ZMQServer, ZMQClient
class BlockManagerBase:
    def __init__(self, config):
        self.config = config
        self.role = config.role
        self.block_size = config.kvcache_block_size
        num_blocks = config.num_kvcache_blocks
        assert num_blocks > 0
        self.blocks = []
        self.used_block_ids: set[int] = set()
        self.occupied_block_ids: set[int] = set()
        self.hash_to_block_id: dict[int, int] = {}
        self.block_id_to_hash: dict[int, int] = {}

    @classmethod
    def compute_hash(cls, token_ids: list, prefix: int = -1):
        for i in range(len(token_ids)):
            if isinstance(token_ids[i], list):
                token_ids[i] = token_ids[i][0]
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8, "little"))
        h.update(np.array(token_ids).tobytes())
        return h.intdigest()

    def _allocate_block(self, block_id: int) -> Block:
        block = self.blocks[block_id]
        assert block.ref_count == 0
        block.reset()
        self.free_block_ids.remove(block_id)
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id: int) -> Block:
        assert self.blocks[block_id].ref_count == 0
        self.used_block_ids.remove(block_id)
        self.free_block_ids.append(block_id)

    def allocate(self, seq: Sequence):
        # 只针对prefill阶段
        assert not seq.block_table
        h = -1
        cache_miss = False
        last_full_block_idx = 0
        for i in range(seq.num_blocks):
            token_ids = seq.block(i)
            h = self.compute_hash(token_ids, h) if len(token_ids) == self.block_size else -1
            block_id = self.hash_to_block_id.get(h, -1)
            if block_id == -1:
            # if block_id == -1 or self.blocks[block_id].token_ids != token_ids:
                cache_miss = True
            if cache_miss:
                block_id = self.free_block_ids[0]
                # print(f"新block:{block_id}")
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
                    self.hash_to_block_id.pop(old_hash,None)
                self.hash_to_block_id[h] = block_id
                self.block_id_to_hash[block_id] = h
            else:
                #擦除已有的缓存
                old_hash = self.block_id_to_hash.get(block_id,-1)
                if old_hash != -1:
                    # print(f"old_hash:{old_hash}")
                    self.hash_to_block_id.pop(old_hash,None)
                    self.block_id_to_hash.pop(block_id,None)
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

    def can_allocate(self):
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement can_allocate() method."
        )
    
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

    def usage_rate(self):
        total_blocks = len(self.blocks)
        used_blocks = len(self.used_block_ids)
        return used_blocks / total_blocks if total_blocks > 0 else 0.0


class MasterBlockManager(BlockManagerBase):
    def __init__(self, config):
        super().__init__(config)
        num_blocks_start_end = config.external_kvcache_config.num_blocks_start_end
        local_num_blocks = config.local_num_blocks
        self.slave_rank_to_idx = {f'slave{rank}': i for i, rank in enumerate(config.slave_list)}
        self.minimum_scaling_block_count = [0 for _ in range(len(config.slave_list))]
        self.zmq_server = ZMQServer()
        
        # collect scaling block count from slaves
        self.collect_minimum_scaling_block_count_from_slave()

        # init block list
        self.blocks = [
            Block(i, compare_mode='local_first', external_blocks_num=num_blocks_start_end[-1])
            for i in range(config.num_kvcache_blocks)
        ]

        # init free block ids manager
        self.free_block_ids = MultiFreeBlockIds(
            self.blocks, num_blocks_start_end, local_num_blocks, config, self.minimum_scaling_block_count
        )

        # self.can_allocate = self.master_can_allocate

        # notify slaves ready
        self.broadcast_ready_to_slaves()

    def print_block_info(self):
        hit_count  = [block.hit_count for block in self.blocks]
        print('------hit_count:',hit_count,len(self.blocks))
        import json
        with open("hit_count.json", "w", encoding="utf-8") as f:
            json.dump(hit_count, f, ensure_ascii=False, indent=4)

    def can_allocate(self, seq: Sequence) -> bool:
        return len(self.free_block_ids) >= seq.num_blocks

    def collect_minimum_scaling_block_count_from_slave(self):
        n = len(self.config.slave_list)
        while n > 0:
            ident, result = self.zmq_server.recv_dict()
            idx = self.slave_rank_to_idx[ident.decode('utf-8')]
            self.minimum_scaling_block_count[idx] = result['minimum_scaling_block_count']
            n -= 1

    def broadcast_ready_to_slaves(self):
        for slave_id in self.slave_rank_to_idx:
            self.zmq_server.send_dict(slave_id.encode('utf-8'), {'message': "ready"})

    def master_check_blocks_update(self):
        messages = self.zmq_server.recv_all_dict_nonblock()
        if messages:
            for slave_name, info in messages:
                slave_idx = self.slave_rank_to_idx[slave_name.decode('utf-8')]
                free_block_ids = self.free_block_ids.multi_free_block_ids[slave_idx]
                free_block_ids.scale_down(info['times_to_scale_up'])
            self.free_block_ids.sync_num_free_blocks()


class SlaveBlockManager(BlockManagerBase):
    def __init__(self, config):
        super().__init__(config)
        self.zmq_client = ZMQClient(name=f'slave{config.rank}')
        self.blocks = [Block(i) for i in range(config.num_kvcache_blocks)]
        self.free_block_ids = SlaveFreeBlockIds(
            blocks=self.blocks, offset=0, config=config,
            minimum_scaling_block_count=config.slave_minimum_scaling_block_count
        )
        # self.can_allocate = self.slave_can_allocate
        self.notify_master_minimum_scaling_block_count()
        self.wait_for_master_ready()

    def can_allocate(self, seq: Sequence) -> bool:
        if len(self.free_block_ids) >= seq.num_blocks:
            return True
        can_used = len(self.free_block_ids) + self.free_block_ids.num_blocks_lent
        if can_used >= seq.num_blocks:
            num_blocks_to_scale_up = seq.num_blocks - len(self.free_block_ids)
            times_to_scale_up = (
                num_blocks_to_scale_up + self.config.slave_minimum_scaling_block_count - 1
            ) // self.config.slave_minimum_scaling_block_count
            self.zmq_client.send_dict({
                'num_blocks_to_scale_up': num_blocks_to_scale_up,
                'times_to_scale_up': times_to_scale_up
            })
            self.free_block_ids.scale_up_with_specific_blocks(num_blocks_to_scale_up)
            return True
        return False

    def notify_master_minimum_scaling_block_count(self):
        self.zmq_client.send_dict({
            'minimum_scaling_block_count': self.config.master_minimum_scaling_block_count
        })

    def wait_for_master_ready(self):
        result = self.zmq_client.recv_dict()
        assert result['message'] == 'ready'


def create_block_manager(config:Config):
    if config.role == 'master':
        return MasterBlockManager(config)
    elif config.role == 'slave':
        return SlaveBlockManager(config)
    else:
        raise ValueError(f"Unknown role: {config.role}")