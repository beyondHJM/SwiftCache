from bisect import bisect_right
from itertools import accumulate
import json
import pickle
import torch
import torch.distributed as dist
import time
import os
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
from transformers import AutoConfig

from swiftcache.config import Config, ExternalKVCacheConfig
from swiftcache.engine.sequence import Sequence,VisionSequence
from swiftcache.layers.sampler import Sampler
from swiftcache.utils.context import set_context, get_context, reset_context, set_tp_group,set_external_kvcache,set_role, set_external_local_num_blocks
from swiftcache.utils.loader import load_model
from swiftcache.engine.cpu_manager import CpuKVCacheManager 
from swiftcache.engine.external_kvcache import ExternalKVCacheController, ExternalKVCacheManager
from torch.cuda.nvtx import range_push, range_pop
from swiftcache.global_config import global_config
import math

class ModelRunner:

    def __init__(self, config: Config, rank: int, event: Event | list[Event], vl_chat_processor=None):
        self.pid = os.getpid()
        self.rank = rank
        self.record_pid()
        self.role = config.role
        self.prefix_str = f'[PID:{self.pid}, RANK:{rank} ROLE:{self.role}]'
        self.config = config
        hf_config = config.hf_config
        hf_language_config = getattr(hf_config, 'language_config', None)
        self.hf_language_config = hf_language_config if hf_language_config is not None else hf_config
        self.block_size = config.kvcache_block_size
        self.enforce_eager = config.enforce_eager
        self.tp_size = config.tensor_parallel_size
        self.num_external_kvcache = len(config.slave_list)
        self.master_list = config.master_list
        self.slave_list = config.slave_list
        # self.world_size = config.tensor_parallel_size 
        self.world_size = config.tensor_parallel_size + self.num_external_kvcache
        self.event = event

        if self.role == 'master':
            self.controller = ExternalKVCacheController(self.rank,config.slave_event,config.slave_ready_event,self.master_list, self.slave_list,config.dist_port)
            
        elif self.role == 'slave':
                self.manager = ExternalKVCacheManager(self.rank, config.slave_event, config.slave_ready_event,self.master_list, self.slave_list,config.dist_port)
        else:
            dist.init_process_group("nccl", f"tcp://localhost:{config.dist_port}", world_size=self.world_size, rank=rank)

        self.vl_chat_processor=vl_chat_processor
        # dist.init_process_group("nccl", "tcp://localhost:2334", world_size=self.world_size, rank=rank)
        self.tp_group = dist.new_group(ranks=config.tp_group)
        self.first_rank = config.tp_group[0]
        set_tp_group(self.tp_group,config.tp_group[0])
        set_role(self.role)
        # context = get_context()
        # context.layer_elem = config.kvcache_block_size * hf_config.num_hidden_layers
       
        torch.cuda.set_device(rank)
        default_dtype = torch.get_default_dtype()
        hf_config.torch_dtype = torch.bfloat16 
        torch.set_default_dtype(hf_config.torch_dtype)
        torch.set_default_device(f"cuda")
        self.model = self.init_model(hf_config)
        load_model(self.model, config.model)
        self.cpu_kv_cache_manager = None
        self.sampler = Sampler()
        
        self.warmup_model()

        # ============
        if global_config.get('kv_cache_strategy') == 'cpu':
            self.allocate_kv_cache_with_cpu_kvcache()
        elif global_config.get('kv_cache_strategy') == 'normal':
            self.allocate_kv_cache()
        elif self.role == 'master':
            self.allocate_kv_cache_with_external_kvcache()
        elif self.role == 'slave':
            self.allocate_kv_cache()


        # ============
        if not self.enforce_eager:
            self.capture_cudagraph()
        # torch.set_default_device("cpu")
        torch.set_default_dtype(default_dtype)
       

        if self.tp_size > 1:
            if rank == 0:
                self.tp_shm = SharedMemory(name="swiftcache", create=True, size=2**20)
                dist.barrier(self.tp_group)
            else:
                dist.barrier(self.tp_group)
                self.tp_shm = SharedMemory(name="swiftcache")
                self.loop()


    def exit(self):
        if self.tp_size > 1:
            self.tp_shm.close()
            dist.barrier(self.tp_group)
            if self.rank == self.first_rank:
                self.tp_shm.unlink()
        if not self.enforce_eager:
            del self.graphs, self.graph_pool
        

        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        while True:
            try:
                method_name, args = self.read_shm()
                self.call(method_name, *args)
                if method_name == "exit":
                    break
            except Exception as e:
                print(f"[rank {self.rank}] Error in loop: {e}")
                import traceback
                traceback.print_exc()
                exit(1)

    def read_shm(self):
        assert self.tp_size > 1 and self.rank
        self.event.wait()
        n = int.from_bytes(self.tp_shm.buf[0:4], "little")
        method_name, *args = pickle.loads(self.tp_shm.buf[4:n+4])
        self.event.clear()
        return method_name, args

    def write_shm(self, method_name, *args):
        assert self.tp_size > 1 and not self.rank
        data = pickle.dumps([method_name, *args])
        n = len(data)
        self.tp_shm.buf[0:4] = n.to_bytes(4, "little")
        self.tp_shm.buf[4:n+4] = data
        for event in self.event:
            event.set()

    def call(self, method_name, *args):
        if self.tp_size > 1 and self.rank == self.first_rank:
            self.write_shm(method_name, *args)
        method = getattr(self, method_name, None)
        res =  method(*args)
        return res

    def warmup_model(self):
        t = time.perf_counter()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        max_num_batched_tokens, max_model_len = self.config.max_num_batched_tokens, self.config.max_model_len
        print(self.config.max_num_batched_tokens,self.config.max_model_len,self.config.max_num_seqs)
        num_seqs = min(max_num_batched_tokens // max_model_len, self.config.max_num_seqs)
        print(f'num_seqs:{num_seqs}')
        seqs = [Sequence([0] * max_model_len) for _ in range(num_seqs)]
        self.run(seqs, True)
        torch.cuda.empty_cache()
        print(f"warmup end, time usage:{(time.perf_counter()-t)*1000:.2f} ms")

    def allocate_kv_cache_with_cpu_kvcache(self):
        GREEN = "\033[92m"    # 亮绿色
        RED = "\033[91m"      # 亮红色
        RESET = "\033[0m"     # 重置为默认颜色
        print(f"{RED}使用CPU KV Cache辅助{RESET}")

        config = self.config
        # config.gpu_memory_utilization = global_config['gpu_memory_utilization']
        hf_config = self.hf_language_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        print(f'peak:{peak},current:{current},used:{used}')
        num_kv_heads = hf_config.num_key_value_heads // self.tp_size

        head_dim = getattr(hf_config,'head_dim',None)
        head_dim = head_dim if head_dim else hf_config.hidden_size // hf_config.num_key_value_heads

        num_layers_with_extra_cache  = 1 #hf_config.num_hidden_layers
        block_bytes = 2  *num_layers_with_extra_cache* self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        total_bytes_for_kvcache = int(total * config.gpu_memory_utilization - used - peak + current)
        config.num_kvcache_blocks = total_bytes_for_kvcache // block_bytes
        global_num_kvcache_blocks = global_config['num_kvcache_blocks']
        if global_num_kvcache_blocks != -1:
            config.num_kvcache_blocks = global_num_kvcache_blocks
            print(f'Use global config "num_kvcache_blocks:{global_num_kvcache_blocks}".')
        print(f'num_hidden_layers:{hf_config.num_hidden_layers},cpu_kv_cache_manager needs {block_bytes*config.num_kvcache_blocks*hf_config.num_hidden_layers/(1024*1024)} MB host memory.')
        print(f'total_bytes_for_kvcache_each_layer:{block_bytes*config.num_kvcache_blocks/(1024*1024)}  MB ,num_kvcache_blocks:{config.num_kvcache_blocks},max_num_tokens:{config.num_kvcache_blocks*self.block_size}')
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2,  config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        # self.kv_cache = torch.zeros(2,num_layers_with_extra_cache,config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # module.k_cache = self.kv_cache[0, layer_id]
                # module.v_cache = self.kv_cache[1, layer_id]
                # layer_id += 1
                module.k_cache = self.kv_cache[0]
                module.v_cache = self.kv_cache[1]
        self.cpu_kv_cache_manager = CpuKVCacheManager(hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)

    def allocate_kv_cache_with_external_kvcache(self):
        assert self.controller is not None
        GREEN = "\033[92m"    # 亮绿色
        RED = "\033[91m"      # 亮红色
        BLUE = "\033[94m"      # 亮蓝色
        RESET = "\033[0m"     # 重置为默认颜色
        print(f"{BLUE}使用External KV Cache辅助{RESET}")

        config = self.config
        # config.gpu_memory_utilization = global_config['gpu_memory_utilization']
        # print(f'config.gpu_memory_utilization:{config.gpu_memory_utilization}')
        hf_config = self.hf_language_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        print(f'peak:{peak},current:{current},used:{used}')
        num_kv_heads = hf_config.num_key_value_heads // self.tp_size

        head_dim = getattr(hf_config,'head_dim',None)
        head_dim = head_dim if head_dim else hf_config.hidden_size // hf_config.num_key_value_heads

        num_layers_with_extra_cache  = 1 #hf_config.num_hidden_layers
        block_bytes = 2  *num_layers_with_extra_cache* self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        total_bytes_for_kvcache = int(total * config.gpu_memory_utilization - used - peak + current)
        config.num_kvcache_blocks = total_bytes_for_kvcache // block_bytes
        global_num_kvcache_blocks = global_config['num_kvcache_blocks']
        if global_num_kvcache_blocks != -1:
            config.num_kvcache_blocks = global_num_kvcache_blocks
            print(f'Use global config "num_kvcache_blocks:{global_num_kvcache_blocks}".')
        print(f'num_hidden_layers:{hf_config.num_hidden_layers},external_kv_cache_manager needs {block_bytes*config.num_kvcache_blocks*hf_config.num_hidden_layers/(1024*1024)} MB host memory.')
        # print(f'{BLUE}total_bytes_for_kvcache_each_layer:{block_bytes*config.num_kvcache_blocks/(1024*1024)}  MB ,num_kvcache_blocks:{config.num_kvcache_blocks},max_num_tokens:{config.num_kvcache_blocks*self.block_size}{RESET}')
        print(f"{BLUE}若只保留一层,支持的最大block数量为:{config.num_kvcache_blocks},若保留所有层,则支持的数量为{config.num_kvcache_blocks//hf_config.num_hidden_layers}{RESET}")
        assert config.num_kvcache_blocks > 0
        self.kv_cache = torch.zeros(2,  config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        print(f'{BLUE}kv_cache.shape:{self.kv_cache.shape}{RESET}')
        # self.kv_cache = torch.zeros(2,num_layers_with_extra_cache,config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # module.k_cache = self.kv_cache[0, layer_id]
                # module.v_cache = self.kv_cache[1, layer_id]
                # layer_id += 1
                module.k_cache = self.kv_cache[0]
                module.v_cache = self.kv_cache[1]

        # self.controller.init_kvcache(hf_config.num_hidden_layers, config.external_kvcache_config.num_blocks_start_end, self.block_size, num_kv_heads, head_dim)
        if self.role == 'master':
            self.controller.send_master_kvcache_info(hf_config.num_hidden_layers, self.block_size, num_kv_heads, head_dim)
            sorted_slaves, num_blocks_start_end = self.controller.gather_info_from_slave()
            external_num_blocks = num_blocks_start_end[-1]
            print(f'{BLUE}master block buffer :{config.num_kvcache_blocks}{RESET}')
            # expected_total_num_blocks  = ((num_blocks_start_end[-1]* hf_config.num_hidden_layers) + config.num_kvcache_blocks )//(hf_config.num_hidden_layers + 1)
            local_num_blocks = (config.num_kvcache_blocks - external_num_blocks)//hf_config.num_hidden_layers
            local_num_blocks = local_num_blocks if local_num_blocks > 0 else 0
            total_num_blocks = local_num_blocks + external_num_blocks
            original_num_blocks_without_external_kvcache = config.num_kvcache_blocks // hf_config.num_hidden_layers
            self.local_num_blocks = local_num_blocks
            for item in sorted_slaves:
                print(f'slave_id:{item[0]},num_blocks:{item[1]}')
            print(f'master_id:{self.rank},num_blocks:{self.local_num_blocks}')
            ratio = total_num_blocks / original_num_blocks_without_external_kvcache
            print(f"{BLUE}[Prompt Capacity] {total_num_blocks} blocks, max length:{total_num_blocks*config.kvcache_block_size} — {ratio:.2f}× baseline capacity.{RESET}")
            config.external_kvcache_config.num_blocks_start_end = num_blocks_start_end
            config.external_kvcache_config.num_external_kvcache = len(self.slave_list)
            config.num_kvcache_blocks = total_num_blocks
            config.local_num_blocks = self.local_num_blocks
            self.local_block_id_offset = total_num_blocks #由于临时kvcache(用于计算)，本地kvcache使用同一块tensor，所以需要计算偏移量
            self.global_block_id_offset = num_blocks_start_end[-1]
            print(f'{RED}num_blocks_start_end_with_local_blocks:{num_blocks_start_end}{RESET}')
            set_external_local_num_blocks(external_num_blocks, local_num_blocks)
            set_external_kvcache(self.controller)
            context = get_context()
            context.layer_elem = config.kvcache_block_size * config.local_num_blocks
            context.boundary = config.kvcache_block_size * external_num_blocks
            print(f'context.layer_elem:{context.layer_elem}, context.boundary:{context.boundary}')
            self.controller.send_slave_offset(num_blocks_start_end)
        
        print(f'{self.prefix_str} All external kvcache initialized!,num_blocks_start_end = {num_blocks_start_end}')
        
    def allocate_kv_cache(self):
        GREEN = "\033[92m"    # 亮绿色
        RED = "\033[91m"      # 亮红色
        RESET = "\033[0m"     # 重置为默认颜色
        if self.role == 'slave':
            print(f"{GREEN}使用正常的KV Cache,and this instance is a slave{RESET}")
        else:
            print(f"{GREEN}使用正常的KV Cache{RESET}")
        config = self.config
        # config.gpu_memory_utilization = global_config['gpu_memory_utilization']
        hf_config = self.hf_language_config
        free, total = torch.cuda.mem_get_info()
        used = total - free
        peak = torch.cuda.memory_stats()["allocated_bytes.all.peak"]
        current = torch.cuda.memory_stats()["allocated_bytes.all.current"]
        print(f'peak:{peak},current:{current},used:{used}')
        num_kv_heads = hf_config.num_key_value_heads // self.tp_size

        head_dim = getattr(hf_config,'head_dim',None)
        head_dim = head_dim if head_dim else hf_config.hidden_size // hf_config.num_key_value_heads

        # num_layers_with_extra_cache  = hf_config.num_hidden_layers #hf_config.num_hidden_layers
        block_bytes = 2  * hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim * hf_config.torch_dtype.itemsize
        total_bytes_for_kvcache = int(total * config.gpu_memory_utilization - used - peak + current)
        print(f'total memory:{total * config.gpu_memory_utilization}')
        config.num_kvcache_blocks = total_bytes_for_kvcache // block_bytes
        print(f'max number of kvcache blocks:{config.num_kvcache_blocks}')
        global_num_kvcache_blocks = global_config['num_kvcache_blocks']
        if global_num_kvcache_blocks >0:
            config.num_kvcache_blocks = global_num_kvcache_blocks
            print(f'Use global config "num_kvcache_blocks:{global_num_kvcache_blocks}".')
        print(f'total_bytes_for_kvcache_all_layer:{block_bytes*config.num_kvcache_blocks/(1024*1024)} MB ,num_kvcache_blocks:{config.num_kvcache_blocks},max_num_tokens:{config.num_kvcache_blocks*self.block_size}')
        assert config.num_kvcache_blocks > 0
        # self.kv_cache = torch.zeros(2,  hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        self.kv_cache = torch.zeros(2,  config.num_kvcache_blocks * hf_config.num_hidden_layers, self.block_size, num_kv_heads, head_dim)
        print(f"----config.num_kvcache_blocks:{config.num_kvcache_blocks*hf_config.num_hidden_layers*self.block_size}")
        layer_id = 0
        for module in self.model.modules():
            if hasattr(module, "k_cache") and hasattr(module, "v_cache"):
                # module.k_cache = self.kv_cache[0,layer_id]
                # module.v_cache = self.kv_cache[1,layer_id]
                layer_id += 1
                module.k_cache = self.kv_cache[0]
                module.v_cache = self.kv_cache[1]
        # self.cpu_kv_cache_manager = CpuKVCacheManager(hf_config.num_hidden_layers, config.num_kvcache_blocks, self.block_size, num_kv_heads, head_dim)
        context = get_context()
        context.num_hidden_layers = hf_config.num_hidden_layers

        if self.role == 'slave':
            while not self.manager.recieved_master_kvcache_info:
                time.sleep(1)
            self.manager.init_kvcache_with_existing_tensor(self.kv_cache[0], self.kv_cache[1])
        master_block_element_all_layers = self.manager.master_block_element_all_layers
        slave_block_element_all_layers =  hf_config.num_hidden_layers * self.block_size * num_kv_heads * head_dim
        lcm_val = math.lcm(master_block_element_all_layers, slave_block_element_all_layers)
        print(f"{self.prefix_str} LCM =", lcm_val)
        master_minimum_scaling_block_count = lcm_val//master_block_element_all_layers
        slave_minimum_scaling_block_count = lcm_val//slave_block_element_all_layers
        print(f'master_minimum_scaling_block_count:{master_minimum_scaling_block_count}')
        print(f'slave_minimum_scaling_block_count:{slave_minimum_scaling_block_count}')
        config.master_minimum_scaling_block_count = master_minimum_scaling_block_count
        config.slave_minimum_scaling_block_count = slave_minimum_scaling_block_count
        # print(f"yyyyy:{config.slave_minimum_scaling_block_count}")


    def prepare_block_tables(self, seqs: list[Sequence]):
        max_len = max(len(seq.block_table) for seq in seqs)
        if self.role == 'master' and not global_config['transfer_full_kvcache']:
            block_tables =[seq.local_block_table + [-1] * (max_len - len(seq.local_block_table)) for seq in seqs]
        else:
            block_tables = [seq.block_table + [-1] * (max_len - len(seq.block_table)) for seq in seqs]
        block_tables = torch.tensor(block_tables, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        return block_tables

    def set_local_block_tables(self, seqs,is_prefill, config):
        local_cached_blocks = []
        local_uncached_blocks = []
        cached_blocks=[]
        uncached_blocks = []
        local_block_id = 0
        global_local_mapping = dict()
        slot_mapping = []
        block_id_to_fix = []
        if not seqs[0].block_table: #wamrup
            return cached_blocks, uncached_blocks,local_cached_blocks, local_uncached_blocks, [0], [0], slot_mapping, block_id_to_fix
        external_num_blocks = config.external_kvcache_config.num_blocks_start_end[-1]
        local_num_blocks = config.local_num_blocks
        num_kvcache_blocks = config.num_kvcache_blocks
        
        # block_id_set = set()
        cum_blocks_per_slave = [0] * (self.num_external_kvcache+1)
        loaded_num = [0] * (self.num_external_kvcache)
        stored_num  = [0] * (self.num_external_kvcache)
        counters = [0] * (self.num_external_kvcache) #external_kvcache + local_kvcache

        for seq in seqs:
            for i in range(len(seq.cum_blocks_per_slave)):
                cum_blocks_per_slave[i] += seq.cum_blocks_per_slave[i]
        #保证在单个slave中cache连续，这样可以优化传输，目前只能保证prefill阶段的uncache连续, 
        if is_prefill:
            #prefix cache
            for idx, seq in enumerate(seqs):
                seq.local_block_table = []
                
                # seq.local_cached_blocks = list(range(block_ids_sum, block_ids_sum + seq.num_cached_blocks))
                for i in range(seq.num_cached_blocks):
                    global_block_id = seq.block_table[i]
                    if global_block_id >= external_num_blocks:
                        id = global_block_id
                        seq.local_block_table.append(id)
                        block_id_to_fix.append((idx, i))
                    else:
                        local_id = global_local_mapping.get(global_block_id,None)
                        if local_id is None:
                            slave_id = seq.block_belong_to_slave[i]
                            local_block_id_offset = cum_blocks_per_slave[slave_id]
                            local_block_id = local_block_id_offset + counters[slave_id]
                            global_local_mapping[global_block_id] = local_block_id
                            seq.local_block_table.append(local_block_id)
                            counters[slave_id] += 1
                            loaded_num[slave_id] += 1
                        else:
                            seq.local_block_table.append(local_id)
                        
            for idx, seq in enumerate(seqs):
                # seq.local_uncached_blocks = list(range(block_ids_sum , block_ids_sum + seq.num_blocks - seq.num_cached_blocks))
                # seq.local_block_table.extend(seq.local_uncached_blocks)
                for i in range(seq.num_cached_blocks, seq.num_blocks):
                    global_block_id = seq.block_table[i]
                    if global_block_id >= external_num_blocks:
                        id = global_block_id
                        seq.local_block_table.append(id)
                        block_id_to_fix.append((idx, i))
                    else:
                        local_id = global_local_mapping.get(global_block_id,None)
                        if local_id is None:
                            slave_id = seq.block_belong_to_slave[i]
                            local_block_id_offset = cum_blocks_per_slave[slave_id]
                            local_block_id = local_block_id_offset + counters[slave_id]
                            global_local_mapping[global_block_id] = local_block_id
                            seq.local_block_table.append(local_block_id)
                            counters[slave_id] += 1
                            stored_num[slave_id] += 1
                        else:
                            seq.local_block_table.append(local_id)
                
                    block_id = seq.block_table[i] if global_config['transfer_full_kvcache'] else seq.local_block_table[i]
                    start = block_id * self.block_size
                    if i != seq.num_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_num_tokens 
                    slot_mapping.extend(list(range(start, end)))
                    
                # print('seq.local_block_table', seq.local_block_table)
                cached_blocks.extend(seq.block_table[:seq.num_cached_blocks])
                uncached_blocks.extend(seq.block_table[seq.num_cached_blocks:])
                local_cached_blocks.extend(seq.local_block_table[:seq.num_cached_blocks])
                local_uncached_blocks.extend(seq.local_block_table[seq.num_cached_blocks:])
        else:
            for idx, seq in enumerate(seqs):
                # print(f'seq.block_table:{seq.block_table}') 
                seq.local_block_table = []
                for i in range(seq.num_blocks):
                    global_block_id = seq.block_table[i]
                    if global_block_id >= external_num_blocks:
                        id = global_block_id
                        seq.local_block_table.append(id)
                        block_id_to_fix.append((idx, i))
                    else:
                        local_id = global_local_mapping.get(global_block_id,None)
                        if local_id is None:
                            slave_id = seq.block_belong_to_slave[i]
                            local_block_id_offset = cum_blocks_per_slave[slave_id]
                            local_block_id = local_block_id_offset + counters[slave_id]
                            global_local_mapping[global_block_id] = local_block_id
                            seq.local_block_table.append(local_block_id)
                            counters[slave_id] += 1
                            loaded_num[slave_id] += 1
                            if i == (seq.num_blocks - 1):
                                stored_num[slave_id] += 1
                        else:
                            seq.local_block_table.append(local_id)
                # seq.local_cached_blocks = list(range(block_ids_sum, block_ids_sum + seq.num_blocks))
                # seq.local_block_table = seq.local_cached_blocks
                # seq.local_uncached_blocks = [seq.local_cached_blocks[-1]]
                block_id = seq.block_table[-1] if global_config['transfer_full_kvcache'] else seq.local_block_table[-1]
                slot_mapping.append(block_id * self.block_size + seq.last_block_num_tokens  - 1)
                cached_blocks.extend(seq.block_table)
                uncached_blocks.extend([seq.block_table[-1]])
                local_cached_blocks.extend(seq.local_block_table)
                local_uncached_blocks.append(seq.local_block_table[-1])
        # print(global_local_mapping)
        num_blocks_start_end = self.config.external_kvcache_config.num_blocks_start_end
        cached_blocks = list(dict.fromkeys(cached_blocks))
        uncached_blocks = list(dict.fromkeys(uncached_blocks))
        # if not is_prefill:
        #     print('local_cached_blocks_before',cached_blocks)
        cached_blocks = self.sort_by_intervals(num_blocks_start_end, cached_blocks)
        # if not is_prefill:
        #     print('local_cached_blocks_after',cached_blocks)
        uncached_blocks = self.sort_by_intervals(num_blocks_start_end, uncached_blocks)
        local_cached_blocks = [global_local_mapping[global_block_id] for global_block_id in cached_blocks]
        local_uncached_blocks = [global_local_mapping[global_block_id] for global_block_id in uncached_blocks]
        loaded_start_end = [0] + list(accumulate(loaded_num))
        stored_start_end = [0] + list(accumulate(stored_num))
        local_cached_blocks = list(dict.fromkeys(local_cached_blocks))
        local_uncached_blocks = list(dict.fromkeys(local_uncached_blocks))
        # print(f'cached_blocks:{cached_blocks}')
        # print(f'uncached_blocks:{uncached_blocks}')
        # print(f'local_cached_blocks:{local_cached_blocks}')
        # print(f'local_uncached_blocks:{local_uncached_blocks}')
        # print(f'loaded_start_end:{loaded_start_end}')
        # print(f'stored_start_end:{stored_start_end}')
        # print(f'block_id_to_fix:{block_id_to_fix}')
        # print('-'*30)
        return cached_blocks, uncached_blocks,local_cached_blocks, local_uncached_blocks, loaded_start_end,stored_start_end, slot_mapping, block_id_to_fix
    
    def cal_num_blocks_per_slave(self,seqs:list[Sequence],config:Config):
        num_blocks_start_end = config.external_kvcache_config.num_blocks_start_end
        for seq in seqs:
            # print(f'{self.prefix_str} seq_id:{seq.seq_id},block_table:{seq.block_table}')
            num_blocks_per_slave = [0]*(len(num_blocks_start_end)-1)
            block_belong_to_slave = [0]*len(seq.block_table)
            for i in range(len(seq.block_table)):
                block_id = seq.block_table[i]
                idx = bisect_right(num_blocks_start_end, block_id) - 1
                if 0 <= idx < len(num_blocks_per_slave):
                    num_blocks_per_slave[idx] += 1
                    block_belong_to_slave[i] = idx
                # else:
                #     print("xxxxx")

            cum_blocks_per_slave = [0] + list(accumulate(num_blocks_per_slave))
            seq.num_blocks_per_slave = num_blocks_per_slave
            seq.cum_blocks_per_slave = cum_blocks_per_slave
            seq.block_belong_to_slave = block_belong_to_slave
    
    def sort_by_intervals(self,list1, list2):
        # 构造区间列表
        intervals = [(list1[i], list1[i+1]) for i in range(len(list1)-1)]
        # 初始化分组，每个组是一个列表
        groups = [[] for _ in range(len(intervals))]
        
        for num in list2:
            # 判断 num 属于哪个区间
            for idx, (low, high) in enumerate(intervals):
                if low <= num < high:
                    groups[idx].append(num)
                    break
            else:
                # 如果不属于任何区间（比如 num >= list1[-1]），可以按需求处理
                # 这里按题目例子，list2 所有数都在区间内
                pass
        
        # 按区间顺序拼接分组
        result = []
        for g in groups:
            result.extend(g)
        return result
                

    def prepare_prefill(self, seqs: list[Sequence]):
        # t1 = time.perf_counter()
        input_ids = []
        positions = []
        input_embeds = []
        cu_seqlens_q = [0]
        cu_seqlens_k = [0]
        max_seqlen_q = 0
        max_seqlen_k = 0
        slot_mapping = []
        block_tables = None
        non_cached_block_table = []
        cached_block_table = []
        block_id_to_fix = []
        for seq in seqs:
            seqlen = len(seq)
            input_ids.extend(seq[seq.num_cached_tokens:])
            print(f'{self.prefix_str} 请求{seq.seq_id},命中了{seq.num_cached_tokens}个token，请求的prompt长度为{seq.num_prompt_tokens}')
            positions.extend(list(range(seq.num_cached_tokens, seqlen)))
            if seq.input_embeds is not None:
                input_embeds.append(seq.input_embeds[seq.num_cached_tokens:,:])
            seqlen_q = seqlen - seq.num_cached_tokens
            seqlen_k = seqlen
            cu_seqlens_q.append(cu_seqlens_q[-1] + seqlen_q)
            cu_seqlens_k.append(cu_seqlens_k[-1] + seqlen_k)
            max_seqlen_q = max(seqlen_q, max_seqlen_q)
            max_seqlen_k = max(seqlen_k, max_seqlen_k)
            # 预热用的？
            if not seq.block_table:
                continue

            if self.role != 'master':
                
                for i in range(seq.num_cached_blocks):
                    cached_block_table.append(seq.block_table[i])
                for i in range(seq.num_cached_blocks, seq.num_blocks):
                    
                    start = seq.block_table[i] * self.config.hf_config.num_hidden_layers * self.block_size
                    # start = seq.block_table[i] * self.block_size
                    non_cached_block_table.append(seq.block_table[i])
                    if i != seq.num_blocks - 1:
                        end = start + self.block_size
                    else:
                        end = start + seq.last_block_num_tokens 
         
                    slot_mapping.extend(list(range(start, end)))
            # print(seq.block_table)
            # print(slot_mapping)
        # print(seqs[0].block_table)
        if self.role == 'master':

            cached_block_table, non_cached_block_table,local_cached_blocks, local_uncached_blocks, loaded_start_end, stored_start_end, slot_mapping, block_id_to_fix = self.set_local_block_tables(seqs,True,self.config)
 
        if self.cpu_kv_cache_manager is not None:
            cached_block_table_grouped = self.find_continuous_groups(cached_block_table)
            non_cached_block_table_grouped= self.find_continuous_groups(non_cached_block_table)
            self.cpu_kv_cache_manager.cached_block_table_grouped = cached_block_table_grouped
            self.cpu_kv_cache_manager.non_cached_block_table_grouped = non_cached_block_table_grouped
        
            # print(f'[prefill] cached_block_table{cached_block_table_grouped },non_cached_block_table:{non_cached_block_table_grouped}')
        # print(f'**********{block_table_cpu}')
        # print('prefill block_table',seq[0].block_table)
        if cu_seqlens_k[-1] > cu_seqlens_q[-1]:    # prefix cache
            block_tables= self.prepare_block_tables(seqs)
        # print(f"{self.prefix_str},{input_ids}")
        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_q = torch.tensor(cu_seqlens_q, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        cu_seqlens_k = torch.tensor(cu_seqlens_k, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_id_to_fix = torch.tensor(block_id_to_fix , dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        set_context(True, cu_seqlens_q, cu_seqlens_k, max_seqlen_q, max_seqlen_k, slot_mapping, None, block_tables, block_id_to_fix, self.cpu_kv_cache_manager)

        if self.role == 'master' and self.controller.kv_cache_initialized:
            # print('len(cached_block_table)',len(cached_block_table))
    
            self.controller.set_blocks_list(cached_block_table, non_cached_block_table, local_cached_blocks, local_uncached_blocks, loaded_start_end, stored_start_end)
            set_external_kvcache(self.controller)
        if len(input_embeds)>0:
            return input_ids, positions,torch.cat(input_embeds,dim=0)
        # print(f'{time.perf_counter()-t1}')
        return input_ids, positions,None

    def prepare_decode(self, seqs: list[Sequence]):
        # print("prepare decode")
        input_ids = []
        positions = []
        slot_mapping = []
        context_lens = []
        non_cached_block_table = []
        cached_block_table = []
        block_id_to_fix = []
        for seq in seqs:
            input_ids.append(seq.last_token)
            positions.append(len(seq))
            context_lens.append(len(seq))
            # slot_mapping.append(seq.block_table[-1]* self.block_size + seq.last_block_num_tokens  - 1)
            slot_mapping.append(seq.block_table[-1] * self.config.hf_config.num_hidden_layers * self.block_size + seq.last_block_num_tokens  - 1)
            cached_block_table.extend(seq.block_table)
            non_cached_block_table.append(seq.block_table[-1])
            # print(f'[decoding] seq.block_table:{seq.block_table},len:{len(seq.block_table)}')
            # print(f'[decoding]  cached_block_table{cached_block_table},non_cached_block_table:{non_cached_block_table}')
        if self.role == 'master':
            cached_block_table, non_cached_block_table,local_cached_blocks, local_uncached_blocks, loaded_start_end, stored_start_end, slot_mapping, block_id_to_fix = self.set_local_block_tables(seqs,False,self.config)
        if self.cpu_kv_cache_manager is not None:
            cached_block_table_grouped = self.find_continuous_groups(cached_block_table)
            non_cached_block_table_grouped= self.find_continuous_groups(non_cached_block_table)
            self.cpu_kv_cache_manager.cached_block_table_grouped = cached_block_table_grouped
            self.cpu_kv_cache_manager.non_cached_block_table_grouped = non_cached_block_table_grouped


        input_ids = torch.tensor(input_ids, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        positions = torch.tensor(positions, dtype=torch.int64, pin_memory=True).cuda(non_blocking=True)
        slot_mapping = torch.tensor(slot_mapping, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        context_lens = torch.tensor(context_lens, dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        block_id_to_fix = torch.tensor(block_id_to_fix , dtype=torch.int32, pin_memory=True).cuda(non_blocking=True)
        
        block_tables = self.prepare_block_tables(seqs)
        set_context(False, slot_mapping=slot_mapping, context_lens=context_lens, block_tables=block_tables, block_id_to_fix = block_id_to_fix, cpu_kv_cache_manager = self.cpu_kv_cache_manager)
        if self.role == 'master':
            self.controller.set_blocks_list(cached_block_table,non_cached_block_table,local_cached_blocks, local_uncached_blocks,loaded_start_end, stored_start_end)
            set_external_kvcache(self.controller)
        return input_ids, positions,None

    def prepare_sample(self, seqs: list[Sequence]):
        temperatures = []
        for seq in seqs:
            temperatures.append(seq.temperature)
        temperatures = torch.tensor(temperatures, dtype=torch.float32, pin_memory=True).cuda(non_blocking=True)
        return temperatures

    @torch.inference_mode()
    def run_model(self, input_ids: torch.Tensor, positions: torch.Tensor,input_embeds:torch.Tensor, is_prefill: bool,is_input_embeds: bool=False):
        if is_prefill or self.enforce_eager or input_ids.size(0) > 512:
            if is_input_embeds:
                return self.model.compute_logits(self.model(input_ids = input_ids, positions=positions,inputs_embeds=input_embeds))
            else:
                return self.model.compute_logits(self.model(input_ids, positions))
        else:
    
            bs = input_ids.size(0)
            context = get_context()
            # print('next(x for x in self.graph_bs if x >= bs)',next(x for x in self.graph_bs if x >= bs))
            graph = self.graphs[next(x for x in self.graph_bs if x >= bs)]
            graph_vars = self.graph_vars
            for k, v in graph_vars.items():
                if k != "outputs":
                    v.zero_()
            graph_vars["input_ids"][:bs] = input_ids
            graph_vars["positions"][:bs] = positions 
            graph_vars["slot_mapping"][:bs] = context.slot_mapping
            graph_vars["context_lens"][:bs] = context.context_lens
            graph_vars["block_tables"][:bs, :context.block_tables.size(1)] = context.block_tables
            graph.replay()
            return self.model.compute_logits(graph_vars["outputs"][:bs])

    def run(self, seqs: list[Sequence], is_prefill: bool) -> list[int]:
        self.cal_num_blocks_per_slave(seqs,self.config)
        t1 = time.perf_counter()
        input_ids, positions,input_embeds = self.prepare_prefill(seqs) if is_prefill else self.prepare_decode(seqs)
        # print('input_ids:',input_ids.shape)
        temperatures = self.prepare_sample(seqs) if self.rank == self.first_rank else None
        logits = self.run_model(input_ids, positions,input_embeds, is_prefill,input_embeds is not None)
        range_push("sample")

        token_ids = self.sampler(logits, temperatures) if self.rank == self.first_rank else None
        token_ids = token_ids.tolist()
        # print('run time:',time.perf_counter() - t1)
        range_pop()
        reset_context()
        return token_ids

    @torch.inference_mode()
    def capture_cudagraph(self):
        config = self.config
        hf_config = self.hf_language_config
        max_bs = min(self.config.max_num_seqs, 512)
        max_num_blocks = (config.max_model_len + self.block_size - 1) // self.block_size
        input_ids = torch.zeros(max_bs, dtype=torch.int64)
        positions = torch.zeros(max_bs, dtype=torch.int64)
        slot_mapping = torch.zeros(max_bs, dtype=torch.int32)
        context_lens = torch.zeros(max_bs, dtype=torch.int32)
        block_tables = torch.zeros(max_bs, max_num_blocks, dtype=torch.int32)
        outputs = torch.zeros(max_bs, hf_config.hidden_size)

        print(f'{self.prefix_str} Capture start')
        self.graph_bs = [1, 2, 4, 8] + list(range(16, max_bs + 1, 16))
        self.graphs = {}
        self.graph_pool = None
        for bs in reversed(self.graph_bs):
            graph = torch.cuda.CUDAGraph()
            set_context(False, slot_mapping=slot_mapping[:bs], context_lens=context_lens[:bs], block_tables=block_tables[:bs])
            outputs[:bs] = self.model(input_ids[:bs], positions[:bs]) # warmup
            with torch.cuda.graph(graph, self.graph_pool):
                outputs[:bs] = self.model(input_ids[:bs], positions[:bs])    # capture
            if self.graph_pool is None:
                self.graph_pool = graph.pool()
            self.graphs[bs] = graph
            torch.cuda.synchronize()
            reset_context()

        self.graph_vars = dict(
            input_ids=input_ids,
            positions=positions,
            slot_mapping=slot_mapping,
            context_lens=context_lens,
            block_tables=block_tables,
            outputs=outputs,
        )
        print('cuda graph capture successed!')


    def init_model(self, hf_config: AutoConfig):
        match hf_config.model_type:
            case 'qwen3':
                from swiftcache.models.qwen3 import Qwen3ForCausalLM
                return Qwen3ForCausalLM(hf_config)
            
            case 'llama':
                from swiftcache.models.llama3 import Llama3ForCausalLM
                return Llama3ForCausalLM(hf_config)
            
            case 'deepseek_vl_v2':
                from swiftcache.models.deepseek import DeepSeekVLV2ForCausalLM
                return DeepSeekVLV2ForCausalLM(hf_config)

            case _:
                raise ValueError(
                    f"Unsupported model type: '{hf_config.model_type}'. "
                    f"Supported model types are: 'qwen3', 'llama', 'deepseek'"
                )

    @torch.inference_mode()
    def image_process(self,seq: VisionSequence):
        prepare_inputs = self.vl_chat_processor(
        conversations=seq.conversation,
        images=seq.pil_images,
        force_batchify=True,
        system_prompt="")

        return prepare_inputs

    @torch.inference_mode()
    def prepare_inputs_embeds(self, prepare_inputs):
        return self.model.prepare_inputs_embeds(**prepare_inputs)

    def find_continuous_groups(self,nums: list[int]):
        if not nums:
            return []

        num_set = set(nums)  # O(n) 构建集合
        visited = set()
        result = []

        for num in nums:  # O(n)
            if num in visited:
                continue
            
            # 找连续段的起始
            if num - 1 not in num_set:  # 说明是起点
                length = 1
                visited.add(num)
                next_num = num + 1
                while next_num in num_set:  # 找连续段
                    visited.add(next_num)
                    length += 1
                    next_num += 1

                result.append({
                    "start": num,
                    "length": length
                })

        return result

    def record_pid(self):
        pid_dict = {'pid':self.pid}
        file_path = global_config['os_pid_dir']+f'rank{self.rank}.json'
        with open(file_path, 'w') as f:
            json.dump(pid_dict, f) 