import os
import pickle
import torch
import torch.distributed as dist
import triton
import triton.language as tl
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import atexit
import time
import json
from swiftcache.global_config import global_config
from torch.cuda.nvtx import range_push,range_pop
import threading
import glob

@triton.jit
def store_kvcache_kernel(
    key_ptr, #[len(blocks_stored_list),block_size * num_heads * head_dim]
    value_ptr, #[len(blocks_stored_list),block_size * num_heads * head_dim]
    k_cache_ptr,  #[num_kvcache_blocks,self.block_elem_count]
    v_cache_ptr, #[num_kvcache_blocks,self.block_elem_count]
    blocks, #[len(blocks_stored_list)]
    block_elem_count: tl.constexpr,
    block_elem_count_pow2: tl.constexpr,
):
    idx = tl.program_id(0)
    mask = tl.arange(0,block_elem_count_pow2)<block_elem_count
    kv_offsets = idx*block_elem_count+tl.arange(0,block_elem_count_pow2)
    key = tl.load(key_ptr+kv_offsets,mask=mask)
    value = tl.load(value_ptr+kv_offsets,mask=mask)
    block_idx = tl.load(blocks+idx)
    cache_offsets = block_idx*block_elem_count+tl.arange(0,block_elem_count_pow2)
    tl.store(k_cache_ptr+cache_offsets,key,mask = mask)
    tl.store(v_cache_ptr+cache_offsets,value, mask = mask)

def store_kvcache(key:torch.Tensor,value:torch.Tensor,k_cache:torch.Tensor,v_cache:torch.Tensor,blocks:torch.Tensor,block_elem_count:int):
    """
    将连续的kv根据block索引分散存储在kv_cache中
    """
    assert key.is_contiguous()
    assert value.is_contiguous()
    N = blocks.shape[0]
    assert key.shape[0] == N and value.shape[0] == N
    block_elem_count_pow2 = triton.next_power_of_2(block_elem_count)
    store_kvcache_kernel[(N,)](key,value,k_cache,v_cache,blocks,block_elem_count,block_elem_count_pow2)

@triton.jit
def load_kvcache_kernel(
    key_ptr, #[len(blocks_stored_list),block_size * num_heads * head_dim]
    value_ptr, #[len(blocks_stored_list),block_size * num_heads * head_dim]
    k_cache_ptr,  #[num_kvcache_blocks,self.block_elem_count]
    v_cache_ptr, #[num_kvcache_blocks,self.block_elem_count]
    blocks, #[len(blocks_stored_list)]
    block_elem_count: tl.constexpr,
    block_elem_count_pow2: tl.constexpr,
):
    idx = tl.program_id(0)
    mask = tl.arange(0,block_elem_count_pow2)<block_elem_count
    kv_offsets = idx*block_elem_count+tl.arange(0,block_elem_count_pow2)
    # key = tl.load(key_ptr+kv_offsets,mask=mask)
    # value = tl.load(value_ptr+kv_offsets,mask=mask)
    block_idx = tl.load(blocks+idx)
    cache_offsets = block_idx*block_elem_count+tl.arange(0,block_elem_count_pow2)
    k_cache = tl.load(k_cache_ptr+cache_offsets,mask = mask)
    v_cache = tl.load(v_cache_ptr+cache_offsets,mask = mask)
    tl.store(key_ptr+kv_offsets,k_cache,mask = mask)
    tl.store(value_ptr+kv_offsets,v_cache, mask = mask)

def load_kvcache(key:torch.Tensor,value:torch.Tensor,k_cache:torch.Tensor,v_cache:torch.Tensor,blocks:torch.Tensor,block_elem_count:int):
    """
    将存储在kvcache中离散的kv加载到连续的GPU内存中
    """
    assert key.is_contiguous()
    assert value.is_contiguous()
    N = blocks.shape[0]
    assert key.shape[0] == N and value.shape[0] == N
    block_elem_count_pow2 = triton.next_power_of_2(block_elem_count)
    load_kvcache_kernel[(N,)](key,value,k_cache,v_cache,blocks,block_elem_count,block_elem_count_pow2)

class ExternalKVCacheManager:
    def __init__(self,rank,event:Event,ready_event:Event,master_list,slave_list,dist_port = 2334):
        pid = os.getpid()
        self.pid = pid
        self.rank = rank
        # self.record_pid()
        self.offset = 0
        self.recieved_master_kvcache_info = False
        self.prefix_str = f'[PID:{pid},RANK:{rank} Slave]'
        print(f'{self.prefix_str}正在初始化exernal kv cache')
        self.master_list =  master_list
        self.slave_list = slave_list
        self.kv_group_list = master_list+slave_list
        world_size = len(self.kv_group_list)
        self.event = event 
        self.ready_event = ready_event   
        # torch.cuda.set_device(rank)
        torch.set_default_device(f"cuda:{self.rank}")
        torch.cuda.set_device(self.rank)
        torch.set_default_dtype(torch.bfloat16)
        self.send_stream = torch.cuda.Stream()
        self.recv_stream = torch.cuda.Stream()
        self.blocks_loaded_list = None
        self.blocks_stored_list = None
        self.blocks_loaded = None
        self.blocks_stored = None
        self.blocks_stored_buffer = None
        # 每个 block 的元素个数
        print(f'{self.prefix_str}等待连接 world_size:{world_size}')
        dist.init_process_group("nccl", f"tcp://localhost:{dist_port}", world_size=world_size, rank=rank)
        self.kv_group = dist.new_group(ranks = self.kv_group_list )
        dist.barrier(self.kv_group)
        self.actions = []
        self.kv_shm = SharedMemory(name="kvcache")
        self.ready_event.set()
        print(f'{self.prefix_str} External GPU Manager 初始化成功')
        threading.Thread(target=self.loop, daemon=True).start()

    
    def call(self, method_name, *args):
        method = getattr(self, method_name, None)
        return method(*args)
    
    def read_shm(self):
        # print(f"{self.prefix_str} wait start")
        self.event.wait()
        # print(f"{self.prefix_str} wait end")
        n = int.from_bytes(self.kv_shm.buf[0:4], "little")
        result = pickle.loads(self.kv_shm.buf[4:n+4])
        method_name, *args = result.get(f'rank{self.rank}')
        self.event.clear()
        return method_name, args

    def exit(self):
        self.kv_shm.close()
        dist.barrier(self.kv_group)
        torch.cuda.synchronize()
        dist.destroy_process_group()

    def loop(self):
        torch.cuda.set_device(self.rank)
        torch.set_default_device(f"cuda:{self.rank}")
        while True:
            try:
                method_name, args = self.read_shm()
                self.call(method_name, *args)
                # print(f"{self.prefix_str} {method_name} set")
                self.ready_event.set()
                if method_name == "exit":
                    print(f'{self.prefix_str} {self.prefix_str} exits')
                    break
            except Exception as e:
                print(f"[rank {self.rank}] Error in loop: {e}")
                import traceback
                traceback.print_exc()
                exit(1)


    def func1(self,param):
        print(f'{self.prefix_str},调用测试方法func1,参数为{param}')
    
    def set_offset(self,offset:int = 0):
        self.offset = offset

    def set_blocks_list(self, blocks_loaded_subset, blocks_stored_subset):
        self.blocks_loaded_list = blocks_loaded_subset
        self.blocks_stored_list = blocks_stored_subset
        # print(f"{self.prefix_str}, blocks_loaded_list:{self.blocks_loaded_list},blocks_stored_list:{self.blocks_stored_list}")
        for i in range(len(self.blocks_loaded_list)):
            self.blocks_loaded_list[i] = self.blocks_loaded_list[i] - self.offset
        for i in range(len(self.blocks_stored_list)):
            self.blocks_stored_list[i] = self.blocks_stored_list[i] - self.offset
        max_block_id_loaded = max(self.blocks_loaded_list) if len(self.blocks_loaded_list)>0 else 0
        max_block_id_stored = max(self.blocks_stored_list) if len(self.blocks_stored_list)>0 else 0
        assert max_block_id_loaded < self.num_kvcache_blocks
        assert max_block_id_stored < self.num_kvcache_blocks
        self.blocks_loaded = torch.tensor(self.blocks_loaded_list,dtype = torch.int64, pin_memory = True).cuda(non_blocking=True)
        self.blocks_stored = torch.tensor(self.blocks_stored_list,dtype = torch.int64, pin_memory = True).cuda(non_blocking=True)
        self.blocks_stored_buffer  = torch.empty(2,len(self.blocks_stored_list),self.block_size,self.num_kv_heads,self.head_dim,dtype=torch.bfloat16)
        # print("xxxxxx set_blocks_list",self.blocks_stored_buffer.device,torch.get_default_device(),torch.cuda.current_device())
        if len(self.blocks_loaded_list) > 0 and len(self.blocks_stored_list)>0:
            self.kvcache_management()
        elif len(self.blocks_loaded_list) > 0:
            self.send_only()
        elif len(self.blocks_stored_list) > 0:
            self.recv_only()

    def send(self,layer,dst = 0):
        with torch.cuda.stream(self.send_stream):
            range_push(f'send:{layer}')
            # print("xxxxxx",self.k_cache.device, self.blocks_loaded.device)
            # print(f'{self.prefix_str},layer:{layer},self.blocks_loaded:{self.blocks_loaded}')
            
            k_cache = self.k_cache[layer,self.blocks_loaded,:]
            # print(f'{self.prefix_str} {layer},send k',k_cache.shape)
            dist.send(k_cache,dst = dst)
            v_cache = self.v_cache[layer,self.blocks_loaded,:]
            # print(f'{self.prefix_str} {layer},send v',v_cache.shape)
            dist.send(v_cache,dst = dst)
            range_pop()

        # if(layer == 1):
        #     print(k_cache.shape)
    def recv(self,layer,src = 0):
        with torch.cuda.stream(self.recv_stream): 
            range_push(f'recv:{layer}')
            # print(f'{self.prefix_str} {layer},recv k')
            dist.recv(self.blocks_stored_buffer[0],src = src) #recv k
            # print(f'{self.prefix_str} {layer},recv v')
            dist.recv(self.blocks_stored_buffer[1],src = src) #recv v
            range_pop()
            k_cache = self.k_cache[layer]
            v_cache = self.v_cache[layer]
            store_kvcache(self.blocks_stored_buffer[0],self.blocks_stored_buffer[1],k_cache,v_cache,self.blocks_stored,self.block_elem_count)


    def send_full_kvcache(self,layer,dst = 0):
        dist.send(self.k_cache[layer],dst = dst)
        dist.send(self.v_cache[layer],dst = dst)
        torch.cuda.synchronize()
    def recv_full_kvcache(self,layer,src = 0):
        dist.recv(self.k_cache[layer],src = src)
        dist.recv(self.v_cache[layer],src = src) #recv v
        torch.cuda.synchronize()

    def init_kvcache(self,num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim,offset):
        print(f'{self.prefix_str} 开始初始化KV')
        t = time.perf_counter()
        self.num_hidden_layers = num_hidden_layers
        self.num_kvcache_blocks = num_kvcache_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_elem_count = block_size * num_kv_heads * head_dim
        self.offset = offset
        # 总长度: 2 是 K & V
        self.total_length = (2 * num_hidden_layers *
                             num_kvcache_blocks *
                             self.block_elem_count)
        self.shape = (2,num_hidden_layers,num_kvcache_blocks,self.block_size,self.num_kv_heads,self.head_dim)
        print(f'{self.prefix_str}',self.shape)
        print(f'self.total_length:{self.total_length},{self.total_length*2/(1024*1024*1024)} GB')
        # self.memory = torch.empty(self.total_length, dtype=torch.bfloat16, device = 'cpu',pin_memory=True)
        self.memory = torch.empty(self.shape, dtype=torch.bfloat16).cuda()
        self.k_cache = self.memory[0] #[num_hidden_layers, num_kvcache_blocks,self.block_elem_count]
        self.v_cache = self.memory[1] #[num_hidden_layers, num_kvcache_blocks,self.block_elem_count]

        print(f'{self.prefix_str} KVCache initialized. Time Usage:{(time.perf_counter()-t):.2f}s')
    
    def recv_master_kvcache_info(self, num_hidden_layers, block_size, num_kv_heads, head_dim):
        self.num_hidden_layers = num_hidden_layers
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.block_elem_count = block_size * num_kv_heads * head_dim
        self.recieved_master_kvcache_info = True
        self.master_block_element_all_layers = self.block_elem_count * self.num_hidden_layers
        print(f'{self.prefix_str} received kvcache info:({num_hidden_layers},{block_size},{num_kv_heads}, {head_dim})')

    def init_kvcache_with_existing_tensor(self,k_cache, v_cache):
        RED = "\033[91m"      # 亮红色
        RESET = "\033[0m"     # 重置为默认颜色
        self.k_cache = k_cache
        self.v_cache = v_cache
        print(f'{self.prefix_str} {RED}numel:{k_cache.numel()//(self.block_size * self.num_kv_heads * self.head_dim)}{RESET}')
        self.num_kvcache_blocks = k_cache.numel()//(self.num_hidden_layers * self.block_size * self.num_kv_heads * self.head_dim)
        total_elem = self.num_kvcache_blocks * self.num_hidden_layers * self.block_size * self.num_kv_heads * self.head_dim
        self.k_cache = k_cache.view(-1)[:total_elem].view(self.num_hidden_layers, self.num_kvcache_blocks, self.block_size, self.num_kv_heads, self.head_dim)
        self.v_cache = v_cache.view(-1)[:total_elem].view(self.num_hidden_layers, self.num_kvcache_blocks, self.block_size, self.num_kv_heads, self.head_dim)
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        # print("xxxxxx init_kvcache_with_existing_tensor",torch.cuda.current_device())
        print(f'{self.prefix_str} external_blocks_num:{self.num_kvcache_blocks}')
        message_file = f'{script_dir}/num_blocks_{self.rank}.json'
        with open(message_file, 'w') as f:
            json.dump({self.rank:self.num_kvcache_blocks}, f) 
        


    def kvcache_management(self):
        if global_config['transfer_full_kvcache']:
            for layer_id in range(self.num_hidden_layers):
                self.send_full_kvcache(layer_id,0)
                self.recv_full_kvcache(layer_id,0)

        else:
            for layer_id in range(self.num_hidden_layers):
                self.send(layer_id,0)
                self.recv(layer_id,0)
    def send_only(self):
        for layer_id in range(self.num_hidden_layers):
                self.send(layer_id,0)
    
    def recv_only(self):
        for layer_id in range(self.num_hidden_layers):
            self.recv(layer_id,0)
    
    def record_pid(self):
        pid_dict = {'pid':self.pid}
        file_path = global_config['os_pid_dir']+f'rank{self.rank}.json'
        with open(file_path, 'w') as f:
            json.dump(pid_dict, f) 

class ExternalKVCacheController:
    def __init__(self,
                rank,
                slave_events,
                ready_events,
                master_list,
                slave_list,
                dist_port
                ):
        self.pid = os.getpid()
        print(f'ExternalKVCache PID {self.pid}')
        self.clear_json_file()
        self.rank = rank
        self.prefix_str=f'[PID:{self.pid} RANK:{self.rank} Master]'
        self.clear_shared_memory()
        self.events = slave_events
        self.ready_events = ready_events
        self.ps = {}
        self.master_list = master_list
        self.slave_list = slave_list
        self.kv_group_list = master_list + slave_list
        self.world_size = len(self.kv_group_list)
        self.need_load = False
        self.need_store = False
        torch.set_default_dtype(torch.bfloat16)
        self.kv_cache_initialized = False
        print(f'master_list:{self.master_list}, slave_list:{self.slave_list}')
        print(f'{self.prefix_str}等待连接')
        dist.init_process_group("nccl", f"tcp://localhost:{dist_port}", world_size=self.world_size, rank=self.rank)
        self.kv_group = dist.new_group(ranks=self.kv_group_list)
        if len(self.ps)==0:
            print(f'{self.prefix_str}, ExternalKVCacheController is not initialized!')
        self.create_shared_memory()
        print(f'{self.prefix_str},初始化成功！')

    def call(self,slave_ids, arguments):
        data = pickle.dumps(arguments)
        n = len(data)
        self.kv_shm.buf[0:4] = n.to_bytes(4, "little")
        self.kv_shm.buf[4:n+4] = data
        for slave_id in slave_ids:
            # print(f"{self.prefix_str} wait before")
            self.ready_events[slave_id].wait() #只有上一次call执行完毕了才能执下一次call
            # print(f"{self.prefix_str} wait after")
            event = self.events[slave_id]
            # print(type(self.events[slave_id]))
            self.ready_events[slave_id].clear()
            event.set()
            # print(f"{self.prefix_str} end")
            
    
    def wait_last_event(self,slave_ids=None):
        if slave_ids is None:
            slave_ids = self.slave_list
        for slave_id in slave_ids:
            self.ready_events[slave_id].wait()

    def init_kvcache(self,num_hidden_layers, num_kvcache_blocks_start_end, block_size, num_kv_heads, head_dim):
        # 这里后面可能还需要修改，因为不同slave的kvcache未必相同
        assert len(num_kvcache_blocks_start_end) == len(self.slave_list)+1
        print("打算初始化所有external kv")
        self.block_elem_count = block_size * num_kv_heads * head_dim
        self.num_hidden_layers = num_hidden_layers
        args = {}
        for i in range(len(self.slave_list)):
            slave_id = self.slave_list[i]
            num_kvcache_blocks = num_kvcache_blocks_start_end[i+1] - num_kvcache_blocks_start_end[i]
            offset = num_kvcache_blocks_start_end[i]
            args[f'rank{slave_id}']=('init_kvcache',num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim,offset)
        
        self.call(self.slave_list,args)
        self.kv_cache_initialized = True
        print("已经初始化所有external kv")
    # def init_kvcache_with_existing_tensor(self, num_hidden_layers, block_size, num_kv_heads, head_dim):
    #     args = {}
    #     for slave_id in self.slave_list:
    #         args[f'rank{slave_id}']=('init_kvcache_with_existing_tensor',num_hidden_layers, block_size, num_kv_heads, head_dim)
    
    def send_master_kvcache_info(self, num_hidden_layers, block_size, num_kv_heads, head_dim):
        self.block_elem_count = block_size * num_kv_heads * head_dim
        self.num_hidden_layers = num_hidden_layers
        args = {}
        print(f"send_master_kvcache_info,{self.slave_list}")
        for slave_id in self.slave_list:
            args[f'rank{slave_id}']=('recv_master_kvcache_info',num_hidden_layers, block_size, num_kv_heads, head_dim)
        self.call(self.slave_list,args)
    def send_slave_offset(self,offset_list:list[int]):
        args = {}
        idx = 0
        for slave_id in self.slave_list:
            args[f'rank{slave_id}'] = ('set_offset',offset_list[idx])
            idx+=1
        self.call(self.slave_list,args)
    def set_local_kvcache(self,k_cache,v_cache):
        self.k_cache = k_cache
        self.v_cache = v_cache

    def set_blocks_list(self,blocks_loaded_list, blocks_stored_list, local_blocks_loaded_list, local_blocks_stored_list, loaded_start_end, stored_start_end):
        
        assert len(self.slave_list)+1 == len(loaded_start_end) and len(self.slave_list)+1 == len(stored_start_end)
        # print(f'load block number:{len(local_blocks_loaded_list)}')
        self.local_blocks_loaded_list = local_blocks_loaded_list
        self.local_blocks_stored_list = local_blocks_stored_list
        self.need_load = len(local_blocks_loaded_list)>0
        self.need_store = len(local_blocks_stored_list) > 0
        # print(blocks_loaded_list, blocks_stored_list, self.need_load,self.need_store)
        self.loaded_start_end = loaded_start_end
        self.stored_start_end = stored_start_end
        # self.blocks_loaded_buffer  = torch.empty(2,len(blocks_loaded_list), self.block_elem_count).cuda()
        self.blocks_stored_buffer  = torch.empty(2,len(blocks_stored_list), self.block_elem_count,device=f'cuda:{self.rank}')
        self.local_blocks_stored = torch.tensor(local_blocks_stored_list,dtype = torch.int64, pin_memory=True).cuda(non_blocking=True)
        self.slave_info = {}
        args = {}
        blocks_loaded_sum = 0
        blocks_stored_sum = 0

        for i in range(len(self.slave_list)):
            slave_id = self.slave_list[i]
            num_blocks_loaded = loaded_start_end[i+1]-loaded_start_end[i]
            # print(f"slave{i} need to load block:{num_blocks_loaded}")
            blocks_loaded_sum += num_blocks_loaded
            load_start = loaded_start_end[i]
            local_load_start =  local_blocks_loaded_list[loaded_start_end[i]] if num_blocks_loaded > 0 else 0
            blocks_loaded_subset = blocks_loaded_list[load_start:load_start+num_blocks_loaded]
            store_start = stored_start_end[i]
            num_blocks_stored = stored_start_end[i+1]-stored_start_end[i]
            blocks_stored_sum += num_blocks_stored
            # print(f"slave{i} need to store block:{num_blocks_stored}")
            blocks_stored_subset = blocks_stored_list[store_start:store_start+num_blocks_stored]
            # print('blocks_loaded_subset',len(blocks_loaded_subset),num_blocks_loaded,load_start,local_blocks_loaded_list)
            args[f'rank{slave_id}'] = ('set_blocks_list',blocks_loaded_subset,blocks_stored_subset)
            self.slave_info[slave_id] = {
            'load': {
                'start': local_load_start,
                'length': num_blocks_loaded,
            },
            'store': {
                'start': store_start,
                'length': num_blocks_stored,
                'local_blocks_stored': self.local_blocks_stored[store_start:store_start+num_blocks_stored]
            }
            }
        self.call(self.slave_list,args)

    
    def gather_info_from_slave(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)
        while True:
            json_files = glob.glob(os.path.join(script_dir, "*.json"))
            json_count = len(json_files)
            print(f'{self.prefix_str} json_count:{json_count}, len(self.slave_list):{len(self.slave_list)}')
            if json_count == len(self.slave_list):
                break  # 数量够了，跳出循环
            else:
                time.sleep(1)  # 不够就等 0.01 秒重新检查
            # json_files = glob.glob(os.path.join(script_dir, "*.json"))

        # 2. 读取内容到字典
        slave_blocks = {}
        for file_path in json_files:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                # 每个 JSON 只有一对 key-value
                for k, v in data.items():
                    slave_blocks[int(k)] = int(v)

        # 3. 按 slave_id 排序
        sorted_slaves = sorted(slave_blocks.items(), key=lambda x: x[0])

        # 4. 前缀和
        num_blocks_start_end = [0]
        total = 0
        for _, block_num in sorted_slaves:
            total += block_num
            num_blocks_start_end.append(total)
        GREEN = "\033[92m"    # 亮绿色
        RED = "\033[91m"      # 亮红色
        RESET = "\033[0m"     # 重置为默认颜色
        print(f'{GREEN}sorted_slaves:{sorted_slaves}, num_blocks_start_end:{num_blocks_start_end}{RESET}')
        self.kv_cache_initialized = True
        return sorted_slaves, num_blocks_start_end

    def clear_json_file(self):
        script_path = os.path.abspath(__file__)
        script_dir = os.path.dirname(script_path)

        # 找到该目录下所有的 json 文件
        json_files = glob.glob(os.path.join(script_dir, "*.json"))

        # 删除每个文件
        for file_path in json_files:
            try:
                os.remove(file_path)
                print(f"已删除: {file_path}")
            except Exception as e:
                print(f"删除失败 {file_path}: {e}")
                


    def load_kvcache(self,k_cache,v_cache,layer_id):
        # print(f"{self.prefix_str} load start {layer_id}")
        # print(f'{self.prefix_str} layer:{layer_id}',k_cache.shape)
        for slave_id in self.slave_list:
            if global_config['transfer_full_kvcache']:
                dist.recv(k_cache,src = slave_id)
                dist.recv(v_cache,src = slave_id)
            else:
                range_push(f'load:{layer_id}')
                info = self.slave_info[slave_id]['load']
                start,length = info['start'],info['length']
                if length > 0:
                    # print(f'from slave_id {slave_id}',k_cache[start:start+length].shape)
                    # print(f'{self.prefix_str} {layer_id},load k',k_cache[start:start+length].shape)
                    dist.recv(k_cache[start:start+length],src = slave_id)
                    # print(f'slave_id{slave_id}',k_cache[start:start+length])
                    # print(f'{self.prefix_str} {layer_id},load v',v_cache[start:start+length].shape)
                    dist.recv(v_cache[start:start+length],src = slave_id)
                    # print(f'{self.prefix_str} {layer_id},load end')
                range_pop()
        
        # blocks_stored_buffer[0] ---> k_cache, blocks_stored_buffer[1]---> v_cache
        # store_kvcache(self.blocks_loaded_buffer[0],self.blocks_loaded_buffer[1],k_cache,v_cache,self.blocks_loaded,self.block_elem_count)
        # print(f"{self.prefix_str} load end {layer_id}")
    def store_kvcache(self,k_cache,v_cache,layer_id):
        for slave_id in self.slave_list:

            if global_config['transfer_full_kvcache']:
                dist.send(k_cache,dst = slave_id)
                dist.send(v_cache,dst = slave_id)
            else:
                range_push(f'store:{layer_id}')
                info = self.slave_info[slave_id]['store']
                length = info['length']
                if length >0 :
                    local_blocks_stored  = info['local_blocks_stored']
                    # print(f'to slave_id {slave_id}, local_blocks_stored',k_cache[local_blocks_stored].shape)
                    # print(f'{self.prefix_str} {layer_id} ,store k')
                    dist.send(k_cache[local_blocks_stored],dst = slave_id)
                    # print(f'{self.prefix_str} {layer_id},store v')
                    dist.send(v_cache[local_blocks_stored],dst = slave_id)
                range_pop()

        # print(f"{self.prefix_str} store end {layer_id}")
    # def start_external_kvcache_management(self):
    #     self.call(self.slave_list,'kvcache_management')
    def create_shared_memory(self):
        if self.rank == self.master_list[0]:
            self.kv_shm = SharedMemory(name="kvcache", create=True, size=2**20)
            dist.barrier(self.kv_group)
        else:
            dist.barrier(self.kv_group)
            self.kv_shm = SharedMemory(name="kvcache")
    
    def clear_shared_memory(self):
        try:
            existing_shm = SharedMemory(name="kvcache")
            existing_shm.close()
            existing_shm.unlink()
            print(f"{self.prefix_str} 清理了旧的共享内存")
            time.sleep(0.1)  # 给系统一些时间
            return True
        except FileNotFoundError:
            print(f"共享内存 kvcache 不存在，无需清理")
            return False

    
    @property
    def tp_size(self):
        return len(self.master_list)
    
    def exit(self):
        args = {}
        for slave_id in self.slave_list:
            args[f'rank{slave_id}'] = ('exit',)
        self.call(self.slave_list,args)
        self.kv_shm.close()
        dist.barrier(self.kv_group)
        if self.rank == self.master_list[0]:
            print(f'{self.prefix_str},unlink shm!')
            self.kv_shm.unlink()


if __name__ == '__main__':
    #num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim
    num_hidden_layers = 10
    num_kvcache_blocks = 10
    block_size = 256
    num_kv_heads = 8
    head_dim = 128
    kv_cache = torch.ones(2,num_kvcache_blocks,block_size*num_kv_heads*head_dim, dtype=torch.bfloat16).cuda()
    kv_cache[0][0].fill_(3)
    kv_cache[1][0].fill_(3)
    print('传输前的key',kv_cache[0])
    print('传输前的value',kv_cache[1])
    rank = 0
    master_list = [0]
    slave_list = [1]
    c = ExternalKVCacheController(rank,master_list,slave_list)
    c.call([1],'func1','你好')
    time.sleep(0.5)
    c.init_kvcache(num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim)
    time.sleep(0.5)
    blocks_loaded_list = [0,5,6]
    blocks_stored_list = [0]
    loaded_start_end = [0,3]
    stored_start_end = [0,1]
    c.set_blocks_list(blocks_loaded_list, blocks_stored_list, loaded_start_end, stored_start_end)
    c.store_kvcache(kv_cache[0],kv_cache[1],0)
    c.load_kvcache(kv_cache[0],kv_cache[1],0)

    c.exit()
    torch.cuda.synchronize()
    dist.destroy_process_group()