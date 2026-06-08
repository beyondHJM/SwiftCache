import torch

class CpuKVCacheManager:
    def __init__(self, num_hidden_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim):
        """
        初始化一个 CPU KV Cache 管理类（使用 pinned memory 方便异步 GPU copy）

        :param num_hidden_layers: 模型层数
        :param num_kvcache_blocks: KV Cache 分块数量
        :param block_size: 每个 block 的 token 数
        :param num_kv_heads: 注意力头数量
        :param head_dim: 每个注意力头的维度
        """
        print('正在初始化cpu kv cache')
        self.num_hidden_layers = num_hidden_layers
        self.num_kvcache_blocks = num_kvcache_blocks
        self.block_size = block_size
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

        # 每个 block 的元素个数
        self.block_elem_count = block_size * num_kv_heads * head_dim

        # 总长度: 2 是 K & V
        self.total_length = (2 * num_hidden_layers *
                             num_kvcache_blocks *
                             self.block_elem_count)
        print(f'self.total_length:{self.total_length},{self.total_length*2/(1024*1024*1024)} GB')
        # 用 pinned memory 分配一维连续张量（CPU锁页内存）
        self.memory = torch.empty(self.total_length, dtype=torch.bfloat16, device = 'cpu',pin_memory=True)
        self.cached_block_table_grouped = None
        self.non_cached_block_table_grouped = None
        print('cpu kv cache 初始化成功')
    def _calc_block_offset(self, layer_id, block_id, kv_type="k"):
        """
        计算某一层某个 block 在内存中的起始偏移
        kv_type: "k" 或 "v"
        """
        if kv_type not in ("k", "v"):
            raise ValueError("kv_type 必须是 'k' 或 'v'")
        kv_offset = 0 if kv_type == "k" else self.total_length // 2

        # 每层占用的总数（包括所有blocks的 K 或 V）
        per_layer_size = self.num_kvcache_blocks * self.block_elem_count

        offset = kv_offset + layer_id * per_layer_size + block_id * self.block_elem_count
        return int(offset)

    def write_block(self, layer_id, block_id, kv_type, data):
        """
        写入一个 block 的 KV 数据
        :param data: list[float] 或 1D tensor（长度须等于 block_elem_count）
        """
        if len(data) != self.block_elem_count:
            raise ValueError(f"数据长度必须是 {self.block_elem_count}，实际是 {len(data)}")

        offset = self._calc_block_offset(layer_id, block_id, kv_type)

        if isinstance(data, list):
            # 避免创建大 tensor，用 view+逐元素赋值
            view = self.memory[offset:offset+self.block_elem_count]
            for i, v in enumerate(data):
                view[i] = v
        elif isinstance(data, torch.Tensor):
            self.memory[offset:offset+self.block_elem_count].copy_(data, non_blocking=False)
        else:
            raise TypeError("data 必须是 list 或 torch.Tensor")

    def read_block(self, layer_id, block_id, kv_type):
        """
        读取一个 block 的 KV 数据（返回一个 pinned memory tensor 的视图）
        """
        offset = self._calc_block_offset(layer_id, block_id, kv_type)
        return self.memory[offset:offset+self.block_elem_count]

    def load_kvcache(self, request_blocks_grouped, gpu_kv_cache, kv_type,layer_id):
        """
        按 request 的 block 列表拷贝到 GPU KV Cache
        :param request_blocks: [(layer_id, block_id, kv_type), ...]
        :param gpu_kv_cache: GPU 端的同形状连续张量
        gpu_kv_cache shape:[num_kvcache_blocks, block_size, num_kv_heads,head_dim]

        """
        for item in request_blocks_grouped:
            block_id = item['start']
            length = item['length']
            cpu_offset = self._calc_block_offset(layer_id, block_id, kv_type)
            # gpu_offset = self._calc_block_offset(0, block_id, kv_type)
            gpu_offset = block_id * self.block_elem_count
            cpu_slice = self.memory[cpu_offset:cpu_offset+self.block_elem_count*length]
            gpu_slice = gpu_kv_cache[block_id:block_id+length].view(-1)
            # print("_____",cpu_slice.shape, gpu_slice.shape)
            # 直接异步拷贝 pinned memory → GPU
            gpu_slice.copy_(cpu_slice, non_blocking=True)

    def store_kvcache(self, request_blocks_grouped, gpu_kv_cache, kv_type,layer_id):
        """
        按 request 的 block 列表，从 GPU 拷贝回 CPU pinned memory
        :param request_blocks: [(layer_id, block_id, kv_type), ...]
        :param gpu_kv_cache: GPU 端的同形状连续张量
        gpu_kv_cache shape:[num_kvcache_blocks, block_size, num_kv_heads,head_dim]
        """
        for item in request_blocks_grouped:
            block_id = item['start']
            length = item['length']

            cpu_offset = self._calc_block_offset(layer_id, block_id, kv_type)
            # gpu_offset = self._calc_block_offset(0, block_id, kv_type)
            cpu_slice = self.memory[cpu_offset:cpu_offset+self.block_elem_count*length]
            gpu_slice = gpu_kv_cache[block_id:block_id+length].view(-1)
            # print("_____",cpu_slice.shape, gpu_slice.view(-1).shape)
            cpu_slice.copy_(gpu_slice, non_blocking=True)  # GPU → pinned
    
    def copy_layer_kv_to_cpu(self, gpu_kv_cache: torch.Tensor, kv_type: str, layer_id: int):
        """
        按整层 K 或 V 单位，从 GPU KV Cache 拷贝回该层的 CPU pinned memory。

        :param gpu_kv_cache: GPU端的连续张量（形状为 [num_layers, num_kvcache_blocks, block_size, num_kv_heads, head_dim]）
        :param kv_type: "k" 或 "v"
        :param layer_id: 要拷贝的层编号
        """
        if kv_type not in ("k", "v"):
            raise ValueError("kv_type 必须是 'k' 或 'v'")
        if not (0 <= layer_id < self.num_hidden_layers):
            raise ValueError(f"layer_id 必须在 0 ~ {self.num_hidden_layers-1} 范围内")

        # 每层的元素总数（该kv_type）
        per_layer_size = self.num_kvcache_blocks * self.block_elem_count  # 一个K或V层总长度

        # CPU pinned memory中该层的起始位置
        cpu_offset = self._calc_block_offset(layer_id, 0, kv_type)
        cpu_slice = self.memory[cpu_offset:cpu_offset + per_layer_size]

        # GPU端的起始位置
        # 如果 gpu_kv_cache 的shape是[全层][block][...]，可以flatten直接取
        # 注意：如果 gpu_kv_cache 存的是每层的K或V顺序不一样，需要对应偏移计算
        # 这里假设 gpu_kv_cache 是 flatten 后同排布（K前半，V后半）
        gpu_offset = self._calc_block_offset(layer_id, 0, kv_type)
        gpu_slice = gpu_kv_cache.view(-1)[gpu_offset: gpu_offset + per_layer_size]

        # 异步拷贝 GPU → CPU pinned
        cpu_slice.copy_(gpu_slice, non_blocking=True)