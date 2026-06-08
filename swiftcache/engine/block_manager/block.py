
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