#include "free_blocks_ids.h"
#include <iostream>
#include <chrono>


bool compare_block_ptr(const Block* a, const Block* b) {
    return *b < *a;  // 假设 Block::operator< 是 const 方法
}
// ===== FreeBlockIdsBase =====
FreeBlockIdsBase::FreeBlockIdsBase(const std::vector<Block*>& blocks_,
                                   int offset_,
                                   const Config *config_,
                                   int minimum_scaling_block_count_)
    : config(config_), blocks(blocks_), offset(offset_),
      minimum_scaling_block_count(minimum_scaling_block_count_) {
    // std::make_heap(heap.begin(), heap.end(),compare_block_ptr);
}

void FreeBlockIdsBase::scale_up(int) {
    throw std::runtime_error("scale_up() not implemented");
}

void FreeBlockIdsBase::scale_down(int) {
    throw std::runtime_error("scale_down() not implemented");
}

void FreeBlockIdsBase::scale_up_with_specific_blocks(int) {
    throw std::runtime_error("scale_up_with_specific_blocks() not implemented");
}

void FreeBlockIdsBase::sync_num_free_blocks() {
    throw std::runtime_error("sync_num_free_blocks() not implemented");
}

void FreeBlockIdsBase::push(int block_id) {
    heap.push_back(blocks[block_id]);
    std::push_heap(heap.begin(), heap.end(),compare_block_ptr);
    // std::cout<<"堆顶："<<heap[0]->block_id<<std::endl;
}

void FreeBlockIdsBase::append(int block_id) { push(block_id); }

int FreeBlockIdsBase::pop() {
    if (heap.empty()) return -1;
    std::pop_heap(heap.begin(), heap.end(),compare_block_ptr);
    int id = heap.back()->block_id;  // ✅ 用智能指针直接访问对象成员
    heap.pop_back();                 // 智能指针出作用域会自动释放资源
    return id;
}

int FreeBlockIdsBase::peek() const {
    if (heap.empty()) return -1;
    return heap.front()->block_id;
}

bool FreeBlockIdsBase::remove(int block_id) {
    // 考虑 offset（保持原逻辑）
    block_id += offset;

    // 在 heap 中查找 block_id 匹配的元素
    auto it = std::find_if(heap.begin(), heap.end(),
        [block_id](Block* bptr) {
            return bptr && bptr->block_id == block_id; 
            // bptr && 避免解引用空指针
        });

    // 如果找到了
    if (it != heap.end()) {
        // 删除该元素（不释放内存，因为生命周期由外部统一管理）
        heap.erase(it);
        // 重新整理堆结构
        std::make_heap(heap.begin(), heap.end(), compare_block_ptr);
        // 返回 true 表示删除成功
        return true;
    }
    // 没找到
    return false;
}

void FreeBlockIdsBase::remove_batch_with_global_id(const std::vector<int>& global_block_ids) {

    std::set<int> idset(global_block_ids.begin(), global_block_ids.end());

    // 移除匹配的 Block*
    heap.erase(
        std::remove_if(heap.begin(), heap.end(),
            [&](Block* bptr) {
                return bptr && idset.count(bptr->block_id) > 0;
            }
        ),
        heap.end()
    );

    // 重新整理成堆
    std::make_heap(heap.begin(), heap.end(), compare_block_ptr);
}

void FreeBlockIdsBase::push_batch_with_global_id(const std::vector<int>& block_ids) {
    for (int id : block_ids) {
        heap.push_back(blocks[id - offset]); // blocks是Block*容器，所以直接push指针
    }
    std::make_heap(heap.begin(), heap.end(), compare_block_ptr); // 裸指针版本要带比较器
}

int FreeBlockIdsBase::num_blocks_lent() const {
    return static_cast<int>(blocks.size()) - static_cast<int>(heap.size());
}

int FreeBlockIdsBase::size() const { return heap.size(); }
int FreeBlockIdsBase::operator[](int index) const { return heap[index]->block_id; }

// ===== MasterFreeBlockIds =====
MasterFreeBlockIds::MasterFreeBlockIds(const std::vector<Block*>& blocks_,
                                       int offset_,
                                       const Config *config_,
                                       int minimum_scaling_block_count_)
    : FreeBlockIdsBase(blocks_, offset_, config_, minimum_scaling_block_count_),
      init_block_num(minimum_scaling_block_count_) {
    heap.assign(blocks.begin() + init_block_num, blocks.end());
    std::make_heap(heap.begin(), heap.end(), compare_block_ptr);
}

void MasterFreeBlockIds::scale_down(int n) {
    int num_blocks = minimum_scaling_block_count * n;
    std::vector<int> global_block_ids;
    for (int i = init_block_num; i < init_block_num + num_blocks; ++i) {
        global_block_ids.push_back(blocks[i]->block_id);
    }
    std::cout << "[Master] Global block IDs to remove: ";
    for (auto id : global_block_ids) std::cout << id << " ";
    std::cout << "\n";

    auto t1 = std::chrono::high_resolution_clock::now();
    remove_batch_with_global_id(global_block_ids);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << "[Master] scale_down took " << std::chrono::duration<double>(t2 - t1).count() << " sec\n";

    init_block_num += num_blocks;
}

void MasterFreeBlockIds::master_scale_down_many(int n) {
    int num_blocks = minimum_scaling_block_count * n;
    std::vector<int> global_block_ids;
    for (int i = init_block_num; i < init_block_num + num_blocks; ++i) {
        global_block_ids.push_back(blocks[i]->block_id);
    }
    std::cout << "即将被缩减的global_block_ids:";
    for (auto id : global_block_ids) std::cout << id << " ";
    std::cout << "\n";
    auto t1 = std::chrono::high_resolution_clock::now();
    remove_batch_with_global_id(global_block_ids);
    auto t2 = std::chrono::high_resolution_clock::now();
    std::cout << std::chrono::duration<double>(t2 - t1).count() << "\n";
    init_block_num += num_blocks;
}

// ===== SlaveFreeBlockIds =====
SlaveFreeBlockIds::SlaveFreeBlockIds(const std::vector<Block*>& blocks_,
                                     int offset_,
                                     const Config *config_,
                                     int minimum_scaling_block_count_)
    : FreeBlockIdsBase(blocks_, offset_, config_, minimum_scaling_block_count_),
      init_block_num(minimum_scaling_block_count_) {
    heap.assign(blocks.begin(), blocks.begin() + init_block_num);
    std::make_heap(heap.begin(), heap.end(), compare_block_ptr);
}

void SlaveFreeBlockIds::scale_up(int n) {
    int total_new = minimum_scaling_block_count * n;
    heap.insert(heap.end(),
        blocks.begin() + init_block_num,
        blocks.begin() + init_block_num + total_new);
    std::make_heap(heap.begin(), heap.end(), compare_block_ptr);

    init_block_num += total_new;
    std::cout << "[Slave] init_block_num after scale_up: " << init_block_num << "\n";
}

void SlaveFreeBlockIds::scale_up_with_specific_blocks(int num_blocks) {
    int n = (num_blocks + minimum_scaling_block_count - 1) / minimum_scaling_block_count;
    scale_up(n);
}

// ===== MultiFreeBlockIds =====
MultiFreeBlockIds::MultiFreeBlockIds(const std::vector<Block*>& blocks_,
                                     const std::vector<int>& num_blocks_start_end_,
                                     int local_num_blocks_,
                                     const Config *config_,
                                     const std::vector<int>& minimum_scaling_block_count_)
    : blocks(blocks_),
      num_blocks_start_end(num_blocks_start_end_),
      local_num_blocks(local_num_blocks_),
      config(config_),
      minimum_scaling_block_count(minimum_scaling_block_count_) 
{
    n_group = static_cast<int>(num_blocks_start_end.size()) - 1;
    n_blocks = static_cast<int>(blocks.size());
    n_free_blocks = n_blocks;

    // 外部组初始化
    for (int i = 0; i < n_group; ++i) {
        std::vector<Block*> seg;
        seg.reserve(num_blocks_start_end[i + 1] - num_blocks_start_end[i]); // 提前分配空间

        // 显式 push_back，不转移所有权
        for (auto it = blocks.begin() + num_blocks_start_end[i];
             it != blocks.begin() + num_blocks_start_end[i + 1];
             ++it) 
        {
            seg.push_back(*it); // 只是复制指针地址
        }

        multi_free_block_ids.push_back(
            new MasterFreeBlockIds(seg, num_blocks_start_end[i], config, minimum_scaling_block_count[i])
        );
    }

    // 本地组初始化
    std::vector<Block*> local_seg;
    local_seg.reserve(blocks.size() - num_blocks_start_end.back());

    for (auto it = blocks.begin() + num_blocks_start_end.back();
         it != blocks.end();
         ++it) 
    {
        local_seg.push_back(*it); // 裸指针复制
    }
    std::cout<<"初始化local_free_block_ids";
    for(int i=0;i<10;i++){
        std::cout<<local_seg[i]->block_id<<" ";
    }
    std::cout<<std::endl;
    local_free_block_ids = new MasterFreeBlockIds(local_seg, num_blocks_start_end.back(), config);

    counter = 0;

    std::cout << "n_free_blocks init:" << n_free_blocks << "\n";
    sync_num_free_blocks();
}



MultiFreeBlockIds::~MultiFreeBlockIds() {
    for (auto m : multi_free_block_ids) delete m;
    delete local_free_block_ids;
}

int MultiFreeBlockIds::get_group_idx(int block_id) const {
    for (int i = 0; i < n_group; ++i) {
        if (block_id >= num_blocks_start_end[i] && block_id < num_blocks_start_end[i + 1])
            return i;
    }
    return -1;
}

void MultiFreeBlockIds::append(int block_id) {
    n_free_blocks++;
    int group_id = get_group_idx(block_id);
    if (group_id != -1) {
        int block_id_in_group = block_id - num_blocks_start_end[group_id];
        multi_free_block_ids[group_id]->append(block_id_in_group);
    } else {
        int block_id_in_group = block_id - num_blocks_start_end.back();
        local_free_block_ids->append(block_id_in_group);
    }
}

void MultiFreeBlockIds::remove(int block_id) {
    n_free_blocks--;
    int group_id = get_group_idx(block_id);
    if (group_id != -1) {
        int block_id_in_group = block_id - num_blocks_start_end[group_id];
        multi_free_block_ids[group_id]->remove(block_id_in_group);
    } else {
        int block_id_in_group = block_id - num_blocks_start_end.back();
        local_free_block_ids->remove(block_id_in_group);
    }
}

int MultiFreeBlockIds::size() const { return n_free_blocks; }

int MultiFreeBlockIds::operator[](int index) {
    bool all_empty = true;
    for (auto g : multi_free_block_ids) {
        if (g->size() > 0) { all_empty = false; break; }
    }
    if (all_empty) {
        if (local_free_block_ids->size() == 0)
            throw std::runtime_error("external 和 local 的 block 都空了");
        return (*local_free_block_ids)[index];
    }

    for (int i = 0; i < n_group; ++i) {
        if (multi_free_block_ids[counter]->size() > 0) {
            break;
        }
        std::cout << "group_id:" << counter << " 的 block 已经全部用完\n";
        counter = (counter + 1) % n_group;
    }

    Block* external_block = blocks[multi_free_block_ids[counter]->operator[](index)];
    if (local_free_block_ids->size() == 0) {
        counter = (counter + 1) % n_group;
        return external_block->block_id;
    }

    Block* local_block = blocks[local_free_block_ids->operator[](index)];
// 调试打印
    // std::cout << "local_block:    hash=" << local_block->hash
    //         << "  block_id=" << local_block->block_id << std::endl;

    // std::cout << "external_block: hash=" << external_block->hash
    //       << "  block_id=" << external_block->block_id << std::endl;
    if (*external_block < *local_block) {
        counter = (counter + 1) % n_group;
        // std::cout<<"external_block:"<<external_block->block_id<<std::endl;
        return external_block->block_id;
    } else {
        // std::cout<<"local_block:"<<local_block->block_id<<std::endl;
        return local_block->block_id;
    }
}
int MultiFreeBlockIds::peek() {
    // int block_id = (*this)[0];
    return (*this)[0];
}

void MultiFreeBlockIds::master_scale_down_many(int slave_idx, int n) {
    multi_free_block_ids[slave_idx]->master_scale_down_many(n);
    int num_blocks = minimum_scaling_block_count[slave_idx] * n;
    n_free_blocks -= num_blocks;
}

void MultiFreeBlockIds::sync_num_free_blocks() {
    n_free_blocks = 0;
    for (auto f : multi_free_block_ids) {
        n_free_blocks += f->size();
    }
    n_free_blocks += local_free_block_ids->size();
    // std::cout << "n_free_blocks sync:" << n_free_blocks << "\n";
}
// void MultiFreeBlockIds::hello_world() {
//     std::cout << "hello world \n";
// }
