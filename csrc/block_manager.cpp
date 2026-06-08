#include "block_manager.h"
#include "free_blocks_ids.h"
#include <iostream>
// ===== BlockManagerBase 构造 =====
BlockManagerBase::BlockManagerBase(const Config& cfg)
    : config(cfg), role(cfg.role), block_size(cfg.kvcache_block_size)
{
    assert(cfg.num_kvcache_blocks > 0);
    blocks.reserve(cfg.num_kvcache_blocks);
    for (int i = 0; i < cfg.num_kvcache_blocks; ++i) {
        blocks.push_back(new Block(i));
    }
}

uint64_t BlockManagerBase::compute_hash(const std::vector<int>& token_ids, int64_t prefix) {
    XXH64_state_t* state = XXH64_createState();
    if (!state) {
        throw std::runtime_error("Failed to allocate XXH64_state_t");
    }
    XXH64_reset(state, 0); // seed=0，与你Python默认一致
    if (prefix != -1) {
        XXH64_update(state, &prefix, sizeof(prefix));
    }
    XXH64_update(state, token_ids.data(), token_ids.size() * sizeof(int));
    uint64_t digest = XXH64_digest(state);
    XXH64_freeState(state);
    return digest;
}

// ===== 分配/释放单个block =====
Block* BlockManagerBase::allocate_block(int block_id) {
    Block* blk = blocks[block_id];
    assert(blk->ref_count == 0);
    blk->reset();
    free_block_ids->remove(block_id);
    used_block_ids.insert(block_id);
    return blk;
}

void BlockManagerBase::deallocate_block(int block_id) {
    assert(blocks[block_id]->ref_count == 0);
    used_block_ids.erase(block_id);
    free_block_ids->append(block_id);
}

// ===== allocate序列 =====
void BlockManagerBase::allocate(Sequence& seq) {
    assert(seq.block_table.empty());
    int64_t h = -1;
    bool cache_miss = false;
    int last_full_block_idx = 0;

    for (int i = 0; i < seq.num_blocks(); ++i) {
        std::vector<int> token_ids = seq.block(i);
        h = (token_ids.size() == block_size)
            ? compute_hash(token_ids, h)
            : -1;
        int block_id = (hash_to_block_id.count(h) > 0) ? hash_to_block_id[h] : -1;
        if (block_id == -1) cache_miss = true;

        Block* block;
        if (cache_miss) {
            block_id = free_block_ids->peek();
            block = allocate_block(block_id);
            block->hit_count = 0;
            block->prefix_position = i;
            block->prefix_depth = 0;
        } else {
            seq.num_cached_tokens += block_size;
            if (used_block_ids.count(block_id)) {
                block = blocks[block_id];
                block->ref_count++;
            } else {
                block = allocate_block(block_id);
            }
        }

        if (h != -1) {
            last_full_block_idx = std::max(last_full_block_idx, i);
            block->hit_count++;
            block->update(h, token_ids);
            uint64_t old_hash = (block_id_to_hash.count(block_id) ? block_id_to_hash[block_id] : (uint64_t)-1);
            if (old_hash != (uint64_t)-1)
                hash_to_block_id.erase(old_hash);
            hash_to_block_id[h] = block_id;
            block_id_to_hash[block_id] = h;
        } else {
            uint64_t old_hash = (block_id_to_hash.count(block_id) ? block_id_to_hash[block_id] : (uint64_t)-1);
            if (old_hash != (uint64_t)-1) {
                hash_to_block_id.erase(old_hash);
                block_id_to_hash.erase(block_id);
            }
        }
        seq.block_table.push_back(block_id);
    }

    for (int i = 0; i < last_full_block_idx; ++i) {
        Block* blk = blocks[seq.block_table[i]];
        blk->prefix_depth = std::max(blk->prefix_depth, last_full_block_idx);
    }
    seq.extra_info["prefix_cached"] = seq.num_cached_tokens;
}

// ===== deallocate序列 =====
void BlockManagerBase::deallocate(Sequence& seq) {
    for (auto it = seq.block_table.rbegin(); it != seq.block_table.rend(); ++it) {
        Block* blk = blocks[*it];
        blk->ref_count--;
        if (blk->ref_count == 0)
            deallocate_block(*it);
    }
    seq.num_cached_tokens = 0;
    seq.block_table.clear();
}

bool BlockManagerBase::can_append(const Sequence& seq) const {
    return free_block_ids->size() >= (seq.length() % block_size == 1);
}

void BlockManagerBase::may_append(Sequence& seq) {
    // std::cout<<"yyyyy"<<seq.seq_id<<std::endl;
    auto& bt = seq.block_table;
    
    Block* last_block = blocks[bt.back()];
    // std::cout<<"xxxxx"<<std::endl;
    if (seq.length() % block_size == 1) {
        assert(last_block->hash != -1);
        int blk_id = free_block_ids->peek();
        allocate_block(blk_id);
        bt.push_back(blk_id);
    }
    else if (seq.length() % block_size == 0) {
        assert(last_block->hash == -1);
        std::vector<int> token_ids = seq.block(seq.num_blocks() - 1);
        int64_t prefix = (bt.size() > 1)
            ? blocks[bt[bt.size() - 2]]->hash
            : -1;
        uint64_t h = compute_hash(token_ids, prefix);
        last_block->update(h, token_ids);
        hash_to_block_id[h] = last_block->block_id;
    }
    else {
        assert(last_block->hash == -1);
    }
}

double BlockManagerBase::usage_rate() const {
    return blocks.empty() ? 0.0 : static_cast<double>(used_block_ids.size()) / blocks.size();
}

// ===== MasterBlockManager =====
MasterBlockManager::MasterBlockManager(const Config& cfg)
    : BlockManagerBase(cfg),
      zmq_server() // 用Config里地址
{
    for (size_t i = 0; i < cfg.slave_list.size(); ++i) {
        slave_rank_to_idx["slave" + std::to_string(cfg.slave_list[i])] = static_cast<int>(i);
    }
    minimum_scaling_block_count.resize(cfg.slave_list.size(), 0);
    collect_minimum_scaling_block_count_from_slave();
    blocks.clear();

    for (int i = 0; i < cfg.num_kvcache_blocks; ++i) {
        blocks.push_back(new Block(
            i, "local_first", cfg.external_kvcache_config.num_blocks_start_end.back()
        ));
    }

    free_block_ids = std::make_unique<MultiFreeBlockIds>(
        blocks, cfg.external_kvcache_config.num_blocks_start_end,
        cfg.local_num_blocks, &cfg, minimum_scaling_block_count
    );
    // std::cout<<"测试free_block_ids\n";
    // free_block_ids->hello_world();
    broadcast_ready_to_slaves();
}

void MasterBlockManager::allocate(Sequence& seq) {
    assert(seq.block_table.empty());
    int64_t h = -1;
    bool cache_miss = false;
    int last_full_block_idx = 0;

    for (int i = 0; i < seq.num_blocks(); ++i) {
        
        std::vector<int> token_ids = seq.block(i);
        h = (token_ids.size() == block_size)
            ? compute_hash(token_ids, h)
            : -1;
        
        int block_id = (hash_to_block_id.count(h) > 0) ? hash_to_block_id[h] : -1;
        if (block_id == -1) cache_miss = true;
        // std::cout<<"hash:"<<h<<" "<<cache_miss<<std::endl;
        Block* block;
        if (cache_miss) {

            block_id = free_block_ids->peek();
            // std::cout<<"新分配："<<block_id<<std::endl;
            block = allocate_block(block_id);
            block->hit_count = 0;
            block->prefix_position = i;
            block->prefix_depth = 0;
        } else {
            seq.num_cached_tokens += block_size;
            if (used_block_ids.count(block_id)) {
                block = blocks[block_id];
                block->ref_count++;
            } else {
                block = allocate_block(block_id);
            }
        }

        if (h != -1) {
            last_full_block_idx = std::max(last_full_block_idx, i);
            block->hit_count++;
            block->update(h, token_ids);
            uint64_t old_hash = (block_id_to_hash.count(block_id) ? block_id_to_hash[block_id] : (uint64_t)-1);
            if (old_hash != (uint64_t)-1)
                // std::cout<<"即将擦除："<<old_hash<<" "<<"新hash:"<<h<<" block_id:"<<block_id<<std::endl;
                hash_to_block_id.erase(old_hash);
            hash_to_block_id[h] = block_id;
            block_id_to_hash[block_id] = h;
        } else {
            uint64_t old_hash = (block_id_to_hash.count(block_id) ? block_id_to_hash[block_id] : (uint64_t)-1);
            if (old_hash != (uint64_t)-1) {
                hash_to_block_id.erase(old_hash);
                block_id_to_hash.erase(block_id);
            }
        }
        seq.block_table.push_back(block_id);
    }

    for (int i = 0; i < last_full_block_idx; ++i) {
        Block* blk = blocks[seq.block_table[i]];
        blk->prefix_depth = std::max(blk->prefix_depth, last_full_block_idx);
    }
    seq.extra_info["prefix_cached"] = seq.num_cached_tokens;
}

void MasterBlockManager::deallocate(Sequence& seq) {
    for (auto it = seq.block_table.rbegin(); it != seq.block_table.rend(); ++it) {
        Block* blk = blocks[*it];
        blk->ref_count--;
        if (blk->ref_count == 0)
            deallocate_block(*it);
    }
    seq.num_cached_tokens = 0;
    seq.block_table.clear();
}

bool MasterBlockManager::can_append(const Sequence& seq) const {
    return free_block_ids->size() >= (seq.length() % block_size == 1);
}

void MasterBlockManager::may_append(Sequence& seq) {
    auto& bt = seq.block_table;
    Block* last_block = blocks[bt.back()];
    if (seq.length() % block_size == 1) {
        assert(last_block->hash != -1);
        int blk_id = free_block_ids->peek();
        allocate_block(blk_id);
        bt.push_back(blk_id);
    }
    else if (seq.length() % block_size == 0) {
        assert(last_block->hash == -1);
        std::vector<int> token_ids = seq.block(seq.num_blocks() - 1);
        int64_t prefix = (bt.size() > 1)
            ? blocks[bt[bt.size() - 2]]->hash
            : -1;
        uint64_t h = compute_hash(token_ids, prefix);
        last_block->update(h, token_ids);
        hash_to_block_id[h] = last_block->block_id;
    }
    else {
        assert(last_block->hash == -1);
    }
}
void MasterBlockManager::deallocate_block(int block_id) {
    // std::cout<<"ccccc\n";
    assert(blocks[block_id]->ref_count == 0);
    used_block_ids.erase(block_id);
    free_block_ids->append(block_id);
}

bool MasterBlockManager::can_allocate(const Sequence& seq) {
    // std::cout<< free_block_ids->size() <<" master aaaaa\n";
    // free_block_ids->hello_world();
    // std::cout<<"master "<<seq.num_blocks()<<std::endl;
    bool res =free_block_ids->size() >= seq.num_blocks();
    return res;
}

Block* MasterBlockManager::allocate_block(int block_id) {
    Block* blk = blocks[block_id];
    assert(blk->ref_count == 0);
    blk->reset();
    free_block_ids->remove(block_id);
    used_block_ids.insert(block_id);
    return blk;
}

void MasterBlockManager::collect_minimum_scaling_block_count_from_slave() {
    int n = static_cast<int>(config.slave_list.size());
    while (n > 0) {
        auto recv_data = zmq_server.recv_map();
        const std::string& ident = recv_data.first;
        const auto& dict = recv_data.second;

        int idx = slave_rank_to_idx.at(ident);
        minimum_scaling_block_count[idx] = std::stoi(dict.at("minimum_scaling_block_count"));
        n--;
    }
}

void MasterBlockManager::broadcast_ready_to_slaves() {
    std::unordered_map<std::string, std::string> msg = { {"message", "ready"} };
    for (const auto& pair : slave_rank_to_idx) {
        zmq_server.send_map(pair.first, msg);
    }
}

void MasterBlockManager::master_check_blocks_update() {
    auto messages = zmq_server.recv_all_map_nonblock();
    if (!messages.empty()) {
        for (auto& [slave_name, info_map] : messages) {
            int slave_idx = slave_rank_to_idx.at(slave_name);
            MasterFreeBlockIds* free_ids = free_block_ids->multi_free_block_ids[slave_idx];
            int times_to_scale_up = std::stoi(info_map.at("times_to_scale_up"));
            free_ids->scale_down(times_to_scale_up);
        }
        // std::cout<<"before sync"<<std::endl;
        free_block_ids->sync_num_free_blocks();
    }
}
void MasterBlockManager::sync_num_free_blocks(){
    free_block_ids->sync_num_free_blocks();
}
// ===== SlaveBlockManager =====
SlaveBlockManager::SlaveBlockManager(const Config& cfg)
    : BlockManagerBase(cfg),
      zmq_client("slave" + std::to_string(cfg.rank))
{
    blocks.clear();
    for (int i = 0; i < cfg.num_kvcache_blocks; ++i)
        blocks.push_back(new Block(i));

    free_block_ids = std::make_unique<SlaveFreeBlockIds>(
        blocks, 0, &cfg, cfg.slave_minimum_scaling_block_count
    );

    notify_master_minimum_scaling_block_count();
    wait_for_master_ready();
}

bool SlaveBlockManager::can_allocate(const Sequence& seq) {
    // free_block_ids->hello_world(); 
    // std::cout<<"slave "<<seq.num_blocks()<<std::endl;
    if (free_block_ids->size() >= seq.num_blocks())
        // std::cout<<"slave aaaaa\n";
        return true;
    size_t can_used = free_block_ids->size() + free_block_ids->num_blocks_lent();
    if (can_used >= static_cast<size_t>(seq.num_blocks())) {
        int num_blocks_to_scale_up = seq.num_blocks() - static_cast<int>(free_block_ids->size());
        int times_to_scale_up = (num_blocks_to_scale_up + config.slave_minimum_scaling_block_count - 1)
            / config.slave_minimum_scaling_block_count;

        zmq_client.send_map({
            {"num_blocks_to_scale_up", std::to_string(num_blocks_to_scale_up)},
            {"times_to_scale_up", std::to_string(times_to_scale_up)}
        });

        free_block_ids->scale_up_with_specific_blocks(num_blocks_to_scale_up);
        return true;
    }
    return false;
}

void SlaveBlockManager::notify_master_minimum_scaling_block_count() {
    zmq_client.send_map({
        {"minimum_scaling_block_count", std::to_string(config.master_minimum_scaling_block_count)}
    });
}

void SlaveBlockManager::wait_for_master_ready() {
    auto reply = zmq_client.recv_map();
    // std::cout<<reply.at("message")<<std::endl;
    assert(reply.at("message") == "ready");
}
