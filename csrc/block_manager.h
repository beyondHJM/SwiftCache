#pragma once
#include <iostream>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <memory>
#include <algorithm>
#include <cstdint>
#include <cassert>

#include "sequence.h"
#include "block.h"
#include "config.h"
#include "free_blocks_ids.h"     // 包含 MultiFreeBlockIds / SlaveFreeBlockIds
#include "zmq_wrapper.h"      // 你给的ZMQClient/ZMQServer封装
#include "include/xxhash.h"           // xxhash库

// ===== BlockManagerBase =====
class BlockManagerBase {
protected:
    Config config;
    std::string role;
    size_t block_size;
    std::vector<Block*> blocks;
    std::unordered_set<int> used_block_ids;
    std::unordered_set<int> occupied_block_ids;
    std::unordered_map<uint64_t, int> hash_to_block_id;
    std::unordered_map<int, uint64_t> block_id_to_hash;

    // free_block_ids 可以是 MultiFreeBlockIds 或 SlaveFreeBlockIds

public:
    std::unique_ptr<FreeBlockIdsBase> free_block_ids;
    BlockManagerBase(const Config& cfg);
    virtual ~BlockManagerBase() = default;

    static uint64_t compute_hash(const std::vector<int>& token_ids, int64_t prefix = -1);

    virtual bool can_allocate(const Sequence& seq) = 0;
    virtual void hello_world(){std::cout<<"BlockManagerBase Hello World \n";};
    virtual void sync_num_free_blocks(){};
    virtual bool can_append(const Sequence& seq) const;
    virtual void may_append(Sequence& seq);
    double usage_rate() const;

    virtual void allocate(Sequence& seq);
    virtual void deallocate(Sequence& seq);

protected:
    virtual Block* allocate_block(int block_id);
    void deallocate_block(int block_id);
};

// ===== MasterBlockManager =====
class MasterBlockManager : public BlockManagerBase {
private:
    std::unordered_map<std::string, int> slave_rank_to_idx;
    std::vector<int> minimum_scaling_block_count;
    ZMQServer zmq_server;
    void collect_minimum_scaling_block_count_from_slave();
    void broadcast_ready_to_slaves();

public:
    MasterBlockManager(const Config& cfg);
    std::unique_ptr<MultiFreeBlockIds> free_block_ids;
    bool can_allocate(const Sequence& seq) override;
    void master_check_blocks_update();
    virtual void sync_num_free_blocks();
    void allocate(Sequence& seq);
    Block* allocate_block(int block_id);
    void deallocate(Sequence& seq);
    void deallocate_block(int block_id);
    bool can_append(const Sequence& seq) const;
    void may_append(Sequence& seq);
    void hello_world(){std::cout<<"MasterManager Hello World \n";};
};

// ===== SlaveBlockManager =====
class SlaveBlockManager : public BlockManagerBase {
private:
    ZMQClient zmq_client;
    void notify_master_minimum_scaling_block_count();
    void wait_for_master_ready();

public:
    SlaveBlockManager(const Config& cfg);

    bool can_allocate(const Sequence& seq) override;
};

