#pragma once
#include <vector>
#include <set>
#include <algorithm>
#include <memory>
#include "block.h"
#include "config.h"
#include<iostream>

// ===== Base Class =====
class FreeBlockIdsBase {
protected:
    const Config* config;
    std::vector<Block*> blocks;
    int offset;
    int minimum_scaling_block_count;
  

public:
    FreeBlockIdsBase(const std::vector<Block*>& blocks,
                     int offset = 0,
                     const Config* config = nullptr,
                     int minimum_scaling_block_count = 0);
    virtual ~FreeBlockIdsBase() = default;
    std::vector<Block*> heap; // 小根堆
    virtual void scale_up(int n = 1);
    virtual void scale_down(int n = 1);
    virtual void scale_up_with_specific_blocks(int num_blocks);
    virtual void sync_num_free_blocks();
    virtual void hello_world(){std::cout<<"FreeBlockIdsBase Hello World\n";};
    void push(int block_id);
    void append(int block_id);
    int pop();
    int peek() const;

    bool remove(int block_id);
    void remove_batch_with_global_id(const std::vector<int>& global_block_ids);
    void push_batch_with_global_id(const std::vector<int>& block_ids);

    int num_blocks_lent() const;
    int size() const;
    int operator[](int index) const;
};

// ===== Master Class =====
class MasterFreeBlockIds : public FreeBlockIdsBase {
    int init_block_num;

public:
    MasterFreeBlockIds(const std::vector<Block*>& blocks,
                       int offset = 0,
                       const Config* config = nullptr,
                       int minimum_scaling_block_count = 0);

    void scale_down(int n) override;
    void master_scale_down_many(int n);
    void hello_world(){std::cout<<"MasterFreeBlockIds Hello World\n";};
};

// ===== Slave Class =====
class SlaveFreeBlockIds : public FreeBlockIdsBase {
    int init_block_num;

public:
    SlaveFreeBlockIds(const std::vector<Block*>& blocks,
                      int offset = 0,
                      const Config* config = nullptr,
                      int minimum_scaling_block_count = 1);

    void scale_up(int n = 1) override;
    void scale_up_with_specific_blocks(int num_blocks);
    void hello_world(){std::cout<<"SlaveFreeBlockIds Hello World\n";};
};

// ===== MultiFreeBlockIds =====
class MultiFreeBlockIds {
    std::vector<Block*> blocks;
    std::vector<int> num_blocks_start_end;
    int local_num_blocks;
    const Config* config;
    std::vector<int> minimum_scaling_block_count;
    int n_group;
    int n_blocks;
    int n_free_blocks;
    int counter;

public:
    MultiFreeBlockIds(const std::vector<Block*>& blocks,
                      const std::vector<int>& num_blocks_start_end,
                      int local_num_blocks,
                      const Config* config,
                      const std::vector<int>& minimum_scaling_block_count);

    ~MultiFreeBlockIds();
    std::vector<MasterFreeBlockIds*> multi_free_block_ids;
    MasterFreeBlockIds* local_free_block_ids;
    int get_group_idx(int block_id) const;
    void append(int block_id);
    void remove(int block_id);
    int peek();
    int size() const;
    int operator[](int index);
    void master_scale_down_many(int slave_idx, int n);
    void sync_num_free_blocks();
    void hello_world(){std::cout<<"MultiFreeBlockIds Hello World\n";};
};
