#pragma once
#include <vector>
#include <string>
#include<iostream>
// Block 类声明
class Block {
public:
    int block_id;
    int ref_count;
    int hash;
    std::vector<int> token_ids;
    int hit_count;
    int prefix_position;
    int prefix_depth;
    int external_blocks_num;
    std::string compare_mode;

    // 构造函数声明
    Block(int block_id_,
          const std::string& compare_mode_ = "asc",
          int external_blocks_num_ = 0);
    ~Block() {
            std::cout << "[Block] destructor called, block_id=" << block_id << std::endl;
        }
        
    // 更新 token_ids
    void update(int _hash, const std::vector<int>& _token_ids);

    // 重置
    void reset();

    // 前缀深度差
    int dist() const;

    // 小于号比较（用于堆排序或优先队列）
    bool operator<(const Block& other) const;
};
