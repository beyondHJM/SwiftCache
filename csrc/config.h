#pragma once
#include <string>
#include <vector>
#include <map>
#include <any>
#include <optional>

// 外部 KV Cache 配置
struct ExternalKVCacheConfig {
    std::vector<int> num_blocks_start_end;
    int num_external_kvcache = 0;

    ExternalKVCacheConfig() = default;
};

// 主配置结构
struct Config {
    std::string model;
    int max_num_batched_tokens;
    int max_num_seqs;
    int max_model_len;
    float gpu_memory_utilization;
    bool enforce_eager;
    int eos;
    int tensor_parallel_size;
    int kvcache_block_size;
    int num_kvcache_blocks;
    int local_num_blocks;
    int rank;
    std::string role;
    int dist_port;

    ExternalKVCacheConfig external_kvcache_config;

    std::vector<int> master_list;
    std::vector<int> slave_list;

    int master_minimum_scaling_block_count;
    int slave_minimum_scaling_block_count;

    // 构造函数声明
    Config(const std::string& model_name);
    Config(); // 默认构造
};
