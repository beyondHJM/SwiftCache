#include "config.h"

// 带模型名的构造函数
Config::Config(const std::string& model_name)
    : model(model_name),
      max_num_batched_tokens(40960),
      max_num_seqs(1),
      max_model_len(40960),
      gpu_memory_utilization(0.97f),
      enforce_eager(false),
      eos(0),
      tensor_parallel_size(1),
      kvcache_block_size(0),
      num_kvcache_blocks(-1),
      local_num_blocks(-1),
      rank(0),
      role("master"),
      dist_port(2334),
      master_minimum_scaling_block_count(1),
      slave_minimum_scaling_block_count(1) {}

// 默认构造函数
Config::Config()
    : model(""),
      max_num_batched_tokens(40960),
      max_num_seqs(1),
      max_model_len(40960),
      gpu_memory_utilization(0.97f),
      enforce_eager(false),
      eos(0),
      tensor_parallel_size(1),
      kvcache_block_size(0),
      num_kvcache_blocks(-1),
      local_num_blocks(-1),
      rank(0),
      role("master"),
      dist_port(2334),
      master_minimum_scaling_block_count(1),
      slave_minimum_scaling_block_count(1) {}
