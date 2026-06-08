#pragma once
#include <vector>
#include <string>
#include <optional>
#include <any>
#include <cstdint>
#include <unordered_map>
#include <pybind11/pybind11.h>
#include<iostream>
namespace py = pybind11;
// 模拟 SamplingParams
struct SamplingParams {
    float temperature = 1.0f;
    int max_tokens = 0;
    bool ignore_eos = false;
};

enum class SequenceStatus {
    WAITING,
    RUNNING,
    FINISHED
};

class Sequence {
public:
    static int block_size;     // 全局配置中的 kvcache_block_size
    static int counter;        // 全局计数器

    int seq_id;
    SequenceStatus status;
    std::optional<std::any> input_embeds; // 保留泛型属性
    std::vector<int> token_ids;
    int last_token;
    int num_tokens;
    int num_prompt_tokens;
    int num_cached_tokens;
    std::vector<int> block_table;
    std::vector<int> local_block_table;
    std::vector<int> local_cached_blocks;
    std::vector<int> local_uncached_blocks;
    std::vector<int> num_blocks_per_slave;
    std::vector<int> cum_blocks_per_slave;
    std::vector<int> block_belong_to_slave;
    float temperature;
    int max_tokens;
    bool ignore_eos;
    std::vector<double> time_usage;
    std::optional<std::string> request_id;
    std::optional<std::any> finished_event;
    std::unordered_map<std::string, int> extra_info;

    Sequence(const std::vector<int>& token_ids,
             py::object sampling_params,
             std::optional<std::any> input_embeds = std::nullopt,
             std::optional<int> seq_id = std::nullopt);
     ~Sequence() {
                std::cout << "[Sequence] destructor called, seq_id=" << seq_id << std::endl;
            }
    // 基本接口
    int size() const; // 等价于 Python __len__
    int operator[](int key) const; // 等价于 __getitem__

    bool is_finished() const;
    int num_completion_tokens() const;
    std::vector<int> prompt_token_ids() const;
    std::vector<int> completion_token_ids() const;
    int num_cached_blocks() const;
    int num_blocks() const;
    int length() const;
    int last_block_num_tokens() const;
    std::vector<int> block(int i) const;
    void append_token(int token_id);
    void time_usage_append(double t);

    // 序列化/反序列化状态
    std::tuple<int, std::optional<std::any>, int, int, int, std::vector<int>, std::any> get_state() const;
    void set_state(const std::tuple<int, std::optional<std::any>, int, int, int, std::vector<int>, std::any>& state);
};
