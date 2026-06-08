#include "sequence.h"
#include <cassert>
#include <algorithm>
#include <iostream>
// 静态成员初始化
int Sequence::block_size = 256;
int Sequence::counter = 0;



Sequence::Sequence(
    const std::vector<int>& token_ids_,
    py::object sampling_params,
    std::optional<std::any> input_embeds_,
    std::optional<int> seq_id_
)
    : seq_id(seq_id_.value_or(counter++)),
      status(SequenceStatus::WAITING),
      input_embeds(input_embeds_),
      token_ids(token_ids_),
      last_token(token_ids_.empty() ? -1 : token_ids_.back()),
      num_tokens(token_ids_.size()),
      num_prompt_tokens(token_ids_.size()),
      num_cached_tokens(0),
      temperature( sampling_params.attr("temperature").cast<float>() ),   // 从 Python 对象取属性
      max_tokens( sampling_params.attr("max_tokens").cast<int>() ),
      ignore_eos( sampling_params.attr("ignore_eos").cast<bool>() )
{
    block_table.clear();
    local_block_table.clear();
    local_cached_blocks.clear();
    local_uncached_blocks.clear();
    num_blocks_per_slave.clear();
    cum_blocks_per_slave.clear();
    block_belong_to_slave.clear();
    time_usage.clear();
    extra_info.clear();
}

int Sequence::size() const {
    return num_tokens;
}

int Sequence::operator[](int key) const {
    return token_ids.at(key);
}

bool Sequence::is_finished() const {
    return status == SequenceStatus::FINISHED;
}

int Sequence::num_completion_tokens() const {
    return num_tokens - num_prompt_tokens;
}

std::vector<int> Sequence::prompt_token_ids() const {
    if (num_prompt_tokens <= token_ids.size())
        return std::vector<int>(token_ids.begin(), token_ids.begin() + num_prompt_tokens);
    return {};
}

std::vector<int> Sequence::completion_token_ids() const {
    if (num_prompt_tokens <= token_ids.size())
        return std::vector<int>(token_ids.begin() + num_prompt_tokens, token_ids.end());
    return {};
}

int Sequence::num_cached_blocks() const {
    return num_cached_tokens / block_size;
}

int Sequence::num_blocks() const {
    int num_blocks = (num_tokens + block_size - 1) / block_size;
    return num_blocks;
}

int Sequence::length() const {
    return  num_tokens;
}
int Sequence::last_block_num_tokens() const {
    return num_tokens - (num_blocks() - 1) * block_size;
}

std::vector<int> Sequence::block(int i) const {
    assert(i >= 0 && i < num_blocks());
    int start = i * block_size;
    int end = std::min(start + block_size, num_tokens);
    return std::vector<int>(token_ids.begin()+start, token_ids.begin() + end);;
}

void Sequence::append_token(int token_id) {
    token_ids.push_back(token_id);
    last_token = token_id;
    num_tokens++;
}
void Sequence::time_usage_append(double t) {
    time_usage.push_back(t);
}
std::tuple<int, std::optional<std::any>, int, int, int, std::vector<int>, std::any> Sequence::get_state() const {
    if (num_completion_tokens() == 0) {
        return { seq_id, input_embeds, num_tokens, num_prompt_tokens, num_cached_tokens, block_table,
                 std::any(token_ids) };
    } else {
        return { seq_id, input_embeds, num_tokens, num_prompt_tokens, num_cached_tokens, block_table,
                 std::any(last_token) };
    }
}

void Sequence::set_state(const std::tuple<int, std::optional<std::any>, int, int, int, std::vector<int>, std::any>& state) {
    seq_id            = std::get<0>(state);
    input_embeds      = std::get<1>(state);
    num_tokens        = std::get<2>(state);
    num_prompt_tokens = std::get<3>(state);
    num_cached_tokens = std::get<4>(state);
    block_table       = std::get<5>(state);

    if (num_completion_tokens() == 0) {
        token_ids = std::any_cast<std::vector<int>>(std::get<6>(state));
    } else {
        last_token = std::any_cast<int>(std::get<6>(state));
    }
}
