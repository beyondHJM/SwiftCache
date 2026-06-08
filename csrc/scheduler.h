#pragma once

#include <deque>
#include <vector>
#include <unordered_map>
#include <string>
#include <utility>
#include <memory>
#include <pybind11/pybind11.h>

#include "config.h"
#include "sequence.h"
#include "block_manager.h"
namespace py = pybind11;
// === SchedulingType 枚举 ===
// enum class SchedulingType {
//     PREFILL = "prefill",
//     DECODING = "decoding"
// };

class Scheduler {
private:
    Config config;
    std::string role;
    int max_num_seqs;
    int max_num_batched_tokens;
    int eos;

    std::unique_ptr<BlockManagerBase> block_manager;
    std::deque<Sequence*> waiting;
    std::deque<Sequence*> running;

public:
    Scheduler(py::object pyCfg); 

    void check_kvcache_change();

    bool is_finished() const;

    void add(Sequence *seq); // VisionSequence 可以后续扩展

    std::pair<std::vector<Sequence*>, std::string> schedule();

    void preempt(Sequence* seq);

    void create_add(    
        const std::vector<int>& token_ids_,
        py::object sampling_params,
        std::optional<std::any> input_embeds_,
        std::optional<int> seq_id_);

    std::vector<bool> postprocess(
        const std::vector<Sequence* >& seqs,
        const std::vector<int>& token_ids
    );
};
