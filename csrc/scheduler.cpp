#include "scheduler.h"
#include "sequence.h"
#include <pybind11/stl.h>
#include<iostream>
#include <cassert>

std::unique_ptr<BlockManagerBase> create_block_manager(const Config& config) {
    if (config.role == "master") {
        return std::make_unique<MasterBlockManager>(config);
    } else if (config.role == "slave") {
        return std::make_unique<SlaveBlockManager>(config);
    } else {
        throw std::invalid_argument("Unknown role: " + config.role);
    }
}
// === 构造函数 ===
Scheduler::Scheduler(py::object pyCfg) {
    // model
    config.model = pyCfg.attr("model").cast<std::string>();
    config.max_num_batched_tokens = pyCfg.attr("max_num_batched_tokens").cast<int>();
    config.max_num_seqs = pyCfg.attr("max_num_seqs").cast<int>();
    config.max_model_len = pyCfg.attr("max_model_len").cast<int>();
    config.gpu_memory_utilization = pyCfg.attr("gpu_memory_utilization").cast<double>();
    config.tensor_parallel_size = pyCfg.attr("tensor_parallel_size").cast<int>();
    config.enforce_eager = pyCfg.attr("enforce_eager").cast<bool>();
    config.eos = pyCfg.attr("eos").cast<int>();
    config.kvcache_block_size = pyCfg.attr("kvcache_block_size").cast<int>();
    config.num_kvcache_blocks = pyCfg.attr("num_kvcache_blocks").cast<int>();
    config.local_num_blocks = pyCfg.attr("local_num_blocks").cast<int>();
    config.rank = pyCfg.attr("rank").cast<int>();
    config.role = pyCfg.attr("role").cast<std::string>();
    config.dist_port = pyCfg.attr("dist_port").cast<int>();
    // ExternalKVCacheConfig
    auto pyCacheCfg = pyCfg.attr("external_kvcache_config");
    config.external_kvcache_config.num_blocks_start_end =
        pyCacheCfg.attr("num_blocks_start_end").cast<std::vector<int>>();
    // Lists
    config.master_list = pyCfg.attr("master_list").cast<std::vector<int>>();
    config.slave_list = pyCfg.attr("slave_list").cast<std::vector<int>>();

    config.master_minimum_scaling_block_count = pyCfg.attr("master_minimum_scaling_block_count").cast<int>();
    config.slave_minimum_scaling_block_count = pyCfg.attr("slave_minimum_scaling_block_count").cast<int>();

    // Master 校验
    if (config.role == "master") {
        assert(config.num_kvcache_blocks -
               config.external_kvcache_config.num_blocks_start_end.back()
               == config.local_num_blocks);
    }
    role = config.role;
    max_num_seqs = config.max_num_seqs;
    max_num_batched_tokens = config.max_num_batched_tokens;
    eos = config.eos;
    // 创建 block_manager
    block_manager = std::unique_ptr<BlockManagerBase>(create_block_manager(config));
    // if(role=='master'){
    //     block_manager = static_cast<std::unique_ptr<MasterBlockManager>(block_manager);
    // }
    // if(block_manager->free_block_ids){
    //     std::cout<<role<<" 存在\n";
    //     // block_manager->free_block_ids->hello_world();
    //     block_manager->hello_world();
    // }
    // else{
    //     block_manager->hello_world();
    //     std::cout << role << " free_block_ids is not initialized!" << std::endl;
    // }
}

// === KVCache变化检查（暂未实现）===
void Scheduler::check_kvcache_change() {
    // TODO: 实现检查逻辑
}

// === 是否已完成所有任务 ===
bool Scheduler::is_finished() const {
    return waiting.empty() && running.empty();
}

// === 添加一个序列到等待队列 ===
void Scheduler::add(Sequence *seq) {
    std::cout<<"----add!"<<std::endl;
    waiting.push_back(seq);
    Sequence* seq_front = waiting.front();
}
void Scheduler::create_add(    
    const std::vector<int>& token_ids_,
    py::object sampling_params,
    std::optional<std::any> input_embeds_,
    std::optional<int> seq_id_) {
        Sequence* seq = new Sequence(token_ids_, sampling_params, input_embeds_, seq_id_);
        // std::cout<<"waiting长度:"<<waiting.size()<<std::endl;
        // std::cout<<"******pre scheduler里面的seq:"<<seq->seq_id<<std::endl;
        waiting.push_back(seq);
        Sequence* seq_front = waiting.front();
        // std::cout<<"------after scheduler里面的seq:"<<seq_front->seq_id<<std::endl;
    }
// === 核心调度逻辑 ===
std::pair<std::vector<Sequence*>, std::string> Scheduler::schedule() {
    std::vector<Sequence*> prefill_seqs;
    std::vector<Sequence*> decoding_seqs;
    
    if (config.role == "master") {
        block_manager->sync_num_free_blocks();
        static_cast<MasterBlockManager*>(block_manager.get())->master_check_blocks_update();

    }

    // === PREFILL阶段 ===
    int num_seqs = 0;
    int num_batched_tokens = 0;
    while (!waiting.empty() && num_seqs < max_num_seqs) {
        Sequence* seq = waiting.front();
        // std::cout<<"++++++after scheduler里面的seq:"<<seq->seq_id<<std::endl;
        if (num_batched_tokens + seq->length() > max_num_batched_tokens
            || !block_manager->can_allocate(*seq)) {
            break;
        }
        num_seqs++;
        block_manager->allocate(*seq);
        num_batched_tokens += seq->length() - seq->num_cached_tokens;
        seq->status = SequenceStatus::RUNNING;
        waiting.pop_front();
        // std::cout<<"*****prefill的seq:"<<seq->seq_id<<std::endl;
        running.push_back(seq);
        prefill_seqs.push_back(seq);
    }

    // double usage_rate = block_manager->usage_rate();
    if (!prefill_seqs.empty()) {
        // std::cout << "KV Cache 使用率为 " << usage_rate * 100 << "%\n";
        return {prefill_seqs, "prefill"};
    }

    // === DECODING阶段 ===
    while (!running.empty() && num_seqs < max_num_seqs) {
        Sequence* seq = running.front();
        // std::cout<<"*****decoding的seq:"<<seq->seq_id<<std::endl;
        running.pop_front();
        // block_manager->can_append(*seq);
        while (!block_manager->can_append(*seq)) {
            if (!running.empty()) {
                preempt(running.back());
                running.pop_back();
            } else {
                preempt(seq);
                break;
            }
        }
        if (seq->status == SequenceStatus::WAITING) {
            continue; // 说明被抢占了
        }
        num_seqs++;
        block_manager->may_append(*seq);
        decoding_seqs.push_back(seq);
    }

    assert(!decoding_seqs.empty());
    // 将 decoding_seqs 放回 running 队列开头（原顺序恢复）
    running.insert(running.begin(), decoding_seqs.begin(), decoding_seqs.end());

    // std::cout << "KV Cache 使用率为 " << usage_rate * 100 << "%\n";
    return {decoding_seqs, "decoding"};
}

// === 抢占一个序列 ===
void Scheduler::preempt(Sequence* seq) {
    seq->status = SequenceStatus::WAITING;
    block_manager->deallocate(*seq);
    waiting.push_front(seq);
}

// === 后处理：添加token并决定是否结束 ===
std::vector<bool> Scheduler::postprocess(
    const std::vector<Sequence*>& seqs,
    const std::vector<int>& token_ids
) {
    std::vector<bool> finished_flags;
    finished_flags.reserve(seqs.size());

    for (size_t i = 0; i < seqs.size(); ++i) {
        auto& seq = seqs[i];
        int token_id = token_ids[i];
        seq->append_token(token_id);

        bool finished = false;
        if ((!seq->ignore_eos && token_id == eos) || seq->num_completion_tokens() == seq->max_tokens) {
            seq->status = SequenceStatus::FINISHED;
            block_manager->deallocate(*seq);
            // 从 running 移除
            running.erase(
                std::remove(running.begin(), running.end(), seq),
                running.end()
            );
            finished = true;
        }
        finished_flags.push_back(finished);
    }
    return finished_flags;
}
