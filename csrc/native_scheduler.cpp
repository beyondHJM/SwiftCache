// C++17 scheduling data plane. Python owns the acknowledged elastic protocol.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#define XXH_INLINE_ALL
#include "include/xxhash.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <deque>
#include <limits>
#include <memory>
#include <set>
#include <stdexcept>
#include <unordered_map>
#include <vector>

namespace py = pybind11;
using namespace pybind11::literals;
using Clock = std::chrono::steady_clock;
using Token = std::int64_t;
// Sequence.token_ids is a Python list. Borrow its integer elements directly;
// avoid pybind11's generic per-element conversion and reserve decode headroom.
static std::vector<Token> read_tokens(py::handle input, std::size_t headroom = 0) {
    if (!PyList_Check(input.ptr()))
        throw py::type_error("Native token input must be a list");
    std::size_t n = static_cast<std::size_t>(PyList_GET_SIZE(input.ptr()));
    std::vector<Token> result;
    if (headroom > result.max_size() - n)
        throw std::length_error("Token capacity overflow");
    result.reserve(n + headroom);
    for (std::size_t i = 0; i < n; ++i) {
        PyObject *value = PyList_GET_ITEM(input.ptr(), static_cast<Py_ssize_t>(i));
        if (!PyLong_Check(value))
            throw py::type_error("Token IDs must be integers");
        Token token = PyLong_AsLongLong(value);
        if (token == -1 && PyErr_Occurred())
            throw py::error_already_set();
        result.push_back(token);
    }
    return result;
}
static double milliseconds(Clock::time_point start) {
    return std::chrono::duration<double, std::milli>(Clock::now() - start).count();
}

struct NativeBlock {
    int block_id;
    int ref_count = 0;
    bool valid_hash = false;
    bool indexed = false;
    std::uint64_t hash = 0;
    std::vector<Token> tokens;
    std::uint64_t hit_count = 0;
    int prefix_position = 0;
    int prefix_depth = 0;
    int external_blocks_num = 0;
    bool local_first = false;
};

struct PlanRow {
    bool full;
    std::uint64_t hash;
    int candidate;
};

struct NativeSequence {
    std::vector<Token> tokens;
    std::vector<int> table;
    std::vector<PlanRow> rows;
    std::uint64_t epoch = std::numeric_limits<std::uint64_t>::max();
    std::uint64_t plan_owner = 0;
    std::uint64_t allocation_owner = 0;
    std::size_t cached = 0;
    std::size_t planned_cached = 0;
    int needed = 0;
    explicit NativeSequence(std::vector<Token> input) : tokens(std::move(input)) {
        if (tokens.empty())
            throw std::invalid_argument("Empty native sequence");
    }
};

struct NativeGroup {
    int offset, count, reserved;
    bool worker;
};

class NativeBlocks {
    static std::size_t checked_count(int count) {
        if (count <= 0)
            throw std::invalid_argument("Block count must be positive");
        return static_cast<std::size_t>(count);
    }
    struct Less {
        const NativeBlocks *owner;
        bool operator()(int a, int b) const {
            const auto &x = owner->blocks_[a];
            const auto &y = owner->blocks_[b];
            if (x.valid_hash != y.valid_hash)
                return !x.valid_hash;
            if (x.hit_count != y.hit_count)
                return x.hit_count < y.hit_count;
            const int xd = x.prefix_depth - x.prefix_position;
            const int yd = y.prefix_depth - y.prefix_position;
            if (xd != yd)
                return xd < yd;
            if (x.local_first && y.local_first) {
                bool xl = a >= x.external_blocks_num, yl = b >= y.external_blocks_num;
                if (xl != yl)
                    return xl;
            }
            return a < b;
        }
    };
    int size_;
    std::vector<NativeBlock> blocks_;
    std::vector<NativeGroup> groups_;
    std::vector<bool> owned_, draining_;
    std::set<int, Less> free_;
    std::set<int> used_;
    std::unordered_map<std::uint64_t, int> index_;
    std::uint64_t epoch_ = 0;
    inline static std::atomic<std::uint64_t> next_owner_{1};
    const std::uint64_t owner_id_ = next_owner_.fetch_add(1);

    NativeBlock &checked(int id) {
        if (id < 0 || id >= static_cast<int>(blocks_.size()))
            throw std::out_of_range("Invalid KV block ID");
        return blocks_[id];
    }
    void forget(NativeBlock &b) {
        if (b.indexed) {
            auto it = index_.find(b.hash);
            if (it != index_.end() && it->second == b.block_id)
                index_.erase(it);
            b.indexed = false;
        }
    }
    void register_span(int id, std::uint64_t h, const Token *ptr, std::size_t count) {
        auto &b = checked(id);
        bool was_free = free_.erase(id) != 0;
        forget(b);
        b.hash = h;
        b.valid_hash = true;
        b.tokens.assign(ptr, ptr + count);
        if (owned_[id]) {
            index_[h] = id;
            b.indexed = true;
        }
        if (was_free)
            free_.insert(id);
        ++epoch_;
    }

  public:
    NativeBlocks(int count, int size, const std::vector<NativeGroup> &groups, bool local_first,
                 int external)
        : size_(size), groups_(groups), owned_(checked_count(count), false),
          draining_(checked_count(count), false), free_(Less{this}) {
        if (count <= 0 || size <= 0)
            throw std::invalid_argument("Invalid block dimensions");
        blocks_.reserve(count);
        for (int id = 0; id < count; ++id) {
            NativeBlock b;
            b.block_id = id;
            b.local_first = local_first;
            b.external_blocks_num = external;
            blocks_.push_back(std::move(b));
        }
        int end = 0;
        for (const auto &g : groups_) {
            if (g.offset != end || g.count < 0 || g.reserved < 0 || g.reserved > g.count)
                throw std::invalid_argument("Invalid native ownership groups");
            end += g.count;
            if (end > count)
                throw std::invalid_argument("Group exceeds physical capacity");
            for (int id = g.offset; id < end; ++id) {
                owned_[id] = g.worker ? id < g.offset + g.reserved : id >= g.offset + g.reserved;
                if (owned_[id])
                    free_.insert(id);
            }
        }
        if (end != count)
            throw std::invalid_argument("Ownership groups must cover all blocks");
    }
    NativeBlock &block(int id) { return checked(id); }
    int block_size() const { return size_; }
    bool owns(int id) const {
        return id >= 0 && id < static_cast<int>(owned_.size()) && owned_[id];
    }
    std::size_t free_count() const { return free_.size(); }
    int next_free() const {
        if (free_.empty())
            throw std::runtime_error("No free owned KV blocks");
        return *free_.begin();
    }
    std::vector<int> used_ids() const { return {used_.begin(), used_.end()}; }
    py::dict hash_index() const {
        py::dict result;
        for (const auto &item : index_)
            result[py::int_(item.first)] = item.second;
        return result;
    }
    py::dict reverse_index() const {
        py::dict result;
        for (const auto &b : blocks_)
            if (b.indexed)
                result[py::int_(b.block_id)] = py::int_(b.hash);
        return result;
    }
    int reserved(int group) const { return groups_.at(group).reserved; }
    std::vector<int> free_ids(int group) const {
        const auto &g = groups_.at(group);
        std::vector<int> ids;
        for (int id : free_)
            if (id >= g.offset && id < g.offset + g.count)
                ids.push_back(id);
        return ids;
    }
    std::vector<int> draining_ids(int group) const {
        const auto &g = groups_.at(group);
        std::vector<int> ids;
        for (int id = g.offset; id < g.offset + g.count; ++id)
            if (draining_[id])
                ids.push_back(id);
        return ids;
    }
    std::vector<int> drain(int group, int target) {
        auto &g = groups_.at(group);
        if (g.worker || target < g.reserved || target > g.count)
            throw std::runtime_error("Invalid reclaim boundary");
        std::vector<int> ids;
        for (int id = g.offset + g.reserved; id < g.offset + target; ++id) {
            free_.erase(id);
            owned_[id] = false;
            draining_[id] = true;
            ids.push_back(id);
        }
        ++epoch_;
        return ids;
    }
    std::vector<int> set_reserved(int group, int target) {
        auto &g = groups_.at(group);
        if (target < 0 || target > g.count)
            throw std::runtime_error("Invalid reserved boundary");
        int lo = g.offset + std::min(target, g.reserved);
        int hi = g.offset + std::max(target, g.reserved);
        // Validate the complete interval before any mutation.
        for (int id = lo; id < hi; ++id)
            if (blocks_[id].ref_count)
                throw std::runtime_error("Attempted to hand off active KV");
        std::vector<int> ids;
        for (int id = lo; id < hi; ++id) {
            free_.erase(id);
            owned_[id] = g.worker ? id < g.offset + target : id >= g.offset + target;
            draining_[id] = false;
            if (owned_[id])
                free_.insert(id);
            ids.push_back(id);
        }
        g.reserved = target;
        ++epoch_;
        return ids;
    }
    void invalidate(int id) {
        auto &b = checked(id);
        bool was_free = free_.erase(id) != 0;
        forget(b);
        if (!b.ref_count) {
            b.valid_hash = false;
            b.tokens.clear();
        }
        if (was_free)
            free_.insert(id);
        ++epoch_;
    }
    NativeBlock &allocate_id(int id) {
        auto &b = checked(id);
        if (b.ref_count || !owns(id) || !free_.count(id))
            throw std::runtime_error("Allocating referenced, unavailable or unowned KV");
        free_.erase(id);
        forget(b);
        b.ref_count = 1;
        b.valid_hash = false;
        b.tokens.clear();
        used_.insert(id);
        ++epoch_;
        return b;
    }
    void free_id(int id) {
        auto &b = checked(id);
        if (b.ref_count || !used_.count(id))
            throw std::runtime_error("Invalid block release");
        used_.erase(id);
        if (!owns(id))
            invalidate(id);
        else if (!free_.insert(id).second)
            throw std::runtime_error("Double free");
        ++epoch_;
    }
    void register_hash(int id, std::uint64_t h, const std::vector<Token> &tokens) {
        register_span(id, h, tokens.data(), tokens.size());
    }
    static std::uint64_t compute_hash(const Token *ptr, std::size_t count, bool has_prefix,
                                      std::uint64_t prefix) {
        XXH64_state_t state;
        XXH64_reset(&state, 0);
        if (has_prefix) {
            unsigned char bytes[8];
            for (int i = 0; i < 8; ++i)
                bytes[i] = static_cast<unsigned char>(prefix >> (8 * i));
            XXH64_update(&state, bytes, 8);
        }
        XXH64_update(&state, ptr, count * sizeof(Token));
        return XXH64_digest(&state);
    }
    void plan(NativeSequence &s) {
        if (s.epoch == epoch_ && s.plan_owner == owner_id_)
            return;
        s.rows.clear();
        s.planned_cached = 0;
        s.needed = 0;
        std::size_t n = (s.tokens.size() + size_ - 1) / size_;
        s.rows.reserve(n);
        bool miss = false, has_prefix = false;
        std::uint64_t hash = 0;
        for (std::size_t i = 0; i < n; ++i) {
            std::size_t offset = i * size_,
                        len = std::min<std::size_t>(size_, s.tokens.size() - offset);
            bool full = len == static_cast<std::size_t>(size_);
            if (full)
                hash = compute_hash(s.tokens.data() + offset, len, has_prefix, hash);
            has_prefix = full;
            int candidate = -1;
            auto it = full ? index_.find(hash) : index_.end();
            if (!miss && i + 1 < n && it != index_.end() && owns(it->second)) {
                const auto &b = blocks_[it->second];
                if (b.tokens.size() == len &&
                    std::equal(b.tokens.begin(), b.tokens.end(), s.tokens.begin() + offset))
                    candidate = it->second;
            }
            if (candidate >= 0) {
                s.planned_cached += size_;
                s.needed += blocks_[candidate].ref_count == 0;
            } else {
                miss = true;
                ++s.needed;
            }
            s.rows.push_back({full, hash, candidate});
        }
        s.epoch = epoch_;
        s.plan_owner = owner_id_;
    }
    py::dict plan_info(NativeSequence &s) {
        plan(s);
        return py::dict("cached_tokens"_a = s.planned_cached, "needed_blocks"_a = s.needed,
                        "epoch"_a = epoch_);
    }
    void allocate(NativeSequence &s) {
        if (!s.table.empty())
            throw std::runtime_error("Sequence already allocated");
        plan(s);
        if (free_.size() < static_cast<std::size_t>(s.needed))
            throw std::runtime_error("Insufficient owned KV capacity");
        s.cached = 0;
        s.table.reserve(s.rows.size());
        s.allocation_owner = owner_id_;
        ++epoch_;
        for (std::size_t i = 0; i < s.rows.size(); ++i) {
            const auto &row = s.rows[i];
            int id = row.candidate;
            if (id >= 0) {
                auto &b = blocks_[id];
                if (!b.ref_count) {
                    free_.erase(id);
                    used_.insert(id);
                }
                ++b.ref_count;
                s.cached += size_;
            } else {
                id = next_free();
                auto &b = allocate_id(id);
                b.hit_count = 0;
                b.prefix_position = static_cast<int>(i);
                if (row.full)
                    register_span(id, row.hash, s.tokens.data() + i * size_, size_);
            }
            if (row.full)
                ++blocks_[id].hit_count;
            s.table.push_back(id);
        }
        s.epoch = std::numeric_limits<std::uint64_t>::max();
    }
    void deallocate(NativeSequence &s) {
        if (!s.table.empty() && s.allocation_owner != owner_id_)
            throw std::runtime_error("Sequence belongs to a different native allocator");
        for (auto it = s.table.rbegin(); it != s.table.rend(); ++it) {
            auto &b = blocks_[*it];
            if (b.ref_count <= 0)
                throw std::runtime_error("Invalid block reference count");
            if (--b.ref_count == 0)
                free_id(*it);
        }
        s.table.clear();
        s.cached = 0;
        s.allocation_owner = 0;
        ++epoch_;
        s.epoch = std::numeric_limits<std::uint64_t>::max();
    }
    bool can_append(const NativeSequence &s) const {
        return !free_.empty() || s.tokens.size() % size_ != 1;
    }
    void may_append(NativeSequence &s) {
        if (s.allocation_owner != owner_id_)
            throw std::runtime_error("Sequence belongs to a different native allocator");
        if (s.tokens.size() % size_ == 1)
            s.table.push_back(allocate_id(next_free()).block_id);
        else if (s.tokens.size() % size_ == 0) {
            bool previous = s.table.size() > 1 && blocks_[s.table[s.table.size() - 2]].valid_hash;
            std::uint64_t prefix = previous ? blocks_[s.table[s.table.size() - 2]].hash : 0;
            auto h =
                compute_hash(s.tokens.data() + s.tokens.size() - size_, size_, previous, prefix);
            register_span(s.table.back(), h, s.tokens.data() + s.tokens.size() - size_, size_);
        }
    }
    void assert_invariants() const {
        for (const auto &b : blocks_) {
            int id = b.block_id;
            if ((b.ref_count > 0) != (used_.count(id) != 0))
                throw std::runtime_error("Used/refcount invariant");
            if (free_.count(id) && (!owns(id) || b.ref_count))
                throw std::runtime_error("Free/owner invariant");
            if (owns(id) && !b.ref_count && !free_.count(id))
                throw std::runtime_error("Lost owned free block");
        }
        for (const auto &entry : index_) {
            const auto &b = blocks_[entry.second];
            if (!owns(b.block_id) || !b.valid_hash || b.hash != entry.first || !b.indexed)
                throw std::runtime_error("Prefix hash/owner invariant");
        }
    }
};

// Each record owns the Python facade; NativeSequence never owns its facade,
// avoiding an uncollectable Python -> C++ -> Python reference cycle.
struct Record {
    py::object seq;
    std::shared_ptr<NativeSequence> state;
};

static std::shared_ptr<NativeSequence> sequence_state(py::handle seq) {
    if (!py::hasattr(seq, "_cpp_state"))
        seq.attr("_cpp_state") = py::cast(std::make_shared<NativeSequence>(
            read_tokens(seq.attr("token_ids"), seq.attr("max_tokens").cast<std::size_t>())));
    return seq.attr("_cpp_state").cast<std::shared_ptr<NativeSequence>>();
}
static void sync(py::handle seq, const NativeSequence &s) {
    seq.attr("block_table") = py::cast(s.table);
    seq.attr("num_cached_tokens") = py::int_(s.cached);
}

class NativeScheduler {
    std::shared_ptr<NativeBlocks> blocks_;
    py::object manager_, waiting_status_, running_status_, finished_status_;
    std::deque<Record> waiting_, running_;
    int max_seqs_, eos_;
    std::size_t budget_;
    double prefix_ms_ = 0;

    void preempt_record(Record r) {
        blocks_->deallocate(*r.state);
        sync(r.seq, *r.state);
        r.seq.attr("status") = waiting_status_;
        waiting_.push_front(std::move(r));
    }
    static bool erase(std::deque<Record> &queue, py::handle seq) {
        auto it = std::find_if(queue.begin(), queue.end(),
                               [&](const Record &r) { return r.seq.is(seq); });
        if (it == queue.end())
            return false;
        queue.erase(it);
        return true;
    }

  public:
    NativeScheduler(std::shared_ptr<NativeBlocks> blocks, py::object manager, int max_seqs,
                    std::size_t budget, int eos, py::object statuses)
        : blocks_(std::move(blocks)), manager_(std::move(manager)), max_seqs_(max_seqs), eos_(eos),
          budget_(budget) {
        if (max_seqs <= 0 || budget == 0)
            throw std::invalid_argument("Invalid scheduling budget");
        waiting_status_ = statuses.attr("WAITING");
        running_status_ = statuses.attr("RUNNING");
        finished_status_ = statuses.attr("FINISHED");
    }
    void add(py::object seq) {
        auto state = sequence_state(seq);
        waiting_.push_back({std::move(seq), std::move(state)});
    }
    bool is_finished() const { return waiting_.empty() && running_.empty(); }
    int request_floor() const {
        int result = 0;
        for (const auto &r : waiting_)
            result = std::max(
                result, static_cast<int>((r.state->tokens.size() + blocks_->block_size() - 1) /
                                         blocks_->block_size()));
        return result;
    }
    py::list waiting() const {
        py::list out;
        for (const auto &r : waiting_)
            out.append(r.seq);
        return out;
    }
    py::list running() const {
        py::list out;
        for (const auto &r : running_)
            out.append(r.seq);
        return out;
    }
    double prefix_ms() const { return prefix_ms_; }
    py::tuple schedule() {
        prefix_ms_ = 0;
        py::list scheduled;
        std::size_t query = 0;
        while (!waiting_.empty() && py::len(scheduled) < static_cast<std::size_t>(max_seqs_)) {
            Record r = waiting_.front();
            auto start = Clock::now();
            // One callback/request for pending resize and worker growth; all
            // token/block loops and ownership data stay in the native core.
            bool allowed = manager_.attr("can_allocate")(r.seq).cast<bool>();
            prefix_ms_ += milliseconds(start);
            if (!allowed || (query + r.state->tokens.size() - r.state->planned_cached > budget_ &&
                             py::len(scheduled) != 0))
                break;
            start = Clock::now();
            blocks_->allocate(*r.state);
            sync(r.seq, *r.state);
            r.seq.attr("extra_info")[py::str("prefix_cached")] = py::int_(r.state->cached);
            prefix_ms_ += milliseconds(start);
            query += r.state->tokens.size() - r.state->cached;
            r.seq.attr("status") = running_status_;
            waiting_.pop_front();
            running_.push_back(r);
            scheduled.append(r.seq);
        }
        if (py::len(scheduled) != 0)
            return py::make_tuple(scheduled, "prefill");
        std::deque<Record> selected;
        while (!running_.empty() && static_cast<int>(selected.size()) < max_seqs_) {
            Record r = running_.front();
            running_.pop_front();
            bool ready = true;
            while (!manager_.attr("can_append")(r.seq).cast<bool>()) {
                if (!running_.empty()) {
                    auto victim = running_.back();
                    running_.pop_back();
                    preempt_record(std::move(victim));
                } else {
                    preempt_record(r);
                    ready = false;
                    break;
                }
            }
            if (ready) {
                const auto previous_blocks = r.state->table.size();
                blocks_->may_append(*r.state);
                // Most decode steps retain the same table. Crossing the Python
                // boundary with all 400 block IDs every token would waste time.
                if (r.state->table.size() != previous_blocks)
                    sync(r.seq, *r.state);
                selected.push_back(r);
                scheduled.append(r.seq);
            }
        }
        for (auto it = selected.rbegin(); it != selected.rend(); ++it)
            running_.push_front(*it);
        return py::make_tuple(scheduled, py::len(scheduled) == 0 ? "idle" : "decoding");
    }
    void preempt(py::object seq) {
        if (!erase(running_, seq))
            throw std::runtime_error("Preempting a nonrunning request");
        preempt_record({seq, sequence_state(seq)});
    }
    void cancel(py::object seq) {
        bool found = erase(waiting_, seq);
        found = erase(running_, seq) || found;
        if (found) {
            auto state = sequence_state(seq);
            if (!state->table.empty())
                blocks_->deallocate(*state);
            sync(seq, *state);
        }
    }
    void postprocess(const py::list &seqs, const std::vector<Token> &tokens) {
        if (py::len(seqs) != tokens.size())
            throw std::invalid_argument("Mismatched output batch");
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            auto seq = seqs[i];
            auto state = sequence_state(seq);
            state->tokens.push_back(tokens[i]);
            seq.attr("append_token")(tokens[i]);
            bool eos = !seq.attr("ignore_eos").cast<bool>() && tokens[i] == eos_;
            bool limit =
                seq.attr("num_completion_tokens").cast<int>() == seq.attr("max_tokens").cast<int>();
            bool stop = py::hasattr(seq, "force_stop") && seq.attr("force_stop").cast<bool>();
            if (eos || limit || stop) {
                seq.attr("status") = finished_status_;
                blocks_->deallocate(*state);
                sync(seq, *state);
                erase(running_, seq);
            }
        }
    }
};

PYBIND11_MODULE(_native_scheduler, m) {
    m.doc() = "Elastic-safe SwiftCache C++17 scheduler data plane";
    py::class_<NativeBlock>(m, "Block")
        .def_readonly("block_id", &NativeBlock::block_id)
        .def_readwrite("ref_count", &NativeBlock::ref_count)
        .def_readonly("hit_count", &NativeBlock::hit_count)
        .def_readonly("prefix_position", &NativeBlock::prefix_position)
        .def_readonly("prefix_depth", &NativeBlock::prefix_depth)
        .def_property_readonly("hash",
                               [](const NativeBlock &b) -> py::object {
                                   return b.valid_hash ? py::cast(b.hash) : py::int_(-1);
                               })
        .def_property_readonly("token_ids", [](const NativeBlock &b) { return b.tokens; });
    py::class_<NativeSequence, std::shared_ptr<NativeSequence>>(m, "SequenceState")
        .def(py::init([](py::handle tokens, std::size_t headroom) {
                 return std::make_shared<NativeSequence>(read_tokens(tokens, headroom));
             }),
             py::arg("tokens"), py::arg("output_headroom") = 0)
        .def_property_readonly("num_tokens",
                               [](const NativeSequence &s) { return s.tokens.size(); })
        .def("extend",
             [](NativeSequence &s, const std::vector<Token> &tokens) {
                 s.tokens.insert(s.tokens.end(), tokens.begin(), tokens.end());
                 s.epoch = std::numeric_limits<std::uint64_t>::max();
             })
        .def_readonly("cached_tokens", &NativeSequence::cached)
        .def_readonly("table", &NativeSequence::table);
    py::class_<NativeGroup>(m, "Group")
        .def(py::init([](int offset, int count, int reserved, bool worker) {
            return NativeGroup{offset, count, reserved, worker};
        }));
    py::class_<NativeBlocks, std::shared_ptr<NativeBlocks>>(m, "BlockManager")
        .def(py::init<int, int, const std::vector<NativeGroup> &, bool, int>())
        .def("block", &NativeBlocks::block, py::return_value_policy::reference_internal)
        .def("owns", &NativeBlocks::owns)
        .def("free_count", &NativeBlocks::free_count)
        .def("next_free", &NativeBlocks::next_free)
        .def("used_ids", &NativeBlocks::used_ids)
        .def("hash_index", &NativeBlocks::hash_index)
        .def("reverse_index", &NativeBlocks::reverse_index)
        .def("reserved", &NativeBlocks::reserved)
        .def("free_ids", &NativeBlocks::free_ids)
        .def("draining_ids", &NativeBlocks::draining_ids)
        .def("drain", &NativeBlocks::drain)
        .def("set_reserved", &NativeBlocks::set_reserved)
        .def("invalidate", &NativeBlocks::invalidate)
        .def("allocate_id", &NativeBlocks::allocate_id, py::return_value_policy::reference_internal)
        .def("free_id", &NativeBlocks::free_id)
        .def("register_hash", &NativeBlocks::register_hash)
        .def("plan", &NativeBlocks::plan_info)
        .def("allocate", &NativeBlocks::allocate)
        .def("deallocate", &NativeBlocks::deallocate)
        .def("can_append", &NativeBlocks::can_append)
        .def("may_append", &NativeBlocks::may_append)
        .def("assert_invariants", &NativeBlocks::assert_invariants)
        .def_static(
            "compute_hash",
            [](const std::vector<Token> &tokens, py::object prefix) {
                bool present = !prefix.is_none();
                return NativeBlocks::compute_hash(tokens.data(), tokens.size(), present,
                                                  present ? prefix.cast<std::uint64_t>() : 0);
            },
            py::arg("tokens"), py::arg("prefix") = py::none());
    py::class_<NativeScheduler>(m, "Scheduler")
        .def(py::init<std::shared_ptr<NativeBlocks>, py::object, int, std::size_t, int,
                      py::object>())
        .def("add", &NativeScheduler::add)
        .def("schedule", &NativeScheduler::schedule)
        .def("is_finished", &NativeScheduler::is_finished)
        .def("preempt", &NativeScheduler::preempt)
        .def("cancel", &NativeScheduler::cancel)
        .def("postprocess", &NativeScheduler::postprocess)
        .def_property_readonly("waiting", &NativeScheduler::waiting)
        .def_property_readonly("running", &NativeScheduler::running)
        .def_property_readonly("request_floor", &NativeScheduler::request_floor)
        .def_property_readonly("prefix_ms", &NativeScheduler::prefix_ms);
}
