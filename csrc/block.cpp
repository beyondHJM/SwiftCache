#include "block.h"
#include <stdexcept>

Block::Block(int block_id_,
             const std::string& compare_mode_,
             int external_blocks_num_)
    : block_id(block_id_),
      ref_count(0),
      hash(-1),
      hit_count(0),
      prefix_position(0),
      prefix_depth(0),
      external_blocks_num(external_blocks_num_),
      compare_mode(compare_mode_)
{
    if (compare_mode != "asc" && compare_mode != "desc" && compare_mode != "local_first") {
        throw std::invalid_argument("compare_mode must be 'asc', 'desc' or 'local_first'");
    }
}

void Block::update(int _hash, const std::vector<int>& _token_ids) {
    hash = _hash;
    token_ids = _token_ids;
}

void Block::reset() {
    ref_count = 1;
    hash = -1;
    token_ids.clear();
}

int Block::dist() const {
    return prefix_depth - prefix_position;
}

bool Block::operator<(const Block& other) const {
    if (hash == -1 && other.hash != -1)
        return true;
    if (hash != -1 && other.hash == -1)
        return false;
    if (hit_count != other.hit_count)
        return hit_count < other.hit_count;
    if (dist() != other.dist())
        return dist() < other.dist();

    if (compare_mode == "asc")
        return block_id < other.block_id;

    if (compare_mode == "desc")
        return block_id > other.block_id;

    if (compare_mode == "local_first") {
        if (block_id < external_blocks_num && other.block_id < external_blocks_num)
            return block_id < other.block_id;
        if (block_id >= external_blocks_num && other.block_id < external_blocks_num)
            return true;
        if (block_id < external_blocks_num && other.block_id >= external_blocks_num)
            return false;
        if (block_id >= external_blocks_num && other.block_id >= external_blocks_num)
            return block_id < other.block_id;
    }
    return false;
}
