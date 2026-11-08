"""Native scheduling data plane with the existing elastic control state machine.

The C++ core is authoritative for hashes, references, free/owned blocks and queues.
The Python coordinator sees native block/group views, so acknowledged handoff,
draining and GPU quiescence follow exactly the same control protocol as Python.
"""

import time

try:
    from . import _native_scheduler as native
except ImportError as exc:
    raise ImportError(
        "C++ scheduler is not built for this Python. Run bash scripts/build_cpp_scheduler.sh"
    ) from exc

from .block_manager.block_manager import MasterBlockManager, SlaveBlockManager
from .block_manager import create_block_manager
from .sequence import Sequence, SequenceStatus


def sequence_state(seq):
    state = getattr(seq, "_cpp_state", None)
    if state is None:
        state = seq._cpp_state = native.SequenceState(seq.token_ids, seq.max_tokens)
    elif state.num_tokens < len(seq):
        state.extend(seq.token_ids[state.num_tokens :])
    elif state.num_tokens != len(seq):
        raise RuntimeError("Queued native input may only be extended")
    return state


def sync_sequence(seq, state):
    seq.block_table = state.table
    seq.num_cached_tokens = state.cached_tokens


class NativeGroupView:
    def __init__(self, core, index, blocks, offset, unit=1):
        self.core, self.index, self.blocks, self.offset = core, index, blocks, offset
        self.minimum_scaling_block_count = unit

    @property
    def init_block_num(self):
        return self.core.reserved(self.index)

    @property
    def num_blocks_lent(self):
        return len(self.blocks) - self.init_block_num

    @property
    def _free(self):
        return set(self.core.free_ids(self.index))

    @property
    def draining(self):
        return set(self.core.draining_ids(self.index))

    def owns(self, block_id):
        return self.offset <= block_id < self.offset + len(
            self.blocks
        ) and self.core.owns(block_id)

    def __len__(self):
        return len(self.core.free_ids(self.index))

    def __getitem__(self, index):
        return self.core.free_ids(self.index)[index]

    def drain(self, target):
        return set(self.core.drain(self.index, target))

    def set_reserved(self, target):
        return set(self.core.set_reserved(self.index, target))


class NativeFreeView:
    def __init__(self, core, groups, boundaries):
        self.core, self.groups = core, groups
        self.multi_free_block_ids = groups[:-1]
        self.local_free_block_ids = groups[-1]
        self.num_blocks_start_end = boundaries

    def owns(self, block_id):
        return self.core.owns(block_id)

    def __len__(self):
        return self.core.free_count()

    def __getitem__(self, index):
        if index != 0:
            raise IndexError("Only the next free block is supported")
        return self.core.next_free()


class NativeManagerMixin:
    @property
    def used_block_ids(self):
        return set(self._native.used_ids())

    @property
    def hash_to_block_id(self):
        return self._native.hash_index()

    @property
    def block_id_to_hash(self):
        return self._native.reverse_index()

    def invalidate(self, block_id):
        self._native.invalidate(block_id)

    def register_hash(self, block, h, tokens):
        self._native.register_hash(block.block_id, h, tokens)

    def _allocate_block(self, block_id):
        return self._native.allocate_id(block_id)

    def _deallocate_block(self, block_id):
        self._native.free_id(block_id)

    def allocation_plan(self, seq):
        return self._native.plan(sequence_state(seq))

    def can_allocate(self, seq):
        if self.role == "slave" and self.pending:
            return False
        needed = self.allocation_plan(seq)["needed_blocks"]
        free = self._native.free_count()
        if free >= needed:
            return True
        if self.role == "slave":
            need = self.free_block_ids.init_block_num + needed - free
            if need <= len(self.blocks):
                self.request_resize(need)
        return False

    def allocate(self, seq):
        state = sequence_state(seq)
        self._native.allocate(state)
        sync_sequence(seq, state)
        seq.extra_info["prefix_cached"] = seq.num_cached_tokens

    def deallocate(self, seq):
        state = sequence_state(seq)
        self._native.deallocate(state)
        sync_sequence(seq, state)

    def can_append(self, seq):
        if self._native.can_append(sequence_state(seq)):
            return True
        if self.role == "slave" and not self.pending:
            self.request_resize(self.free_block_ids.init_block_num + 1)
        return False

    def may_append(self, seq):
        state = sequence_state(seq)
        self._native.may_append(state)
        sync_sequence(seq, state)

    def assert_invariants(self):
        self._native.assert_invariants()


class NativeMasterManager(NativeManagerMixin, MasterBlockManager):
    pass


class NativeSlaveManager(NativeManagerMixin, SlaveBlockManager):
    pass


def enable_native(manager):
    """Attach native storage after the normal empty-manager handshake.

    This is an initialization operation, never a live migration. The original
    coordinator methods are retained; their block and ownership views now refer
    directly to the native core instead of independently mirrored metadata.
    """
    if isinstance(manager, NativeManagerMixin):
        return manager
    if manager.used_block_ids or manager.hash_to_block_id:
        raise RuntimeError("Native allocator must be enabled before any requests")
    old_free = manager.free_block_ids
    old_groups = old_free.groups if manager.role == "master" else [old_free]
    groups = [
        native.Group(g.offset, len(g.blocks), g.init_block_num, manager.role == "slave")
        for g in old_groups
    ]
    external = old_free.num_blocks_start_end[-1] if manager.role == "master" else 0
    core = native.BlockManager(
        len(manager.blocks),
        manager.block_size,
        groups,
        manager.role == "master",
        external,
    )
    manager._native = core
    manager.blocks = [core.block(i) for i in range(len(manager.blocks))]
    views = [
        NativeGroupView(
            core,
            i,
            manager.blocks[g.offset : g.offset + len(g.blocks)],
            g.offset,
            g.minimum_scaling_block_count,
        )
        for i, g in enumerate(old_groups)
    ]
    manager.free_block_ids = (
        NativeFreeView(core, views, old_free.num_blocks_start_end)
        if manager.role == "master"
        else views[0]
    )
    # Native getters replace the initialization-only Python dictionaries/sets.
    for key in ["used_block_ids", "hash_to_block_id", "block_id_to_hash"]:
        manager.__dict__.pop(key)
    manager.__class__ = (
        NativeMasterManager if manager.role == "master" else NativeSlaveManager
    )
    return manager


class Scheduler:
    backend = "cpp"

    def __init__(self, config):
        self._initialize(config, create_block_manager(config))

    @classmethod
    def from_block_manager(cls, manager, config):
        obj = cls.__new__(cls)
        obj._initialize(config, manager)
        return obj

    def _initialize(self, config, manager):
        self.config, self.role = config, config.role
        self.block_manager = enable_native(manager)
        self._queue = native.Scheduler(
            self.block_manager._native,
            self.block_manager,
            config.max_num_seqs,
            config.max_num_batched_tokens,
            config.eos,
            SequenceStatus,
        )
        self.last_metrics = {}

    @property
    def waiting(self):
        return self._queue.waiting

    @property
    def running(self):
        return self._queue.running

    def add(self, seq):
        if not isinstance(seq, Sequence):
            raise TypeError("Native scheduler supports text Sequence objects")
        self.block_manager.observe(seq)
        self._queue.add(seq)

    def schedule(self):
        start = time.perf_counter_ns()
        self.block_manager.request_floor = self._queue.request_floor
        self.block_manager.poll()
        control_ms = (time.perf_counter_ns() - start) / 1e6
        result = self._queue.schedule()
        self.last_metrics = dict(
            control_wall_ms=control_ms, prefix_match_allocate_ms=self._queue.prefix_ms
        )
        return result

    def postprocess(self, seqs, token_ids):
        self._queue.postprocess(seqs, token_ids)

    def preempt(self, seq):
        self._queue.preempt(seq)

    def cancel(self, seq):
        self._queue.cancel(seq)

    def is_finished(self):
        return self._queue.is_finished()
