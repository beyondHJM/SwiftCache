from collections import deque
import random
import time
from types import SimpleNamespace
import pytest

from swiftcache.engine.elastic import DemandWindow, align_up, elastic_units
from swiftcache.engine.block_manager.block import Block
from swiftcache.engine.block_manager.free_blocks_ids import MasterFreeBlockIds, SlaveFreeBlockIds, MultiFreeBlockIds
from swiftcache.engine.block_manager.block_manager import BlockManagerBase, MasterBlockManager, SlaveBlockManager
from swiftcache.engine.sequence import Sequence
from swiftcache.sampling_params import SamplingParams


def config(role='slave', total=12, initial=1, unit=1):
    return SimpleNamespace(role=role,rank=0 if role=='master' else 1,
        kvcache_block_size=256,num_kvcache_blocks=total,elastic_initial_blocks=initial,
        slave_minimum_scaling_block_count=unit,master_minimum_scaling_block_count=unit,
        elastic_enabled=True,elastic_window_seconds=.01,elastic_check_seconds=0,
        control_timeout_seconds=5,initial_reserved_master=[initial])


def pair(total=12):
    """Actual coordinator state machines with an in-memory message transport."""
    master=object.__new__(MasterBlockManager)
    BlockManagerBase.__init__(master,config('master',total))
    master.blocks=[Block(i) for i in range(total)]
    master.free_block_ids=MultiFreeBlockIds(master.blocks,[0,total],0,master.config,[1])
    master.peers={b'slave1':dict(reserved=1,total=total,worker_unit=1,master_unit=1)}
    master.slave_rank_to_idx={b'slave1':0}
    master.pending={};master.replies={};master.last_ids={b'slave1':0}
    worker=object.__new__(SlaveBlockManager)
    BlockManagerBase.__init__(worker,config('slave',total))
    worker.blocks=[Block(i) for i in range(total)]
    worker.free_block_ids=SlaveFreeBlockIds(worker.blocks,config=worker.config)
    worker.unit=1;worker.pending=None;worker.request_id=0
    worker.window=DemandWindow(.01,time.monotonic);worker.last_check=time.monotonic()
    worker.config.elastic_check_seconds=1000
    incoming=deque();outgoing=deque()
    def drain():
        data=list(incoming);incoming.clear();return data
    master.zmq_server=SimpleNamespace(recv_all_dict_nonblock=drain,
        send_dict=lambda ident,msg:outgoing.append(dict(msg)))
    worker.zmq_client=SimpleNamespace(send_dict=lambda msg:incoming.append((b'slave1',dict(msg))),
        recv_dict_nonblock=lambda:outgoing.popleft() if outgoing else None)
    return master,worker,incoming,outgoing


def check_pair(master,worker):
    master.assert_invariants();worker.assert_invariants()
    # In-flight intervals can be quarantined by both; never owned by both.
    for i in range(len(worker.blocks)):
        assert not (master.free_block_ids.owns(i) and worker.free_block_ids.owns(i))
        assert not (master.blocks[i].ref_count and worker.blocks[i].ref_count)


def test_algorithm_alignment_and_window():
    assert elastic_units(36*256*8*128,40*256*8*128)==(10,9)
    assert align_up(11,9)==18
    clock=[0.0]
    window=DemandWindow(10,lambda:clock[0])
    window.observe(8);clock[0]=5;window.observe(2)
    assert window.maximum()==8
    clock[0]=11
    assert window.maximum()==2
    clock[0]=16
    assert window.maximum()==0
    with pytest.raises(ValueError): elastic_units(0,4)


def test_lent_count_excludes_active_local_allocations():
    cfg=config(total=12,initial=4,unit=2)
    blocks=[Block(i) for i in range(12)]
    free=SlaveFreeBlockIds(blocks,config=cfg,minimum_scaling_block_count=2)
    free.remove(0);blocks[0].ref_count=1
    assert free.num_blocks_lent==8
    assert len(free)==3
    with pytest.raises(RuntimeError): free.set_reserved(0)
    assert free.init_block_num==4


def test_global_ids_with_nonzero_group_offsets():
    cfg=config('master',total=10)
    cfg.initial_reserved_master=[1,2]
    blocks=[Block(i) for i in range(10)]
    free=MultiFreeBlockIds(blocks,[0,4,8],2,cfg,[1,1])
    assert free.multi_free_block_ids[1]._free=={6,7}
    free.remove(6);free.append(6)
    free.remove(8);free.append(8)
    assert len(free)==7


def test_active_master_reclaim_and_duplicate_messages():
    master,worker,incoming,outgoing=pair()
    active=master._allocate_block(1)
    master.register_hash(active,123,[7]*256)
    worker.request_resize(5)
    request=dict(worker.pending['message'])
    master.poll()
    worker.poll()
    assert worker.pending is not None and len(master.pending)==1
    assert 123 not in master.hash_to_block_id
    assert not master.free_block_ids.owns(1)
    extra=master._allocate_block(master.free_block_ids[0])
    assert extra.block_id>=5
    check_pair(master,worker)
    active.ref_count=0;master._deallocate_block(1)
    master.poll();worker.poll()
    assert worker.free_block_ids.init_block_num==5
    incoming.append((b'slave1',request))
    master.poll();worker.poll()
    assert master.peers[b'slave1']['reserved']==5
    assert not worker.pending and not master.pending
    check_pair(master,worker)


def test_worker_shrink_invalidates_cached_data_before_handoff():
    master,worker,_,_=pair()
    worker.request_resize(6);master.poll();worker.poll()
    block=worker._allocate_block(5);worker.register_hash(block,789,[9]*256)
    block.ref_count=0;worker._deallocate_block(5)
    worker.request_resize(1)
    assert 789 not in worker.hash_to_block_id
    assert not worker.free_block_ids.owns(5)
    master.poll();worker.poll()
    assert master.free_block_ids.owns(5)
    check_pair(master,worker)


def test_shrink_protects_worker_highwater():
    master,worker,_,_=pair()
    worker.request_resize(6);master.poll();worker.poll()
    worker._allocate_block(4)
    worker.config.elastic_check_seconds=0
    worker.poll()
    assert worker.pending['message']['target']==5
    master.poll();worker.poll()
    assert worker.free_block_ids.owns(4)
    check_pair(master,worker)


def test_exact_block_prefix_and_stale_hash_eviction():
    master,worker,_,_=pair()
    worker.request_resize(10);master.poll();worker.poll()
    tokens=[11]*256+[12]*256
    first=Sequence(tokens,SamplingParams(temperature=0,max_tokens=2))
    worker.allocate(first);worker.deallocate(first)
    again=Sequence(tokens,SamplingParams(temperature=0,max_tokens=2))
    worker.allocate(again)
    # Recompute the last full block so prefill is never empty.
    assert again.num_cached_tokens==256
    worker.deallocate(again)
    worker.request_resize(1);master.poll();worker.poll()
    worker.request_resize(10);master.poll();worker.poll()
    check_pair(master,worker)
    assert all(worker.blocks[i].hash==h for h,i in worker.hash_to_block_id.items())


def test_random_handoffs_do_not_overlap_or_leak():
    master,worker,_,_=pair(total=24)
    rng=random.Random(20261108)
    active_m=[];active_w=[]
    for _ in range(1000):
        choice=rng.randrange(5)
        if choice==0 and not worker.pending:
            highwater=max(worker.used_block_ids,default=-1)+1
            worker.request_resize(rng.randint(max(1,highwater),24))
        elif choice==1 and len(master.free_block_ids):
            active_m.append(master._allocate_block(master.free_block_ids[0]).block_id)
        elif choice==2 and not worker.pending and len(worker.free_block_ids):
            active_w.append(worker._allocate_block(worker.free_block_ids[0]).block_id)
        elif choice==3 and active_m:
            i=active_m.pop(rng.randrange(len(active_m)))
            master.blocks[i].ref_count=0;master._deallocate_block(i)
        elif choice==4 and active_w:
            i=active_w.pop(rng.randrange(len(active_w)))
            worker.blocks[i].ref_count=0;worker._deallocate_block(i)
        master.poll();worker.poll();check_pair(master,worker)
    for manager,active in [(master,active_m),(worker,active_w)]:
        for i in active:
            manager.blocks[i].ref_count=0;manager._deallocate_block(i)
    master.poll();worker.poll();check_pair(master,worker)
    assert not master.used_block_ids and not worker.used_block_ids
    assert len(master.free_block_ids)+len(worker.free_block_ids)==24


def test_draining_live_block_cannot_reenter_prefix_cache():
    master,worker,_,_=pair()
    block=master._allocate_block(1)
    master.register_hash(block,123,[1]*256)
    worker.request_resize(4);master.poll();worker.poll()
    # The running decoder completes another full block while reclaim is pending.
    master.register_hash(block,456,[2]*256)
    assert block.hash==456  # still usable by the running sequence's hash chain
    assert 456 not in master.hash_to_block_id  # unavailable to new sequences
    check_pair(master,worker)


def test_invalid_resize_fails_before_mutating_ownership():
    master,worker,incoming,_=pair()
    before=master.free_block_ids.multi_free_block_ids[0]._free.copy()
    incoming.append((b'slave1',dict(op='resize',request_id=1,old=1,target=99)))
    with pytest.raises(RuntimeError,match='invalid resize'):
        master.poll()
    assert master.free_block_ids.multi_free_block_ids[0]._free==before
    check_pair(master,worker)
