"""Prefix allocation and acknowledged, iteration-boundary cache handoff."""
import json
import time
import xxhash
import numpy as np
from swiftcache.engine.block_manager.block import Block
from swiftcache.engine.block_manager.free_blocks_ids import MultiFreeBlockIds, SlaveFreeBlockIds
from swiftcache.engine.elastic import align_up, DemandWindow
from swiftcache.utils.zmq_wrapper import ZMQServer, ZMQClient


class BlockManagerBase:
    def __init__(self, config):
        self.config = config
        self.role = config.role
        self.block_size = config.kvcache_block_size
        self.blocks = []
        self.used_block_ids = set()
        self.hash_to_block_id = {}
        self.block_id_to_hash = {}
        self.quiesce = lambda: None  # installed by LLMEngine before requests
        self.events = []

    def log(self, event, **fields):
        record = dict(event=event, role=self.role, rank=self.config.rank,
                      time=time.time(), **fields)
        self.events.append(record)
        self.events = self.events[-256:]
        print('ELASTIC '+json.dumps(record), flush=True)

    @staticmethod
    def compute_hash(token_ids, prefix=-1):
        h = xxhash.xxh64()
        if prefix != -1:
            h.update(prefix.to_bytes(8,'little'))
        h.update(np.asarray(token_ids,dtype=np.int64).tobytes())
        return h.intdigest()

    def invalidate(self, block_id):
        old = self.block_id_to_hash.pop(block_id, None)
        if old is not None and self.hash_to_block_id.get(old) == block_id:
            del self.hash_to_block_id[old]
        if self.blocks[block_id].ref_count == 0:
            self.blocks[block_id].hash = -1
            self.blocks[block_id].token_ids = []

    def register_hash(self, block, h, tokens):
        self.invalidate(block.block_id)
        block.update(h, list(tokens))
        if self.free_block_ids.owns(block.block_id):
            self.hash_to_block_id[h] = block.block_id
            self.block_id_to_hash[block.block_id] = h

    def _allocate_block(self, block_id):
        block = self.blocks[block_id]
        if block.ref_count or not self.free_block_ids.owns(block_id):
            raise RuntimeError('Allocating a referenced/unowned block')
        self.free_block_ids.remove(block_id)
        self.invalidate(block_id)
        block.reset()
        self.used_block_ids.add(block_id)
        return block

    def _deallocate_block(self, block_id):
        self.used_block_ids.remove(block_id)
        if not self.free_block_ids.owns(block_id):
            self.invalidate(block_id)
        self.free_block_ids.append(block_id)

    def allocate(self, seq):
        assert not seq.block_table
        h = -1
        miss = False
        for i in range(seq.num_blocks):
            tokens = seq.block(i)
            h = self.compute_hash(tokens,h) if len(tokens)==self.block_size else -1
            candidate = self.hash_to_block_id.get(h,-1)
            hit = (not miss and i < seq.num_blocks-1 and candidate >= 0 and
                   self.free_block_ids.owns(candidate) and
                   self.blocks[candidate].token_ids == tokens)
            if hit:
                block = self.blocks[candidate]
                if block.ref_count:
                    block.ref_count += 1
                else:
                    block = self._allocate_block(candidate)
                seq.num_cached_tokens += self.block_size
            else:
                miss = True
                block = self._allocate_block(self.free_block_ids[0])
                block.hit_count = 0
                block.prefix_position = i
            if h != -1:
                self.register_hash(block,h,tokens)
                block.hit_count += 1
            seq.block_table.append(block.block_id)
        seq.extra_info['prefix_cached'] = seq.num_cached_tokens

    def deallocate(self, seq):
        for block_id in reversed(seq.block_table):
            block = self.blocks[block_id]
            if block.ref_count <= 0:
                raise RuntimeError('Invalid block reference count')
            block.ref_count -= 1
            if block.ref_count == 0:
                self._deallocate_block(block_id)
        seq.num_cached_tokens = 0
        seq.block_table.clear()

    def can_append(self, seq):
        return len(self.free_block_ids) >= (len(seq)%self.block_size == 1)

    def may_append(self, seq):
        if len(seq)%self.block_size == 1:
            block = self._allocate_block(self.free_block_ids[0])
            seq.block_table.append(block.block_id)
        elif len(seq)%self.block_size == 0:
            block = self.blocks[seq.block_table[-1]]
            prefix = self.blocks[seq.block_table[-2]].hash if len(seq.block_table)>1 else -1
            self.register_hash(block,self.compute_hash(seq.block(seq.num_blocks-1),prefix),seq.block(seq.num_blocks-1))

    def usage_rate(self):
        return len(self.used_block_ids)/max(1,len(self.blocks))

    def observe(self, seq):
        pass

    def poll(self):
        pass

    def snapshot(self):
        self.assert_invariants()
        return dict(role=self.role, free=len(self.free_block_ids), used=len(self.used_block_ids),
                    total=len(self.blocks), events=self.events)

    def assert_invariants(self):
        referenced = {b.block_id for b in self.blocks if b.ref_count > 0}
        assert referenced == self.used_block_ids
        groups = getattr(self.free_block_ids,'groups',[self.free_block_ids])
        for group in groups:
            assert all(group.owns(i) and self.blocks[i].ref_count==0 for i in group._free)
        assert all(self.blocks[i].hash == h and self.free_block_ids.owns(i)
                   for h,i in self.hash_to_block_id.items())


class MasterBlockManager(BlockManagerBase):
    def __init__(self, config):
        super().__init__(config)
        self.zmq_server = ZMQServer()
        self.slave_rank_to_idx = {f'slave{rank}'.encode():i for i,rank in enumerate(config.slave_list)}
        self.peers = {}
        deadline = time.monotonic()+config.control_timeout_seconds
        while len(self.peers) < len(config.slave_list):
            msg = self.zmq_server.recv_dict_nonblock()
            if msg:
                ident,info=msg
                if info.get('op') != 'hello' or ident not in self.slave_rank_to_idx:
                    raise RuntimeError('Invalid worker handshake')
                self.peers[ident]=info
            elif time.monotonic()>deadline:
                raise TimeoutError('Worker cache handshake timed out')
            else:
                time.sleep(.01)
        ordered = [self.peers[f'slave{r}'.encode()] for r in config.slave_list]
        self.minimum_scaling_block_count = [p['master_unit'] for p in ordered]
        config.initial_reserved_master = [p['reserved']//p['worker_unit']*p['master_unit'] for p in ordered]
        boundaries = config.external_kvcache_config.num_blocks_start_end
        for i,p in enumerate(ordered):
            assert p['total']//p['worker_unit']*p['master_unit'] == boundaries[i+1]-boundaries[i]
        self.blocks=[Block(i,compare_mode='local_first',external_blocks_num=boundaries[-1])
                     for i in range(config.num_kvcache_blocks)]
        self.free_block_ids=MultiFreeBlockIds(self.blocks,boundaries,config.local_num_blocks,config,self.minimum_scaling_block_count)
        self.pending = {}
        self.replies = {}
        self.last_ids = {ident:0 for ident in self.peers}
        for ident in self.peers:
            self.zmq_server.send_dict(ident,dict(op='ready'))

    def can_allocate(self, seq):
        return len(self.free_block_ids) >= seq.num_blocks

    def master_check_blocks_update(self):
        self.poll()

    def poll(self):
        for ident,req in self.zmq_server.recv_all_dict_nonblock():
            request_id=req.get('request_id')
            key=(ident,request_id)
            if key in self.replies:
                self.zmq_server.send_dict(ident,self.replies[key])
                continue
            if ident not in self.peers or req.get('op') != 'resize':
                raise RuntimeError('Invalid resize control message')
            if ident in self.pending:
                if self.pending[ident] != req:
                    raise RuntimeError('Conflicting in-flight resize')
                continue
            peer=self.peers[ident]
            target=req['target']
            if (request_id != self.last_ids[ident]+1 or req['old'] != peer['reserved'] or
                    target < peer['worker_unit'] or target > peer['total'] or target%peer['worker_unit']):
                raise RuntimeError('Stale or invalid resize boundary')
            idx=self.slave_rank_to_idx[ident]
            group=self.free_block_ids.multi_free_block_ids[idx]
            master_target=target//peer['worker_unit']*peer['master_unit']
            if master_target > group.init_block_num:
                # Quarantine now so new requests cannot keep these blocks busy.
                ids=group.drain(master_target)
                for i in ids:
                    self.invalidate(i)
                active=sum(self.blocks[i].ref_count>0 for i in ids)
                self.log('reclaim_pending',request_id=request_id,target=target,active_blocks=active,worker=ident.decode())
            self.pending[ident]=req
        for ident,req in list(self.pending.items()):
            peer=self.peers[ident]
            idx=self.slave_rank_to_idx[ident]
            group=self.free_block_ids.multi_free_block_ids[idx]
            target=req['target']//peer['worker_unit']*peer['master_unit']
            ids=range(group.offset+min(group.init_block_num,target),group.offset+max(group.init_block_num,target))
            if any(self.blocks[i].ref_count for i in ids):
                # A long live request is progress, not a failed control channel.
                # Heartbeats keep the worker waiting safely without a false timeout.
                if not hasattr(self,'last_wait_sent'):
                    self.last_wait_sent={}
                now=time.monotonic()
                if now-self.last_wait_sent.get(ident,0)>=1:
                    self.zmq_server.send_dict(ident,dict(op='resize_wait',request_id=req['request_id'],target=req['target']))
                    self.last_wait_sent[ident]=now
                continue
            self.quiesce()
            for i in ids:
                self.invalidate(i)
            group.set_reserved(target)
            old=peer['reserved']; peer['reserved']=req['target']
            reply=dict(op='resized',request_id=req['request_id'],reserved=req['target'])
            self.replies[(ident,req['request_id'])]=reply
            # Keep bounded replay history; older IDs fail closed rather than reapply.
            if len(self.replies)>256:
                self.replies.pop(next(iter(self.replies)))
            self.last_ids[ident]=req['request_id']
            del self.pending[ident]
            self.log('resize_committed',request_id=req['request_id'],old=old,reserved=req['target'],worker=ident.decode())
            self.zmq_server.send_dict(ident,reply)

    def snapshot(self):
        result=super().snapshot()
        result.update(worker_reserved={k.decode():v['reserved'] for k,v in self.peers.items()},
                      pending=len(self.pending),draining=sum(len(g.draining) for g in self.free_block_ids.multi_free_block_ids))
        return result


class SlaveBlockManager(BlockManagerBase):
    def __init__(self, config):
        super().__init__(config)
        self.zmq_client=ZMQClient(name=f'slave{config.rank}')
        self.blocks=[Block(i) for i in range(config.num_kvcache_blocks)]
        self.unit=config.slave_minimum_scaling_block_count
        self.free_block_ids=SlaveFreeBlockIds(self.blocks,config=config,minimum_scaling_block_count=self.unit)
        self.pending=None
        self.request_id=0
        self.window=DemandWindow(config.elastic_window_seconds,time.monotonic)
        self.last_check=time.monotonic()
        self.zmq_client.send_dict(dict(op='hello',master_unit=config.master_minimum_scaling_block_count,
                                      worker_unit=self.unit,reserved=self.free_block_ids.init_block_num,total=len(self.blocks)))
        deadline=time.monotonic()+config.control_timeout_seconds
        while True:
            msg=self.zmq_client.recv_dict_nonblock()
            if msg:
                assert msg['op']=='ready'
                break
            if time.monotonic()>deadline:
                raise TimeoutError('Master cache handshake timed out')
            time.sleep(.01)

    def observe(self, seq):
        # Include output headroom; concurrent live allocations are protected below.
        self.window.observe((len(seq)+seq.max_tokens+self.block_size-1)//self.block_size)

    def request_resize(self, target):
        if self.pending or not self.config.elastic_enabled:
            return
        old=self.free_block_ids.init_block_num
        target=min(len(self.blocks),align_up(max(self.unit,target),self.unit))
        if target==old:
            return
        self.quiesce()
        if target<old:
            # Relinquish locally BEFORE master may expose the returned interval.
            for i in range(target,old):
                if self.blocks[i].ref_count:
                    raise RuntimeError('Cannot shrink active worker tail')
                self.invalidate(i)
            self.free_block_ids.set_reserved(target)
        self.request_id+=1
        msg=dict(op='resize',request_id=self.request_id,old=old,target=target)
        self.pending=dict(message=msg,since=time.monotonic(),sent=time.monotonic())
        self.log('resize_requested',request_id=self.request_id,old=old,target=target)
        self.zmq_client.send_dict(msg)

    def poll(self):
        while True:
            msg=self.zmq_client.recv_dict_nonblock()
            if msg is None:
                break
            if msg.get('request_id',0)<self.request_id:
                continue
            if self.pending is None:
                # Duplicate acknowledgment of the latest committed transaction.
                if msg.get('request_id')==self.request_id:
                    continue
                raise RuntimeError('Unexpected resize acknowledgment')
            req=self.pending['message']
            if msg == dict(op='resize_wait',request_id=req['request_id'],target=req['target']):
                self.pending['progress']=time.monotonic()
                continue
            if msg != dict(op='resized',request_id=req['request_id'],reserved=req['target']):
                raise RuntimeError('Mismatched resize acknowledgment')
            self.quiesce()
            if req['target']>req['old']:
                for i in range(req['old'],req['target']):
                    self.invalidate(i)
                self.free_block_ids.set_reserved(req['target'])
            self.log('resize_applied',request_id=req['request_id'],old=req['old'],reserved=req['target'])
            self.pending=None
        now=time.monotonic()
        if self.pending:
            if now-self.pending.get('progress',self.pending['since'])>self.config.control_timeout_seconds:
                raise TimeoutError('Resize acknowledgment timed out; ownership remains quarantined')
            if now-self.pending['sent']>1:
                self.zmq_client.send_dict(self.pending['message'])
                self.pending['sent']=now
            return
        if self.config.elastic_enabled and now-self.last_check>=self.config.elastic_check_seconds:
            self.last_check=now
            highwater=max(self.used_block_ids,default=-1)+1
            target=align_up(max(self.unit,self.config.elastic_initial_blocks,self.window.maximum(),highwater,getattr(self,"request_floor",0)),self.unit)
            if target<self.free_block_ids.init_block_num:
                self.request_resize(target)

    def can_allocate(self, seq):
        if self.pending:
            return False
        if len(self.free_block_ids)>=seq.num_blocks:
            return True
        need=self.free_block_ids.init_block_num+seq.num_blocks-len(self.free_block_ids)
        if need<=len(self.blocks):
            self.request_resize(need)
        return False

    def can_append(self, seq):
        if super().can_append(seq):
            return True
        if not self.pending:
            self.request_resize(self.free_block_ids.init_block_num+1)
        return False

    def snapshot(self):
        result=super().snapshot()
        result.update(reserved=self.free_block_ids.init_block_num,pending=self.pending is not None)
        return result


def create_block_manager(config):
    if config.role=='master':
        return MasterBlockManager(config)
    if config.role=='slave':
        return SlaveBlockManager(config)
    raise ValueError(f'Unknown role: {config.role}')
