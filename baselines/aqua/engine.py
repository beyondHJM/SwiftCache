"""Minimal Aqua + Qwen3 adapter with serialized, chunked HF execution.

No SwiftCache scheduler, allocator, prefix matcher or Layer Stream Cache is used.
Only one request KV context is resident on the consumer between forward calls.
"""

from concurrent.futures import Future
from dataclasses import dataclass, field
from collections import OrderedDict
import threading
import time
import uuid

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache

from .runtime import Client, Producer, Store, Trace


def cache_bytes(cache):
    if cache is None:
        return 0
    return sum(
        t.numel() * t.element_size() for pair in cache.to_legacy_cache() for t in pair
    )


def pack_cache(cache):
    legacy = cache.to_legacy_cache()
    if not legacy:
        raise ValueError("Cannot archive an empty KV cache")
    if any(k.shape != legacy[0][0].shape or v.shape != k.shape for k, v in legacy):
        raise ValueError(
            "Minimal Aqua adapter requires uniform, non-sliding Qwen3 KV layers"
        )
    return torch.stack([torch.stack((k, v)) for k, v in legacy]).contiguous()


def unpack_cache(packed):
    return DynamicCache.from_legacy_cache(
        tuple((packed[i, 0], packed[i, 1]) for i in range(packed.shape[0]))
    )


@dataclass(eq=False)
class Request:
    token_ids: list
    max_tokens: int
    role: str = "consumer"
    prefix_id: str | None = None
    ident: str = field(default_factory=lambda: uuid.uuid4().hex)
    arrival_ns: int = field(default_factory=time.monotonic_ns)
    future: Future = field(default_factory=Future)
    generated: list = field(default_factory=list)
    pos: int = 0
    prefill_work: int = 0
    cached_tokens: int = 0
    next_token: int | None = None
    cache: object = None
    swapped: object = None
    first_token_ns: int | None = None
    output_times_ns: list = field(default_factory=list)
    cancelled: bool = False

    @property
    def prefill(self):
        return self.pos < len(self.token_ids)


def plan_cfs(requests, token_budget, decode_slots, chunk_size):
    """Paper §7: reserve decode slots, rank each phase by attained work.

    This returns a logical batch plan. The minimal HF executor runs its entries
    serially; it does not implement the upstream vLLM batched CUDA kernels.
    """
    if not 1 <= decode_slots < token_budget or chunk_size < 1:
        raise ValueError(
            "Require 1 <= decode_slots < token_budget and positive chunk size"
        )
    prefills = sorted(
        (r for r in requests if r.prefill),
        key=lambda r: (r.prefill_work, r.arrival_ns, r.ident),
    )
    decodes = sorted(
        (r for r in requests if not r.prefill),
        key=lambda r: (len(r.generated), r.arrival_ns, r.ident),
    )[:decode_slots]
    budget = token_budget - decode_slots
    selected = []
    for r in prefills:
        count = min(chunk_size, len(r.token_ids) - r.pos, budget)
        if count:
            selected.append([r, count])
            budget -= count
        if budget == 0:
            break
    # Unused decode slots are available to prefills, as in Aqua's partition.
    extra = decode_slots - len(decodes)
    for r in prefills:
        if extra == 0:
            break
        entry = next((a for a in selected if a[0] is r), None)
        already = entry[1] if entry else 0
        count = min(extra, len(r.token_ids) - r.pos - already, chunk_size - already)
        if count > 0:
            if entry:
                entry[1] += count
            else:
                selected.append([r, count])
            extra -= count
    return [(r, n) for r, n in selected] + [(r, 1) for r in decodes]


class Model:
    def __init__(self, path, device, max_model_len=8192, ignore_model_max_length=False):
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        config = AutoConfig.from_pretrained(path)
        if config.model_type != "qwen3":
            raise ValueError("This baseline adapter is validated for Qwen3 only")
        self.original_limit = config.max_position_embeddings
        self.max_len = (
            max_model_len
            if ignore_model_max_length
            else min(max_model_len, self.original_limit)
        )
        if ignore_model_max_length:
            config.max_position_embeddings = max(self.original_limit, max_model_len)
        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            config=config,
            torch_dtype=torch.bfloat16,
            attn_implementation="sdpa",
            device_map={"": str(device)},
        ).eval()
        if self.model.config.model_type != "qwen3":
            raise ValueError("This baseline adapter is validated for Qwen3 only")
        eos = self.model.generation_config.eos_token_id
        self.eos = set(eos if isinstance(eos, list) else [eos])

    def ids(self, prompt=None, messages=None):
        if messages is not None:
            if not messages or any(
                m.get("role") not in ("system", "user", "assistant")
                or not isinstance(m.get("content"), str)
                for m in messages
            ):
                raise ValueError(
                    "Only nonempty text system/user/assistant messages are supported"
                )
            return self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
        if isinstance(prompt, str):
            return self.tokenizer.encode(prompt)
        if isinstance(prompt, list):
            size = len(self.tokenizer)
            if prompt and all(type(t) is int and 0 <= t < size for t in prompt):
                return prompt.copy()
        raise ValueError("Expected prompt text or token-ID list")

    @torch.inference_mode()
    def forward(self, ids, cache):
        past = cache.get_seq_length() if cache is not None else 0
        with torch.cuda.device(self.device):
            out = self.model(
                input_ids=torch.tensor([ids], device=self.device),
                past_key_values=cache,
                cache_position=torch.arange(past, past + len(ids), device=self.device),
                attention_mask=torch.ones(
                    (1, past + len(ids)), dtype=torch.long, device=self.device
                ),
                use_cache=True,
                logits_to_keep=1,
            )
            if not torch.isfinite(out.logits).all():
                raise RuntimeError("Nonfinite Qwen3 logits")
            next_token = int(out.logits[0, -1].argmax().item())
            return out.past_key_values, next_token

    @torch.inference_mode()
    def reference(self, ids, max_tokens):
        tensor = torch.tensor([ids], device=self.device)
        out = self.model.generate(
            tensor,
            attention_mask=torch.ones_like(tensor),
            do_sample=False,
            max_new_tokens=max_tokens,
            pad_token_id=self.tokenizer.eos_token_id,
        )
        return out[0, len(ids) :].tolist()


class Engine:
    def __init__(
        self,
        model_path,
        physical_gpus,
        coordinator_port=18101,
        offer_bytes=512 << 20,
        host_limit_bytes=2 << 30,
        chunk_size=128,
        token_budget=256,
        decode_slots=1,
        producer_model=True,
        max_model_len=8192,
        high=10,
        low=2,
        ignore_model_max_length=False,
    ):
        self.trace = Trace()
        self.lock = threading.RLock()
        self.jobs = []
        self.prefixes = OrderedDict()
        self.closed = False
        self.cleaned = False
        self.fatal = None
        self.resident = None
        self.producer_resident = None
        self.thread = None
        self.chunk_size = chunk_size
        self.token_budget = token_budget
        self.decode_slots = decode_slots
        if chunk_size < 1 or not 1 <= decode_slots < token_budget:
            raise ValueError("Invalid scheduling budgets")
        self.consumer = Model(
            model_path, "cuda:0", max_model_len, ignore_model_max_length
        )
        self.producer_model = (
            Model(model_path, "cuda:1", max_model_len, ignore_model_max_length)
            if producer_model
            else None
        )
        free, _ = torch.cuda.mem_get_info(1)
        if offer_bytes < 0 or free < offer_bytes + (1 << 30):
            raise ValueError("Insufficient provider memory for configured offer")
        self.client = Client("127.0.0.1", coordinator_port, physical_gpus[0])
        self.producer = Producer(
            self.client,
            physical_gpus[1],
            physical_gpus[0],
            offer_bytes,
            self.trace,
            high,
            low,
        )
        self.store = Store(
            self.client,
            {physical: logical for logical, physical in enumerate(physical_gpus)},
            host_limit_bytes=host_limit_bytes,
            trace=self.trace,
        )
        self.mode = "cfs"
        self.iteration = 0
        self.trace.log(
            "initialized",
            physical_gpus=physical_gpus,
            producer_model=producer_model,
            p2p=bool(torch.cuda.can_device_access_peer(0, 1)),
            offer_bytes=offer_bytes,
        )

    def submit(self, ids, max_tokens=32, role="consumer", prefix_id=None):
        model = self.consumer if role == "consumer" else self.producer_model
        if role not in ("consumer", "producer") or model is None:
            raise ValueError("Unavailable model role")
        if (
            not ids
            or type(max_tokens) is not int
            or max_tokens < 1
            or len(ids) + max_tokens > model.max_len
        ):
            raise ValueError(
                "Request must be nonempty and fit the runtime context limit"
            )
        req = Request(list(ids), max_tokens, role, prefix_id)
        with self.lock:
            if self.closed or self.fatal:
                raise RuntimeError("Baseline is stopped or unhealthy")
            if len(self.jobs) >= 64:
                raise ValueError("Minimal baseline pending-request limit is 64")
            if prefix_id is not None:
                if role != "consumer" or prefix_id not in self.prefixes:
                    raise ValueError("Unknown prefix ID")
                prefix, _, _ = self.prefixes[prefix_id]
                if req.token_ids[: len(prefix)] != prefix:
                    raise ValueError("Prefix ID does not match request tokens")
            self.jobs.append(req)
        self.trace.log(
            "arrival", request_id=req.ident, role=role, input_tokens=len(ids)
        )
        return req

    def cache_prefix(self, ids, ident=None):
        if self.jobs or self.resident or self.thread is not None:
            raise RuntimeError("Create prefixes before submitting the workload")
        if not ids or len(ids) >= self.consumer.max_len:
            raise ValueError("Prefix exceeds context limit")
        cache = None
        next_token = None
        for offset in range(0, len(ids), self.chunk_size):
            cache, next_token = self.consumer.forward(
                ids[offset : offset + self.chunk_size], cache
            )
        packed = pack_cache(cache)
        rt = self.store.put(packed, "prefix")
        ident = ident or uuid.uuid4().hex
        if ident in self.prefixes:
            self.store.close(self.prefixes[ident][1])
        self.prefixes[ident] = (list(ids), rt, next_token)
        del cache, packed
        self.store.flush()
        self.trace.log("prefix_cached", prefix_id=ident, tokens=len(ids))
        return ident

    def _pause(self, req):
        if req is None or req.cache is None:
            return True
        if not self.store.can_fit(cache_bytes(req.cache)):
            return False
        packed = pack_cache(req.cache)
        req.swapped = self.store.put(packed)
        req.cache = None
        del packed
        self.store.flush()
        self.trace.log(
            "preempt", request_id=req.ident, tokens=req.pos + len(req.generated)
        )
        return True

    def _activate(self, req):
        if self.resident is req:
            return True
        if self.resident is not None and not self._pause(self.resident):
            return False
        self.resident = req
        if req.swapped is not None:
            packed = self.store.restore(req.swapped)
            req.cache = unpack_cache(packed)
            self.store.close(req.swapped)
            req.swapped = None
            self.trace.log("resume", request_id=req.ident)
        elif req.prefix_id is not None and req.pos == 0:
            prefix, rt, next_token = self.prefixes[req.prefix_id]
            req.cache = unpack_cache(self.store.restore(rt))
            req.next_token = next_token
            req.pos = len(prefix)
            req.cached_tokens = len(prefix)
            self.trace.log("prefix_hit", request_id=req.ident, tokens=len(prefix))
        return True

    def _finish(self, req, error=None):
        req.cache = None
        if req.swapped is not None:
            self.store.close(req.swapped)
            req.swapped = None
        if self.resident is req:
            self.resident = None
        if self.producer_resident is req:
            self.producer_resident = None
        self.store.flush()
        ended = time.monotonic_ns()
        result = dict(
            id=req.ident,
            role=req.role,
            token_ids=req.generated,
            text=self.consumer.tokenizer.decode(
                req.generated, skip_special_tokens=True
            ),
            prompt_tokens=len(req.token_ids),
            cached_tokens=req.cached_tokens,
            completion_tokens=len(req.generated),
            ttft_ms=(req.first_token_ns - req.arrival_ns) / 1e6
            if req.first_token_ns
            else None,
            total_ms=(ended - req.arrival_ns) / 1e6,
            output_times_ns=req.output_times_ns,
            finish_reason="stop"
            if req.generated and req.generated[-1] in self.consumer.eos
            else "length",
        )
        with self.lock:
            if req in self.jobs:
                self.jobs.remove(req)
        if not req.future.done():
            if error:
                req.future.set_exception(error)
            else:
                req.future.set_result(result)
        self.trace.log(
            "finished" if not error else "failed", request_id=req.ident, role=req.role
        )

    def _step(self, req, count):
        model = self.consumer if req.role == "consumer" else self.producer_model
        if req.cancelled or req.future.cancelled():
            self._finish(req, RuntimeError("Cancelled"))
            return
        phase = "prefill" if req.prefill else "decode"
        if req.prefill:
            count = min(count, len(req.token_ids) - req.pos)
            req.cache, req.next_token = model.forward(
                req.token_ids[req.pos : req.pos + count], req.cache
            )
            req.pos += count
            req.prefill_work += count
        else:
            token = req.next_token
            req.generated.append(token)
            now = time.monotonic_ns()
            req.output_times_ns.append(now)
            if req.first_token_ns is None:
                req.first_token_ns = now
            if token in model.eos or len(req.generated) == req.max_tokens:
                self._finish(req)
                return
            req.cache, req.next_token = model.forward([token], req.cache)
        self.trace.log(
            "forward", request_id=req.ident, role=req.role, phase=phase, tokens=count
        )

    def tick(self):
        if self.closed:
            return
        self.iteration += 1
        with self.lock:
            jobs = list(self.jobs)
        for req in jobs:
            if req.cancelled or req.future.cancelled():
                self._finish(req, RuntimeError("Cancelled"))
        with self.lock:
            jobs = list(self.jobs)
        producers = [r for r in jobs if r.role == "producer"]
        used = (
            cache_bytes(self.producer_resident.cache) if self.producer_resident else 0
        )
        self.producer.poll(len(producers), used)
        # Original Aqua manager responds only at inference loop safe points.
        self.store.respond()
        self.producer.poll(len(producers), used)
        if producers and self.producer.state != "reclaiming":
            req = self.producer_resident or min(producers, key=lambda r: r.arrival_ns)
            self.producer_resident = req
            self._step(
                req,
                min(self.chunk_size, len(req.token_ids) - req.pos)
                if req.prefill
                else 1,
            )
        consumers = [r for r in jobs if r.role == "consumer" and not r.future.done()]
        if not consumers:
            return
        mode = "fcfs" if self.store.has_dram() else "cfs"
        if mode != self.mode:
            self.mode = mode
            self.trace.log("scheduler_mode", mode=mode)
        if mode == "fcfs":
            req = self.resident or min(consumers, key=lambda r: r.arrival_ns)
            plan = [(req, self.chunk_size if req.prefill else 1)]
        else:
            plan = plan_cfs(
                consumers, self.token_budget, self.decode_slots, self.chunk_size
            )
        for req, count in plan:
            if req.future.done():
                continue
            if not self._activate(req):
                self.trace.log("host_budget_fcfs")
                if self.resident:
                    self._step(
                        self.resident, self.chunk_size if self.resident.prefill else 1
                    )
                break
            self._step(req, count)

    def drain(self, requests, timeout=300):
        deadline = time.monotonic() + timeout
        while not all(r.future.done() for r in requests):
            if time.monotonic() > deadline:
                raise TimeoutError("Aqua workload timeout")
            self.tick()
        return [r.future.result() for r in requests]

    def loop(self):
        try:
            while not self.closed:
                self.tick()
                with self.lock:
                    idle = not self.jobs
                if idle:
                    time.sleep(0.05)
        except Exception as exc:
            self.fatal = repr(exc)
            for req in list(self.jobs):
                self._finish(req, exc)

    def start(self):
        if self.thread is not None:
            raise RuntimeError("Inference loop already started")
        self.thread = threading.Thread(
            target=self.loop, name="aqua-inference", daemon=True
        )
        self.thread.start()

    def close(self):
        if self.cleaned:
            return
        self.closed = True
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=30)
            if self.thread.is_alive():
                raise RuntimeError("Aqua inference loop did not stop")
        for req in list(self.jobs):
            self._finish(req, RuntimeError("Engine stopped"))
        self.resident = None
        self.producer_resident = None
        for _, rt, _ in self.prefixes.values():
            self.store.close(rt)
        self.prefixes.clear()
        self.store.close_all()
        self.producer.close()
        self.cleaned = True
        self.consumer.model = None
        if self.producer_model:
            self.producer_model.model = None
        self.store.flush()
