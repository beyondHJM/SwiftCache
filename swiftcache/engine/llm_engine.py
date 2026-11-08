import os

USE_CSCHEDULER = os.environ.get("USE_CSCHEDULER", "").lower() in ("1", "true", "yes")

import atexit
from dataclasses import fields
import time
from time import perf_counter
from tqdm.auto import tqdm
from transformers import AutoTokenizer
import torch.multiprocessing as mp
import torch
import torch.distributed as dist
from multiprocessing.shared_memory import SharedMemory
from swiftcache.config import Config
from swiftcache.sampling_params import SamplingParams
from swiftcache.engine.sequence import Sequence
from swiftcache.engine.scheduler_factory import get_scheduler_class

from swiftcache.engine.model_runner import ModelRunner
from swiftcache.engine.external_kvcache import ExternalKVCacheManager
from swiftcache.utils.common_utils import init_vl_processor
from torch.cuda.nvtx import range_push, range_pop
from swiftcache.engine.request import Request
from swiftcache.global_config import global_config
import asyncio
import json




class LLMEngine:

    def __init__(self, model, **kwargs):
        print("start init")
        config_fields = {field.name for field in fields(Config)}
        config_kwargs = {k: v for k, v in kwargs.items() if k in config_fields}
        if USE_CSCHEDULER and 'scheduler_backend' not in config_kwargs:
            config_kwargs['scheduler_backend'] = 'cpp'
        config = Config(model, **config_kwargs)
        scheduler_class = get_scheduler_class(config.scheduler_backend)
        self.config = config
        self.seqs_in_scheduler = {}
        self.request_dict = {}
        self.batch_counter = 0
        # self.num_external_kvcache = global_config['num_external_kvcache']
        vl_chat_processor = init_vl_processor(model)
        self.is_vision_language = vl_chat_processor is not None 
        self.ps = []
        self.events = []

        ctx = mp.get_context("spawn")
        for i in range(1, config.tensor_parallel_size):
            event = ctx.Event()
            process = ctx.Process(target=ModelRunner, args=(config, i, event))
            process.start()
            self.ps.append(process)
            self.events.append(event)

        self.model_runner = ModelRunner(config, config.rank, self.events,vl_chat_processor)
        self.tokenizer = AutoTokenizer.from_pretrained(config.model, use_fast=True)
        config.eos = self.tokenizer.eos_token_id
        print(f"{config.role}{config.rank}调度器开始创建")
        self.scheduler = scheduler_class(config)
        self.scheduler.block_manager.quiesce = self.model_runner.quiesce
        print(f"{config.role}{config.rank}调度器创建成功")
        # time.sleep(5)
        self.warm_up()
        self.request_dict = {}
        atexit.register(self.exit)

        print("end init")

    def exit(self):
        if global_config['kv_cache_strategy'] == 'external_gpu' and self.config.role == 'master':
            self.model_runner.controller.exit()
        self.model_runner.call("exit")
        del self.model_runner
        for p in self.ps:
            p.join()

        

    def prepare_token_ids(self,input_embeds:torch.Tensor):
        sum_input_embeds = input_embeds.sum(dim=-1).tolist()
        for i in range(len(sum_input_embeds)):
            sum_input_embeds[i]=int(sum_input_embeds[i]*10000%10000)
        return sum_input_embeds
        
    def add_request(self, prompt: str | list[int], sampling_params: SamplingParams,is_vision_language:bool):
        # Only for language-only models: encode string prompt into token IDs
        if is_vision_language:
            seq = VisionSequence(prompt, sampling_params)
        else:
            if isinstance(prompt, str):
                prompt = self.tokenizer.encode(prompt)
                
            if not prompt or len(prompt)+sampling_params.max_tokens>self.config.max_model_len:
                raise ValueError('Request exceeds runtime context limit')
            capacity=self.scheduler.block_manager.maximum_blocks()*self.config.kvcache_block_size
            if len(prompt)+sampling_params.max_tokens-1>capacity:
                raise ValueError('Request exceeds maximum usable KV capacity')
            seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        self.seqs_in_scheduler[seq.seq_id] = seq
        
        print(f'add success!,seq_id:{seq.seq_id}')
   

    def step(self):
        t0=time.perf_counter_ns();cpu0=time.thread_time_ns()
        seqs,kind=self.scheduler.schedule()
        scheduled=time.perf_counter_ns();cpu1=time.thread_time_ns()
        if not seqs: return [],0
        is_prefill=kind=='prefill'
        token_ids=self.model_runner.call('run',seqs,is_prefill)
        for seq,token in zip(seqs,token_ids):
            req=self.request_dict.get(seq.seq_id)
            if req is None: continue
            if is_prefill and 'first_cached_tokens' not in req.metrics:
                req.metrics['first_cached_tokens']=min(seq.num_cached_tokens,seq.num_prompt_tokens)
                req.prefix_cached=req.metrics['first_cached_tokens']
            generated=seq.completion_token_ids+[token]
            text=self.tokenizer.decode(generated,skip_special_tokens=True,clean_up_tokenization_spaces=False)
            positions=[text.find(stop) for stop in req.stop if stop in text]
            if positions:
                text=text[:min(positions)];seq.force_stop=True;req.finish_reason='stop'
            elif not seq.ignore_eos and token==self.config.eos:
                req.finish_reason='stop'
            elif len(generated)>=seq.max_tokens:
                req.finish_reason='length'
            req.visible_text=text
        p0=time.perf_counter_ns()
        self.scheduler.postprocess(seqs,token_ids)
        p1=time.perf_counter_ns()
        self.batch_counter+=1
        metric=dict(batch_id=self.batch_counter,is_prefill=is_prefill,
            scheduler_backend=self.config.scheduler_backend,
            schedule_wall_ms=(scheduled-t0)/1e6,schedule_cpu_ms=(cpu1-cpu0)/1e6,
            postprocess_wall_ms=(p1-p0)/1e6,**getattr(self.scheduler,'last_metrics',{}),
            **getattr(self.model_runner,'last_run_metrics',{}))
        for seq,token in zip(seqs,token_ids):
            seq.time_usage_append((p1-t0)/1e9)
            req=self.request_dict.get(seq.seq_id)
            if req is not None:
                req.output_token_ids=seq.completion_token_ids
                req.metrics['batches'].append(metric)
                req.metrics.setdefault('first_token_ns',time.perf_counter_ns())
                text=req.visible_text
                if not seq.is_finished:
                    # Hold stop-string prefixes and incomplete UTF-8 characters.
                    hold=max((n for stop in req.stop for n in range(1,len(stop)) if text.endswith(stop[:n])),default=0)
                    if hold: text=text[:-hold]
                    text=text.rstrip('\ufffd')
                if req.streaming:
                    req.stream_queue.put_nowait(dict(text=text,finished=seq.is_finished))
                if seq.is_finished:
                    req.output=req.visible_text;req.time_usage=seq.time_usage
                    req.metrics['finished_ns']=time.perf_counter_ns()
                    req.finished_event.set()
            if seq.is_finished:
                self.seqs_in_scheduler.pop(seq.seq_id,None)
        outputs=[(seq.seq_id,seq.completion_token_ids,seq.time_usage,seq.extra_info) for seq in seqs if seq.is_finished]
        return outputs,sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)

    def is_finished(self):
        return self.scheduler.is_finished()

    def generate(
        self,
        prompts: list[str] | list[list[int]],
        sampling_params: SamplingParams | list[SamplingParams],
        is_vision_language = False,
        use_tqdm: bool = True,
    ) -> list[str]:
        if use_tqdm:
            total = 1 if is_vision_language else len(prompts)
            pbar = tqdm(total=total, desc="Generating", dynamic_ncols=True)
        
        if not isinstance(sampling_params, list) and not is_vision_language:
            sampling_params = [sampling_params] * len(prompts)
        if is_vision_language:
            # 对于视觉语言模型，目前只支持batch size = 1
            self.add_request(prompts,sampling_params,True)
        else:
            for prompt, sp in zip(prompts, sampling_params):
                self.add_request(prompt, sp, False)
        outputs = {}
        time_usages={}
        prefill_throughput = decode_throughput = 0.
        cnt=0
        while not self.is_finished():
            t = perf_counter()
            range_push("step")
            output, num_tokens = self.step()
            range_pop()
            # if scheduling_type in [SchedulingType.IMAGE_PROCESSING,SchedulingType.VISION_ENCODING]:
            #     print("not implement",scheduling_type)
            if False:
                pass
            else:
                if use_tqdm:
                    if num_tokens > 0:
                        prefill_throughput = num_tokens / (perf_counter() - t)
                    else:

                        decode_throughput = -num_tokens / (perf_counter() - t)
                    pbar.set_postfix({
                        "Prefill": f"{int(prefill_throughput)}tok/s",
                        "Decode": f"{int(decode_throughput)}tok/s",
                    })
                for seq_id, token_ids,time_usage,_ in output:
                    outputs[seq_id] = token_ids
                    time_usages[seq_id] = time_usage
                    if use_tqdm:
                        pbar.update(1)
        outputs = [(outputs[seq_id],time_usages[seq_id]) for seq_id in sorted(outputs)]
        outputs = [{"text": self.tokenizer.decode(token_ids), "token_ids": token_ids,'time_usages':time_usages} for token_ids,time_usages in outputs]
        if use_tqdm:
            pbar.close()
        return outputs
    
    def warm_up(self):
        sampling_params = SamplingParams(temperature=0, max_tokens=50)
        if self.is_vision_language:
            conversation = [
            {
                "role": "<|User|>",
                "content": "这是一张照片: <image>\n"
                        " 请描述下这张图片?",
                "images": [
                    f"./images/grounding_conversation_1.jpeg",
                ],
            },
            {"role": "<|Assistant|>", "content": ""}
            ]
            self.generate(conversation, sampling_params,is_vision_language=True,use_tqdm=False)
        else:
            prompts = [
                        "A, B, C, D, E",
                        # "One Two Three Four Five",
                        # "Monday Tuesday Wednesday, Thursday, Friday",
                    ]
            sampling_params = SamplingParams(temperature=0, max_tokens=10)
            self.generate(prompts, sampling_params)
        
        print("warm up end!")
        # if self.config.role=='master':
        #     self.test()

    async def _main_event_loop(self):
        while True:
            self.scheduler.block_manager.poll()
            if not self.is_finished():
                try:
                    _,tokens=self.step()
                except Exception as exc:
                    for req in self.request_dict.values():
                        req.error=str(exc);req.finished_event.set()
                    raise
                await asyncio.sleep(0 if tokens else .001)
            else:
                await asyncio.sleep(.001)

    def cancel_request(self,request_id):
        seq=self.seqs_in_scheduler.pop(request_id,None)
        if seq is not None:
            self.scheduler.cancel(seq)
        req=self.request_dict.pop(request_id,None)
        if req is not None:
            req.cancelled=True;req.finished_event.set()

    async def process_request(self,req:Request):
        begin=time.perf_counter_ns()
        token_ids=req.token_ids if req.token_ids is not None else self.tokenizer.encode(req.prompt)
        req.metrics['tokenize_wall_ms']=(time.perf_counter_ns()-begin)/1e6
        req.token_ids=token_ids
        if not token_ids or req.max_tokens<=0 or len(token_ids)+req.max_tokens>self.config.max_model_len:
            raise ValueError('Request must be nonempty and fit runtime context limit including output')
        capacity=self.scheduler.block_manager.maximum_blocks()*self.config.kvcache_block_size
        if len(token_ids)+req.max_tokens-1>capacity:
            raise ValueError('Request exceeds physical KV capacity')
        sampling=SamplingParams(req.temperature,req.max_tokens,req.ignore_eos,req.top_p)
        seq=Sequence(token_ids,sampling,seq_id=req.request_id)
        req.metrics['enqueue_ns']=time.perf_counter_ns()
        self.request_dict[req.request_id]=req
        self.seqs_in_scheduler[seq.seq_id]=seq
        self.scheduler.add(seq)
        try:
            await req.finished_event.wait()
            if req.cancelled: raise asyncio.CancelledError()
            if getattr(req,'error',None): raise RuntimeError(req.error)
        finally:
            if not req.finished_event.is_set(): self.cancel_request(req.request_id)
            self.request_dict.pop(req.request_id,None)
