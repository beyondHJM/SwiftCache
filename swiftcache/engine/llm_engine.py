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
from swiftcache.engine.sequence import sequence_cpp_to_py
if USE_CSCHEDULER:
    print("使用c++编写的scheduler!")
    from swiftcache.engine.c_scheduler import Scheduler, Sequence
else:
    print("使用python编写的scheduler!")
    from swiftcache.engine.scheduler import Scheduler
    from swiftcache.engine.sequence import Sequence

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
        config = Config(model, **config_kwargs)
        self.config = config
        self.seqs_in_scheduler = {}
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
        self.scheduler = Scheduler(config)
        print(f"{config.role}{config.rank}调度器创建成功")
        # time.sleep(5)
        self.warm_up()
        self.request_dict = {}
        atexit.register(self.exit)

        print("end init")

    def exit(self):
        if global_config['kv_cache_strategy'] == 'external_gpu':
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
                
            if len(prompt) > self.tokenizer.model_max_length:
                orig_len = len(prompt)
                max_len = self.tokenizer.model_max_length
                prompt = prompt[-max_len:]
                print(f"[Warning] Prompt length {orig_len} exceeds the maximum limit ({max_len}). Truncated to the last {max_len} characters.")
            seq = Sequence(prompt, sampling_params)
        self.scheduler.add(seq)
        self.seqs_in_scheduler[seq.seq_id] = seq
        
        print(f'add success!,seq_id:{seq.seq_id}')
   

    def step(self):
        # t1= perf_counter()
        t = perf_counter()
        seqs, scheduling_type = self.scheduler.schedule()
        
        if USE_CSCHEDULER:
            seqs_py = sequence_cpp_to_py(seqs)
        print(f'{self.config.role} scheduler时间:{(perf_counter()-t)*1000:.3f} ms')
            # for seq in seqs:
            #     self.seqs_in_scheduler[seq.seq_id] = seq

        # print('scheduled success!',scheduling_type)
        # print(json.dumps(self.scheduler.block_manager.hash_to_block_id,indent = ' '))
        # print('scheudler time usage:',perf_counter()-t)

        # if scheduling_type is SchedulingType.IMAGE_PROCESSING:

        #     assert len(seqs) == 1 , "Number of sequences must be 1 in the phase for image processing."
        #     seq = seqs[0]
        #     prepare_inputs = self.model_runner.image_process(seq)
        #     seq = VisionSequence(prepare_inputs = prepare_inputs,status = VisionStatus.ENCODING,sampling_params=seq.sampling_params)
        #     seq.time_usage.append(perf_counter()-t)
        #     self.scheduler.add(seq)
        #     return prepare_inputs, None , scheduling_type

        # elif scheduling_type is SchedulingType.VISION_ENCODING:
        #     assert len(seqs) == 1, "Number of sequences must be 1 in the phase for vision encoding."
        #     seq  = seqs[0]
        #     prepare_inputs = seq.prepare_inputs.to('cuda:0')
        #     input_embeds = self.model_runner.prepare_inputs_embeds(prepare_inputs)
        #     input_embeds = input_embeds.view(-1, input_embeds.size(-1))
        #     seq.time_usage.append(perf_counter()-t)
        #     time_usage = seq.time_usage
        #     seq = Sequence(token_ids = self.prepare_token_ids(input_embeds), sampling_params=seq.sampling_params, input_embeds = input_embeds)
        #     seq.time_usage = time_usage
        #     self.scheduler.add(seq)
        #     return None, None , scheduling_type
        # print(f"xxxxx:{scheduling_type == SchedulingType.PREFILL or scheduling_type == SchedulingType.DECODING}")
        # print(scheduling_type, "prefill", "decoding")
        if scheduling_type == "prefill" or scheduling_type == "decoding":

            is_prefill = True if scheduling_type == "prefill" else False
            if USE_CSCHEDULER:
                token_ids = self.model_runner.call("run", seqs_py, is_prefill)
            else:
                token_ids = self.model_runner.call("run", seqs, is_prefill)
            t2 = perf_counter()
            self.scheduler.postprocess(seqs, token_ids)
            end_time = perf_counter()
            # print(f"postprocess:{(end_time-t2)*1000:.3f} ms")
            # 这部分可以用c++
            for seq in seqs:
                seq.time_usage_append(end_time - t)
                if seq.is_finished:
                    del self.seqs_in_scheduler[seq.seq_id]


            outputs = [(seq.seq_id, seq.completion_token_ids,seq.time_usage,seq.extra_info) for seq in seqs if seq.is_finished]
            num_tokens = sum(len(seq) for seq in seqs) if is_prefill else -len(seqs)
            return outputs, num_tokens

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
        cnt = 1
        while True:
            while not self.is_finished():
                output, num_tokens = self.step()
                # if scheduling_type in [SchedulingType.IMAGE_PROCESSING,SchedulingType.VISION_ENCODING]:
                #     print("not implement",scheduling_type)
                for seq_id, token_ids,time_usage,extra_info in output:
                    req = self.request_dict[seq_id]
                    req.output_token_ids = token_ids
                    req.finished_event.set()
                    req.time_usage = time_usage
                    req.prefix_cached = extra_info.get('prefix_cached',-1)

            await asyncio.sleep(0.001)
            cnt+=1
            
    async def process_request(self,req:Request):
        prompt = req.prompt
        temperature = req.temperature
        max_tokens = req.max_tokens
        finished_event = req.finished_event
        if req.token_ids is None:
            token_ids = self.tokenizer.encode(prompt)
            if len(token_ids)>10000:
                print(f'large token ids : {len(token_ids)}')
            req.token_ids = token_ids
        else:
            token_ids = req.token_ids
            prompt = self.tokenizer.decode(token_ids)
        sampling_params = SamplingParams(temperature,max_tokens,req.ignore_eos)
        request_id = req.request_id
        seq = Sequence(token_ids,sampling_params,seq_id = request_id)
        self.request_dict[request_id] = req
        self.scheduler.add(seq)
        self.seqs_in_scheduler[seq.seq_id] = seq
        # print(f'add success!,seq_id:{seq.seq_id}')
        await finished_event.wait()
        req.output = self.tokenizer.decode(req.output_token_ids)
        self.request_dict[request_id]=None
        #TODO
        
