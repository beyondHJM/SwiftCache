import asyncio
import argparse
import sys
# sys.path.insert(0, '/home/admin/workspace/aop_lab/app_source/hujianmin/SwiftCache')
from swiftcache import LLM
from swiftcache.engine.request import Request
from server.instance_config import InstanceConfig
from server.instance_config import ServerConfig
# from server.instance_config import EngineConfig
import torch.multiprocessing as mp
from multiprocessing.synchronize import Event
from multiprocessing.shared_memory import SharedMemory
import uvicorn
import traceback
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
from fastapi import FastAPI
import fastapi
# 配置
host = "0.0.0.0"
port = 8000
TIMEOUT_KEEP_ALIVE = 60
llm = None
app = FastAPI()
engine=0

@app.get("/")
async def hello_world():
    return {"message": "Hello World!"}

@app.post("/generate")
async def generate(req: fastapi.Request)-> fastapi.Response:
    global llm
    req_dict = await req.json()
    prompt = req_dict.get('prompt')
    token_ids = req_dict.get('token_ids',None)
    max_tokens = req_dict.get('max_tokens')
    raw_req = Request(prompt,max_tokens = max_tokens)
    raw_req.token_ids = token_ids
    await llm.process_request(raw_req)
    time_usage = raw_req.time_usage
    prefill_time = time_usage[0]
    time_usage = time_usage[1:]
    if len(time_usage)>0:
        avg_decoding_time = sum(time_usage)/len(time_usage)
    else:
        avg_decoding_time = -1
    if raw_req.prompt is None:
        raw_req.prompt = llm.tokenizer.decode(raw_req.token_ids[-100:])
    result = {'prompt':raw_req.prompt,'prompt_len':len(raw_req.token_ids),'output':raw_req.output,'output_len':len(raw_req.output_token_ids),'is_hit':raw_req.prefix_cached>0,'prefix_cached_rate':f'{raw_req.prefix_cached}/{len(raw_req.token_ids)}','prefill_time':prefill_time,'avg_decoding_time':avg_decoding_time}
    return fastapi.responses.JSONResponse(content=result)

async def main_coroutine(host = 'localhost', port = 8000):
    
    # 创建 Uvicorn 配置
    host = host
    port = port
    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        timeout_keep_alive=TIMEOUT_KEEP_ALIVE
    )
    uvicorn_server = uvicorn.Server(uvicorn_config)

    engine_task = asyncio.create_task(llm._main_event_loop())
    uvicorn_task = asyncio.create_task(uvicorn_server.serve())
    print('Server Starts!')

# 这段不写，不打印错误信息很烦
    try:
        await engine_task
    except:  # pylint: disable=broad-except

        traceback.print_exc()
        uvicorn_task.cancel()
        os._exit(1) # Kill myself, or it will print tons of errors. Don't know why.
    await uvicorn_task 
 

class ServerStarter:
    def __init__(self,model = "Qwen3-0.6B",host = 'localhost', port = 8000, engine_config:dict = {}):
        global llm
        print(f'rank:{engine_config.get("rank")},engine_config:{engine_config}')
        # if engine_config['rank']!=0:
        #         sys.stdout = open(os.devnull, 'w')
        #         sys.stderr = open(os.devnull, 'w')
        self.models = {
            'Llama3':"/home/admin/workspace/aop_lab/app_data/Llama-3-8B-Instruct",
            'Qwen3-0.6B': "/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-0.6B/snapshots/c1899de289a04d12100db370d81485cdf75e47ca",
            'Qwen3-8B': "/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-8B/snapshots/b968826d9c46dd6066d109eabc6255188de91218",
            'Qwen3-14B': "/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-14B/snapshots/40c069824f4251a91eefaf281ebe4c544efd3e18",
            'Qwen3-32B': "/home/admin/workspace/aop_lab/app_data/.cache/models--Qwen--Qwen3-32B/snapshots/ba1f828c09458ab0ae83d42eaacc2cf8720c7957",
            'DeepSeek': "/home/admin/workspace/aop_lab/app_data/.cache/models--deepseek-ai--deepseek-vl2-tiny/snapshots/66c54660eae7e90c9ba259bfdf92d07d6e3ce8aa",
            'Llama-2-7b':"/home/admin/workspace/aop_lab/app_data/.cache/models--meta-llama--Llama-2-7b-chat-hf/snapshots/c1b0db933684edbfe29a06fa47eb19cc48025e93",
            'LWM': "/home/admin/workspace/aop_lab/app_data/.cache/models--LargeWorldModel--LWM-Text-Chat-1M/snapshots/0598c443b02aeb1a1f9f6788e9af85ea762a452d"
        }

        if self.models.get(model) is None:
            self.model = model
        else:
            self.model = self.models.get(model) 
        
        # llm = LLM(self.model , enforce_eager=True, tensor_parallel_size=1, dist_port = dist_port, master_list = master_list,slave_list = slave_list,slave_event = slave_event, slave_ready_event = slave_ready_event, role = role,rank = rank,tp_group = tp_group)
        llm = LLM(self.model ,**engine_config)
        asyncio.run(main_coroutine(host,port))

class MultiServerStarter:
    def __init__(self, instance_configs:list[InstanceConfig], master_list:list[int], slave_list:list[int]):
        events = {}
        ready_events = {}
        ps = {}
        ctx = mp.get_context("spawn")
        master_configs = [cfg for cfg in instance_configs if cfg.engine_config.get('role') == "master"]
        slave_configs  = [cfg for cfg in instance_configs if cfg.engine_config.get('role') == "slave"]

        for config in slave_configs:
            #需要把event加到启动参数中
            event = ctx.Event()
            ready_event = ctx.Event()
            engine_config = config.engine_config
            engine_config['master_list'] = master_list
            engine_config['slave_list'] = slave_list
            engine_config['slave_event'] = event
            engine_config['slave_ready_event'] = ready_event
            # print(engine_config)
            # process = ctx.Process(target=ServerStarter, args=(config.engine_config.get('model'), config.server_config.host, config.server_config.port, config.engine_config.get('dist_port'), master_list, slave_list, event, ready_event, config.engine_config.get('role'), config.engine_config.get('rank'), config.engine_config.get('tp_group')))
            model = engine_config.pop('model')
            process = ctx.Process(target=ServerStarter, args=(model, config.server_config.host, config.server_config.port,engine_config))
            process.start()
            slave_id = config.engine_config.get('rank')
            ps[slave_id] = process
            events[slave_id] = event
            ready_events[slave_id] = ready_event 

        assert  len(master_configs) == 1
        config = master_configs[0]
        engine_config = config.engine_config
        model = engine_config.pop('model')
        # print('-------',engine_config)
        engine_config['master_list'] = master_list
        engine_config['slave_list'] = slave_list
        engine_config['slave_event'] = events
        engine_config['slave_ready_event'] = ready_events
        # ServerStarter(config.engine_config.get('model'), config.server_config.host, config.server_config.port, config.engine_config.get('dist_port'), master_list, slave_list, events, ready_events, config.engine_config.get('role'), config.engine_config.get('rank'), config.engine_config.get('tp_group'))
        ServerStarter(model, config.server_config.host, config.server_config.port, engine_config)
        



if __name__ == "__main__":
  
    server_config = ServerConfig(host="localhost",port=8000)
    engine_config ={'model':'Qwen3-8B','rank':0,'role':'master','dist_port':2334,'tp_group':[0],'gpu_memory_utilization':0.9,'max_num_batched_tokens':40960,'max_model_len':40960}
    instance1_config = InstanceConfig(server_config, engine_config)
    server_config = ServerConfig(host="localhost",port=8001)
    engine_config ={'model':'Qwen3-8B','rank':1,'role':'slave','dist_port':2334,'tp_group':[1],'gpu_memory_utilization':0.9,'max_num_batched_tokens':40960,'max_model_len':40960}
    instance2_config = InstanceConfig(server_config, engine_config)
    server_config = ServerConfig(host="localhost",port=8002)
    engine_config ={'model':'Qwen3-14B','rank':2,'role':'slave','dist_port':2334,'tp_group':[2],'gpu_memory_utilization':0.85}
    instance3_config = InstanceConfig(server_config, engine_config)
    instance_configs = [instance1_config, instance2_config,instance3_config]
    instance_configs = [instance1_config, instance2_config]
    master_list, slave_list = [], []
    for config in instance_configs:
        rank = config.engine_config.get('rank')
        if config.engine_config.get('role') == 'master':
            master_list.append(rank)
        else:
            slave_list.append(rank)
    MultiServerStarter(instance_configs, master_list, slave_list)