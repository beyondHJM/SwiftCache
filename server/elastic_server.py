"""Two-card SwiftCache launcher with explicit, reproducible cache limits."""
import argparse
import multiprocessing as mp
import os
from pathlib import Path
import signal
import tempfile
import time


def serve(rank, args, events, ready):
    import torch
    torch.cuda.set_device(rank)
    from swiftcache import LLM
    from swiftcache.engine.request import Request
    from fastapi import FastAPI, HTTPException
    import asyncio
    import atexit
    import uvicorn

    role='master' if rank==0 else 'slave'
    llm=LLM(args.model, rank=rank, role=role, dist_port=args.dist_port,
            tp_group=[rank],master_list=[0],slave_list=[1],
            slave_event=events if rank==0 else events[1],
            slave_ready_event=ready if rank==0 else ready[1],
            enforce_eager=True,tensor_parallel_size=1,
            max_model_len=args.max_model_len,max_num_batched_tokens=args.max_model_len*2,
            max_num_seqs=2,cache_block_limit=args.cache_blocks,
            elastic_enabled=not args.fixed,elastic_initial_blocks=args.initial_blocks,
            elastic_window_seconds=args.window,elastic_check_seconds=args.check_interval,
            gpu_memory_utilization=.65)
    # The parent owns process lifetime and the run-specific IPC resources.
    atexit.unregister(llm.exit)
    app=FastAPI()
    task=None

    @app.on_event('startup')
    async def startup():
        nonlocal task
        task=asyncio.create_task(llm._main_event_loop())

    @app.on_event('shutdown')
    async def shutdown():
        if task:
            task.cancel()

    @app.get('/health')
    async def health():
        if task.done():
            task.result()
        return dict(status='ready',role=role,model=args.model,rank=rank)

    @app.get('/cache')
    async def cache():
        if task.done():
            task.result()
        state=llm.scheduler.block_manager.snapshot()
        state['request_records']=len(llm.request_dict)
        state['scheduled_requests']=len(llm.seqs_in_scheduler)
        return state

    @app.post('/generate')
    async def generate(data:dict):
        req=Request(data.get('prompt'),max_tokens=int(data.get('max_tokens',16)))
        req.temperature=float(data.get('temperature',0))
        req.ignore_eos=bool(data.get('ignore_eos',False))
        req.token_ids=data.get('token_ids')
        if req.max_tokens<=0 or req.temperature<0:
            raise HTTPException(400,'Invalid sampling parameters')
        try:
            await llm.process_request(req)
        except ValueError as exc:
            raise HTTPException(400,str(exc)) from exc
        return dict(output=req.output,token_ids=req.output_token_ids,
                    prompt_tokens=len(req.token_ids),cached_tokens=req.prefix_cached,
                    time_usage=req.time_usage)

    uvicorn.run(app,host='127.0.0.1',port=args.port+rank,log_level='info')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model',required=True)
    parser.add_argument('--port',type=int,default=18800)
    parser.add_argument('--dist-port',type=int,default=18334)
    parser.add_argument('--cache-blocks',type=int,default=32)
    parser.add_argument('--initial-blocks',type=int,default=1)
    parser.add_argument('--max-model-len',type=int,default=8192)
    parser.add_argument('--window',type=float,default=60)
    parser.add_argument('--check-interval',type=float,default=1)
    parser.add_argument('--fixed',action='store_true')
    args=parser.parse_args()
    if not 1<=args.initial_blocks<=args.cache_blocks or args.window<=0 or args.check_interval<=0:
        parser.error('Invalid cache capacity or timing configuration')
    if os.environ.get('USE_CSCHEDULER','').lower() in ('1','true','yes'):
        parser.error('Elastic handoff currently uses the Python scheduler; unset USE_CSCHEDULER')
    runtime=Path(tempfile.mkdtemp(prefix='swiftcache-'))
    os.environ['SWIFTCACHE_RUNTIME_DIR']=str(runtime)
    os.environ['SWIFTCACHE_SHM_NAME']=runtime.name.replace('-','_')
    os.environ['SWIFTCACHE_CONTROL_ADDRESS']='ipc://'+str(runtime/'control.sock')
    os.environ['TOKENIZERS_PARALLELISM']='false'
    ctx=mp.get_context('spawn')
    events={1:ctx.Event()}; ready={1:ctx.Event()}
    ps=[ctx.Process(target=serve,args=(rank,args,events,ready),name=f'swiftcache-{rank}') for rank in [1,0]]
    def stop(signum,frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM,stop)
    try:
        for p in ps: p.start()
        print(f'SWIFTCACHE_PARENT pid={os.getpid()} runtime={runtime}',flush=True)
        while all(p.is_alive() for p in ps):
            time.sleep(.5)
        raise RuntimeError('A serving process exited unexpectedly')
    except KeyboardInterrupt:
        pass
    finally:
        for p in ps:
            if p.is_alive(): p.terminate()
        for p in ps:
            p.join(timeout=15)
            if p.is_alive(): p.kill(); p.join()
        from multiprocessing.shared_memory import SharedMemory
        try:
            shm=SharedMemory(name=os.environ['SWIFTCACHE_SHM_NAME'])
            shm.close(); shm.unlink()
        except FileNotFoundError:
            pass
        for path in runtime.iterdir(): path.unlink()
        runtime.rmdir()


if __name__=='__main__':
    main()
