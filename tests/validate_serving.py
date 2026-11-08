"""Real Qwen3-8B checks: run against fixed, then elastic servers, then HF."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
import urllib.request
from transformers import AutoTokenizer


def call(port,path,data=None,timeout=120):
    req=urllib.request.Request(f'http://127.0.0.1:{port}{path}',
        data=None if data is None else json.dumps(data).encode(),
        headers={'Content-Type':'application/json'})
    with urllib.request.urlopen(req,timeout=timeout) as response:
        return json.load(response)


def wait_ready(port):
    deadline=time.monotonic()+180
    while time.monotonic()<deadline:
        try:
            for p in [port,port+1]: assert call(p,'/health')['status']=='ready'
            return
        except Exception:
            time.sleep(1)
    raise TimeoutError('Serving pair did not become ready')


def cases(tokenizer):
    def chat(text):
        return tokenizer.apply_chat_template([dict(role='user',content=text)],
            tokenize=True,add_generation_prompt=True,enable_thinking=False)
    return [
        dict(name='arithmetic',token_ids=chat('What is 2 + 2? Answer with only the number.'),max_tokens=12),
        dict(name='translation',token_ids=chat('Translate into Chinese: The cache is working correctly.'),max_tokens=24),
        dict(name='long_prefix',token_ids=chat('The access code is SILVER-731.\n'+
            'This project manages cache memory safely and preserves every token.\n'*60+
            'What is the access code? Reply with only the code.'),max_tokens=20),
        dict(name='exact_block',token_ids=(tokenizer.encode('The days of the week are Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday. ')*30)[:512],max_tokens=12),
    ]


def run_fixed(args):
    wait_ready(args.port)
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    records=[]
    for case in cases(tokenizer):
        for rank in [0,1]:
            cold=call(args.port+rank,'/generate',case)
            warm=call(args.port+rank,'/generate',case)
            assert cold['token_ids']==warm['token_ids'],(case['name'],rank,cold,warm)
            if len(case['token_ids'])>256: assert warm['cached_tokens']>0
            records.append(dict(case=case,rank=rank,cold=cold,warm=warm))
            print('FIXED',case['name'],rank,len(case['token_ids']),repr(cold['output']),warm['cached_tokens'],flush=True)
    for rank in [0,1]:
        state=call(args.port+rank,'/cache');assert state['used']==0
    (args.artifacts/'fixed-results.json').write_text(json.dumps(records,ensure_ascii=False,indent=2))


def run_elastic(args):
    wait_ready(args.port)
    records=json.loads((args.artifacts/'fixed-results.json').read_text())
    results=[]
    for record in records:
        result=call(args.port+record['rank'],'/generate',record['case'])
        assert result['token_ids']==record['cold']['token_ids'],(record['case']['name'],record['rank'],result,record['cold'])
        results.append(dict(case=record['case']['name'],rank=record['rank'],result=result))
    tokenizer=AutoTokenizer.from_pretrained(args.model)
    def long_case(repetitions,max_tokens=24):
        ids=tokenizer.apply_chat_template([dict(role='user',content=
            'Remember code ALPHA-42.\n'+'All stored keys and values must remain correct.\n'*repetitions+
            'What is the code? Reply only with ALPHA-42.')],tokenize=True,
            add_generation_prompt=True,enable_thinking=False)
        return dict(token_ids=ids,max_tokens=max_tokens)
    history=[]
    for cycle in range(args.cycles):
        # Give old demand time to leave the window; verify both ends agree.
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            w=call(args.port+1,'/cache');m=call(args.port,'/cache')
            if w['reserved']==1 and not w['pending'] and not m['pending']: break
            time.sleep(.25)
        else: raise AssertionError(('worker did not shrink',w,m))
        master_case=long_case(90,64)
        master_case['ignore_eos']=True
        worker_case=long_case(160+cycle*10,24)
        with ThreadPoolExecutor(max_workers=2) as pool:
            future=pool.submit(call,args.port,'/generate',master_case)
            # Submit while master has live remote KV, rather than after completion.
            deadline=time.monotonic()+10
            while time.monotonic()<deadline:
                state=call(args.port,'/cache')
                if state['used']>0: break
                if future.done(): raise AssertionError('Master finished before concurrency probe')
                time.sleep(.01)
            other=pool.submit(call,args.port+1,'/generate',worker_case)
            mr=future.result(timeout=120);wr=other.result(timeout=120)
        # Replay both prompts after cache handoff; output must remain unchanged.
        mr2=call(args.port,'/generate',master_case)
        wr2=call(args.port+1,'/generate',worker_case)
        assert mr['token_ids']==mr2['token_ids'],('master replay',cycle,mr,mr2)
        assert wr['token_ids']==wr2['token_ids'],('worker replay',cycle,wr,wr2)
        m=call(args.port,'/cache');w=call(args.port+1,'/cache')
        assert m['used']==w['used']==0
        assert m['worker_reserved']['slave1']==w['reserved']
        assert m['free']+w['free']==args.cache_blocks
        assert not m['pending'] and not w['pending'] and not m['draining']
        history.append(dict(cycle=cycle,master_input=master_case,worker_input=worker_case,
                            master=mr,worker=wr,master_state=m,worker_state=w))
        print('CYCLE',cycle,'worker_reserved',w['reserved'],'outputs',repr(mr['output']),repr(wr['output']),flush=True)
    # Confirm active reclaim really occurred and that both shrink and grow ran.
    m=call(args.port,'/cache');w=call(args.port+1,'/cache')
    assert any(e['event']=='reclaim_pending' and e['active_blocks']>0 for e in m['events'])
    applied=[e for e in w['events'] if e['event']=='resize_applied']
    assert any(e['reserved']>e['old'] for e in applied)
    assert any(e['reserved']<e['old'] for e in applied)
    (args.artifacts/'elastic-results.json').write_text(json.dumps(dict(results=results,cycles=history),ensure_ascii=False,indent=2))


def run_reference(args):
    import torch
    from transformers import AutoModelForCausalLM
    model=AutoModelForCausalLM.from_pretrained(args.model,torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',device_map='cuda:0').eval()
    records=json.loads((args.artifacts/'fixed-results.json').read_text())
    extra=args.artifacts/'boundary-results.json'
    if extra.exists():
        for record in json.loads(extra.read_text())['records']:
            records.append(dict(case=record['case'],rank=record['rank'],cold=record['result']))
    comparisons=[]
    for record in records:
        case=record['case']
        ids=torch.tensor([case['token_ids']],device='cuda:0')
        with torch.inference_mode():
            output=model.generate(ids,attention_mask=torch.ones_like(ids),do_sample=False,
                max_new_tokens=case['max_tokens'],pad_token_id=model.config.eos_token_id,
                eos_token_id=None if case.get('ignore_eos') else model.generation_config.eos_token_id)
        tokens=output[0,ids.shape[1]:].tolist()
        comparison=dict(case=case['name'],rank=record['rank'],reference=tokens,
                        swiftcache=record['cold']['token_ids'],match=tokens==record['cold']['token_ids'])
        comparisons.append(comparison)
        print('REFERENCE',comparison,flush=True)
    (args.artifacts/'reference-results.json').write_text(json.dumps(comparisons,indent=2))
    assert all(c['match'] for c in comparisons)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('mode',choices=['fixed','elastic','reference'])
    parser.add_argument('--model',default='/root/llm-resource/Models/Qwen3-8B')
    parser.add_argument('--port',type=int,default=18800)
    parser.add_argument('--artifacts',type=Path,default=Path('artifacts'))
    parser.add_argument('--cache-blocks',type=int,default=32)
    parser.add_argument('--cycles',type=int,default=6)
    args=parser.parse_args();args.artifacts.mkdir(exist_ok=True)
    globals()['run_'+args.mode](args)


if __name__=='__main__': main()
