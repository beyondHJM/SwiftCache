"""Exercise decode block rollover, full reclamation, and queued-master recovery."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import time
from transformers import AutoTokenizer
from validate_serving import call, wait_ready


def wait_shrunk(port=18800):
    deadline=time.monotonic()+20
    while time.monotonic()<deadline:
        w=call(port+1,'/cache');m=call(port,'/cache')
        if w['reserved']==1 and not w['pending'] and not m['pending']:
            return
        time.sleep(.1)
    raise AssertionError(('shrink failed',w,m))


def main():
    port=18800;wait_ready(port)
    tokenizer=AutoTokenizer.from_pretrained('/root/llm-resource/Models/Qwen3-8B')
    records=[]
    for rank in [0,1]:
        wait_shrunk(port)
        ids=(tokenizer.encode('The days of the week are Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday. ')*20)[:255]
        case=dict(name='decode_block_rollover',token_ids=ids,max_tokens=24)
        result=call(port+rank,'/generate',case)
        replay=call(port+rank,'/generate',case)
        assert result['token_ids']==replay['token_ids']
        assert len(ids)+len(result['token_ids'])>256
        records.append(dict(case=case,rank=rank,result=result))
        print('ROLLOVER',rank,result['output'],flush=True)
    wait_shrunk(port)
    # Fit a complete request into all 32 physical blocks and retain output room.
    ids=tokenizer.apply_chat_template([dict(role='user',content=
        'Remember code ALPHA-42.\n'+'All stored keys and values must remain correct.\n'*1000+
        'What is the code? Reply only with ALPHA-42.')],tokenize=True,
        add_generation_prompt=True,enable_thinking=False)[-7950:]
    case=dict(name='full_worker_capacity',token_ids=ids,max_tokens=64,ignore_eos=True)
    baseline=json.loads(Path('artifacts/fixed-results.json').read_text())[0]
    with ThreadPoolExecutor(max_workers=2) as pool:
        worker=pool.submit(call,port+1,'/generate',case)
        deadline=time.monotonic()+20
        while time.monotonic()<deadline:
            state=call(port+1,'/cache')
            if state['reserved']==32 and not state['pending']: break
            if worker.done(): worker.result();raise AssertionError('Full worker completed before probe')
            time.sleep(.01)
        else: raise AssertionError(('Full capacity was not reached',state))
        m=call(port,'/cache')
        assert m['free']==0 and m['worker_reserved']['slave1']==32
        start=time.monotonic()
        master=pool.submit(call,port,'/generate',baseline['case'])
        wr=worker.result(timeout=120)
        mr=master.result(timeout=120)
        elapsed=time.monotonic()-start
    assert mr['token_ids']==baseline['cold']['token_ids']
    records.append(dict(case=case,rank=1,result=wr))
    wait_shrunk(port)
    m=call(port,'/cache');w=call(port+1,'/cache')
    assert m['used']==w['used']==0
    assert m['free']+w['free']==32
    assert not m['pending'] and not w['pending'] and not m['draining']
    Path('artifacts/boundary-results.json').write_text(json.dumps(dict(
        records=records,queued_master_seconds=elapsed,master_state=m,worker_state=w),ensure_ascii=False,indent=2))
    print('FULL_CAPACITY_AND_MASTER_RECOVERY',elapsed,wr['output'],flush=True)


if __name__=='__main__': main()
