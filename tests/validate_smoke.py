"""Final serving check after request bookkeeping cleanup."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
from validate_serving import call, wait_ready


def main():
    wait_ready(18800)
    baseline=json.loads(Path('artifacts/fixed-results.json').read_text())
    requests=baseline+[record for record in baseline if record['case']['name']=='long_prefix']*2
    # Concurrent requests exercise both model instances and their live queues.
    with ThreadPoolExecutor(max_workers=4) as pool:
        jobs=[(record,pool.submit(call,18800+record['rank'],'/generate',record['case']))
              for record in requests]
        for record,future in jobs:
            result=future.result(timeout=120)
            assert result['token_ids']==record['cold']['token_ids']
    states=[call(18800+rank,'/cache') for rank in [0,1]]
    for state in states:
        assert state['used']==state['request_records']==state['scheduled_requests']==0
    Path('artifacts/final-smoke-results.json').write_text(json.dumps(states,indent=2))
    print('FINAL_SMOKE: 12 requests including concurrent shared prefixes matched baseline; no retained request records or live KV')


if __name__=='__main__': main()
