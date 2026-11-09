"""Seeded Poisson arrivals via the public OpenAI Completions protocol.

This is an exploratory small-sample queue/load probe, not a p99/SLO estimate.
"""

import argparse
import asyncio
import json
from pathlib import Path
import random
import statistics
import time
import httpx
from transformers import AutoTokenizer


async def run(args):
    tok = AutoTokenizer.from_pretrained("/root/llm-resource/Models/Qwen3-8B")
    head = tok.encode(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nRead this reference document:\n"
    )
    paragraph = tok.encode(
        "This paragraph describes a cache service. Requests reuse historical keys and values. A scheduler chooses the next batch while preserving all data.\n"
    )
    prefix = (head + paragraph * 10000)[:101376]
    tail = tok.encode(
        "\nEnd of document. Please respond only with OK.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    suffix = ([1000] * (1024 - len(tail))) + tail
    public = httpx.AsyncClient(
        base_url="http://127.0.0.1:18000", timeout=900, trust_env=False
    )
    backend = httpx.AsyncClient(
        base_url="http://127.0.0.1:18001", timeout=900, trust_env=False
    )
    for count in [25088, 50176, 75264, 101376]:
        r = await backend.post(
            "/generate",
            json=dict(token_ids=prefix[:count], max_tokens=1, temperature=0),
        )
        r.raise_for_status()
    outputs = []
    for rate in args.rates:
        rng = random.Random(20261108)
        offsets = [0.0]
        for _ in range(args.requests - 1):
            offsets.append(offsets[-1] + rng.expovariate(rate))
        beginning = time.perf_counter()

        async def issue(index, offset):
            await asyncio.sleep(max(0, beginning + offset - time.perf_counter()))
            novel = suffix.copy()
            novel[0] = 120000 + index + int(rate * 100)
            submitted = time.perf_counter()
            response = await public.post(
                "/v1/completions",
                json=dict(
                    model="qwen3-8b-master",
                    prompt=prefix + novel,
                    max_tokens=1,
                    temperature=0,
                ),
            )
            response.raise_for_status()
            body = response.json()
            finished = time.perf_counter()
            assert body["usage"]["prompt_tokens_details"]["cached_tokens"] == len(
                prefix
            )
            metric = await backend.get("/request-metrics/" + body["id"])
            metric.raise_for_status()
            return dict(
                arrival_offset_s=offset,
                submission_offset_s=submitted - beginning,
                client_latency_ms=(finished - submitted) * 1000,
                finish_offset_s=finished - beginning,
                metrics=metric.json(),
            )

        records = await asyncio.gather(*(issue(i, t) for i, t in enumerate(offsets)))
        elapsed = max(r["finish_offset_s"] for r in records)
        latency = [r["client_latency_ms"] for r in records]
        output = dict(
            configured_poisson_rate=rate,
            requests=args.requests,
            observed_arrival_rate=(args.requests - 1) / offsets[-1],
            completed_rps=args.requests / elapsed,
            client_latency_p50_ms=statistics.median(latency),
            client_latency_max_ms=max(latency),
            observed_batch_sizes=[
                r["metrics"]["batches"][0]["batch_size"] for r in records
            ],
            raw=records,
        )
        outputs.append(output)
        print(
            "ARRIVALS",
            args.placement,
            json.dumps({k: v for k, v in output.items() if k != "raw"}),
            flush=True,
        )
        args.output.write_text(
            json.dumps(
                dict(
                    placement=args.placement,
                    small_sample_exploratory=True,
                    cases=outputs,
                ),
                indent=2,
            )
        )
    await public.aclose()
    await backend.aclose()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--placement", choices=["local", "remote"], required=True)
    p.add_argument("--rates", type=float, nargs="+", required=True)
    p.add_argument("--requests", type=int, default=12)
    args = p.parse_args()
    asyncio.run(run(args))


if __name__ == "__main__":
    main()
