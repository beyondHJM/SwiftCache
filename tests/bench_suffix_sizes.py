"""Small diagnostic: same ~100K cached history, 1/256/4096 actual new tokens."""

import argparse
import json
from pathlib import Path
import statistics
import time
import httpx
from transformers import AutoTokenizer


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--placement", choices=["remote", "local"], required=True)
    args = p.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("/root/llm-resource/Models/Qwen3-8B")
    head = tokenizer.encode(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nRead this reference document:\n"
    )
    paragraph = tokenizer.encode(
        "This paragraph describes a cache service. Requests reuse historical keys and values. A scheduler chooses the next batch while preserving all data.\n"
    )
    prefix = (head + paragraph * 10000)[:101376]
    tail = tokenizer.encode(
        " Please reply with OK.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    client = httpx.Client(
        base_url="http://127.0.0.1:18001", timeout=900, trust_env=False
    )
    seed = client.post(
        "/generate", json=dict(token_ids=prefix, max_tokens=1, temperature=0)
    )
    seed.raise_for_status()
    results = []
    counter = 90000
    for size in [1, 256, 4096]:
        records = []
        for i in range(15):
            suffix = ([1000] * (max(size - len(tail), 0)) + tail)[-size:]
            counter += 1
            suffix[0] = counter
            started = time.perf_counter()
            response = client.post(
                "/generate",
                json=dict(token_ids=prefix + suffix, max_tokens=1, temperature=0),
            )
            response.raise_for_status()
            result = response.json()
            assert result["cached_tokens"] == len(prefix)
            metric = result["metrics"]["batches"][0]
            assert metric["input_query_tokens"] == size and metric["logits_finite"]
            if i >= 5:
                records.append(
                    dict(
                        client_ms=(time.perf_counter() - started) * 1000, metric=metric
                    )
                )
        summary = {
            key: statistics.median(r["metric"][key] for r in records)
            for key in [
                "schedule_wall_ms",
                "model_gpu_span_ms",
                "kv_load_span_sum_ms",
                "kv_store_span_sum_ms",
            ]
        }
        summary["client_ms"] = statistics.median(r["client_ms"] for r in records)
        results.append(
            dict(
                cached_tokens=len(prefix),
                new_tokens=size,
                samples=10,
                summary=summary,
                raw=records,
            )
        )
        print("SUFFIX", args.placement, size, json.dumps(summary), flush=True)
    args.output.write_text(
        json.dumps(dict(placement=args.placement, cases=results), indent=2)
    )


if __name__ == "__main__":
    main()
