"""Real two-GPU Qwen3-8B: prefix warmup + exact 1024-token suffix.

Every measured request verifies the observed cache hit and actual query counts.
The request path is legacy /generate only to expose internal timing, while the
public OpenAI API is tested separately with the official SDK.
"""

import argparse
import asyncio
import json
from pathlib import Path
import statistics
import time
import httpx
from transformers import AutoTokenizer


def stats(values):
    values = sorted(values)
    return dict(
        mean=statistics.mean(values),
        p50=statistics.median(values),
        p95=values[min(len(values) - 1, int(0.95 * len(values)))],
        max=max(values),
    )


async def main_async(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    client = httpx.AsyncClient(base_url=args.url, timeout=900, trust_env=False)
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        try:
            r = await client.get("/health")
            if r.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        await asyncio.sleep(1)
    else:
        raise TimeoutError("Model did not start")
    state = (await client.get("/cache")).json()
    assert state["ignore_model_max_length"] and state["context_limit"] >= 102401
    print("MODEL_STATE", json.dumps(state), flush=True)
    head = tokenizer.encode(
        "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nRead this reference document:\n"
    )
    paragraph = tokenizer.encode(
        "This paragraph describes a cache service. Requests reuse historical keys and values. A scheduler chooses the next batch while preserving all data.\n"
    )
    tail = tokenizer.encode(
        "\nEnd of document. Please respond only with OK.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
    )
    filler = tokenizer.encode(" The service preserves each token and its position. ")
    suffix = (filler * 200)[: 1024 - len(tail)] + tail
    assert len(suffix) == 1024
    counter = 0
    results = []

    async def issue(tokens, expected_cached=None):
        start = time.perf_counter()
        response = await client.post(
            "/generate", json=dict(token_ids=tokens, max_tokens=1, temperature=0)
        )
        response.raise_for_status()
        result = response.json()
        result["client_wall_ms"] = (time.perf_counter() - start) * 1000
        if expected_cached is not None:
            assert result["cached_tokens"] == expected_cached, (
                result["cached_tokens"],
                expected_cached,
            )
            batch = result["metrics"]["batches"][0]
            assert batch["input_query_tokens"] == 1024 * batch["batch_size"], batch
            assert batch.get("logits_finite", True)
            if args.placement == "local":
                assert batch["remote_loaded_blocks"] == 0, batch
            else:
                assert batch["remote_loaded_blocks"] >= expected_cached // 256, batch
        return result

    async def request(prefix):
        nonlocal counter
        counter += 1
        unique = suffix.copy()
        unique[0] = 1000 + counter
        return await issue(prefix + unique, len(prefix))

    for total in args.lengths:
        p = total - 1024
        prefix = (head + paragraph * ((p - len(head)) // len(paragraph) + 1))[:p]
        seed = await issue(prefix)
        print(
            "SEEDED",
            total,
            "cached",
            seed["cached_tokens"],
            "wall_ms",
            seed["client_wall_ms"],
            flush=True,
        )
        rounds = []
        for iteration in range(args.rounds):
            for _ in range(args.warmups):
                await request(prefix)
            samples = [await request(prefix) for _ in range(args.samples)]
            keys = [
                "schedule_wall_ms",
                "schedule_cpu_ms",
                "prefix_match_allocate_ms",
                "control_wall_ms",
                "prepare_wall_ms",
                "model_gpu_span_ms",
                "model_and_sample_wall_ms",
                "kv_load_span_sum_ms",
                "kv_store_span_sum_ms",
                "postprocess_wall_ms",
            ]
            values = {k: [s["metrics"]["batches"][0][k] for s in samples] for k in keys}
            values["client_wall_ms"] = [s["client_wall_ms"] for s in samples]
            values["enqueue_to_first_ms"] = [
                (s["metrics"]["first_token_ns"] - s["metrics"]["enqueue_ns"]) / 1e6
                for s in samples
            ]
            summarized = {k: stats(v) for k, v in values.items()}
            rounds.append(dict(summary=summarized, samples=samples))
            print("ROUND", total, iteration, json.dumps(summarized), flush=True)
            record = dict(
                total_tokens=total,
                cached_tokens=p,
                uncached_tokens=1024,
                seed=seed,
                rounds=rounds,
            )
            if results and results[-1]["total_tokens"] == total:
                results[-1] = record
            else:
                results.append(record)
            args.output.write_text(
                json.dumps(
                    dict(
                        placement=args.placement,
                        model=args.model,
                        length_override=True,
                        source="actual two-GPU inference",
                        cases=results,
                    ),
                    indent=2,
                )
            )
    # A shared prefix at 100K, with enough physical space for multiple 1K suffixes.
    concurrency = []
    for count in args.concurrency:
        for _ in range(2):
            await asyncio.gather(*(request(prefix) for _ in range(count)))
        start = time.perf_counter()
        samples = []
        for _ in range(5):
            samples.extend(
                await asyncio.gather(*(request(prefix) for _ in range(count)))
            )
        elapsed = time.perf_counter() - start
        batch_sizes = [s["metrics"]["batches"][0]["batch_size"] for s in samples]
        concurrency.append(
            dict(
                offered_concurrency=count,
                requests=len(samples),
                requests_per_second=len(samples) / elapsed,
                observed_batch_sizes=batch_sizes,
                latency_ms=stats([s["client_wall_ms"] for s in samples]),
            )
        )
        print("CONCURRENCY", json.dumps(concurrency[-1]), flush=True)
    await client.post("/profiling", json={"enabled": False})
    without = [await request(prefix) for _ in range(10)]
    await client.post("/profiling", json={"enabled": True})
    with_profile = [await request(prefix) for _ in range(10)]
    overhead = dict(
        disabled_wall_ms=stats([s["client_wall_ms"] for s in without]),
        enabled_wall_ms=stats([s["client_wall_ms"] for s in with_profile]),
    )
    final = (await client.get("/cache")).json()
    assert final["used"] == final["request_records"] == final["scheduled_requests"] == 0
    data = json.loads(args.output.read_text())
    data.update(concurrency=concurrency, profiling_overhead=overhead, final_state=final)
    args.output.write_text(json.dumps(data, indent=2))
    await client.aclose()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:18001")
    p.add_argument("--model", default="/root/llm-resource/Models/Qwen3-8B")
    p.add_argument("--placement", choices=["remote", "local"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--lengths", type=int, nargs="+", default=[51200, 76800, 102400])
    p.add_argument("--samples", type=int, default=30)
    p.add_argument("--rounds", type=int, default=3)
    p.add_argument("--warmups", type=int, default=10)
    p.add_argument("--concurrency", type=int, nargs="+", default=[1, 2, 4])
    args = p.parse_args()
    args.output.parent.mkdir(exist_ok=True, parents=True)
    asyncio.run(main_async(args))


if __name__ == "__main__":
    main()
