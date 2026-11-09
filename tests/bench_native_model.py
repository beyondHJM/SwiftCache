"""One free GPU, actual Qwen3-8B, local KV: Python/C++ engine comparison.

This intentionally needs no worker GPU. It is not a two-GPU elastic or HTTP
benchmark. Both backends execute the same model/runner and pretokenized inputs.
"""

import argparse
import asyncio
import atexit
from contextlib import suppress
import json
import os
from pathlib import Path
import tempfile
import time
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from bench_scheduler import summary


async def benchmark(args):
    from swiftcache import LLM
    from swiftcache.engine.request import Request

    llm = LLM(
        args.model,
        rank=0,
        role="master",
        tp_group=[0],
        master_list=[0],
        slave_list=[],
        slave_event={},
        slave_ready_event={},
        dist_port=args.dist_port,
        max_model_len=8192 if args.short_only else 131072,
        ignore_model_max_length=not args.short_only,
        max_num_batched_tokens=8192 if args.short_only else 131072,
        max_num_seqs=4,
        cache_block_limit=1152 if args.short_only else 14832,
        elastic_enabled=False,
        warmup_tokens=1024,
        profile_stages=True,
        scheduler_backend=args.backend,
    )
    atexit.unregister(llm.exit)
    task = asyncio.create_task(llm._main_event_loop())

    async def issue(ids, maximum=1):
        req = Request(None, temperature=0, max_tokens=maximum)
        req.token_ids = ids
        start = time.perf_counter_ns()
        await llm.process_request(req)
        elapsed = (time.perf_counter_ns() - start) / 1e6
        return dict(
            output=req.output,
            token_ids=req.output_token_ids,
            cached_tokens=req.prefix_cached,
            engine_request_wall_ms=elapsed,
            metrics=req.metrics,
        )

    try:
        tokenizer = llm.tokenizer
        checks = []
        for prompt in [
            "What is 2 + 2? Reply only with the number.",
            "Translate into Chinese: The cache is working correctly.",
            "Remember ALPHA-42. What is the code? Reply only with the code.",
        ]:
            ids = tokenizer.apply_chat_template(
                [dict(role="user", content=prompt)],
                tokenize=True,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            cold = await issue(ids, 32)
            warm = await issue(ids, 32)
            assert cold["token_ids"] == warm["token_ids"]
            checks.append(dict(prompt=prompt, cold=cold, warm=warm))
        assert checks[0]["cold"]["output"].strip() == "4"
        if args.short_only:
            root = Path("artifacts")
            references = json.loads((root / "fixed-results.json").read_text())
            references += [
                dict(case=row["case"], rank=row["rank"], cold=row["result"])
                for row in json.loads((root / "boundary-results.json").read_text())[
                    "records"
                ]
            ]
            comparisons = []
            for record in references:
                case = record["case"]
                req = Request(
                    None,
                    temperature=0,
                    max_tokens=case["max_tokens"],
                    ignore_eos=case.get("ignore_eos", False),
                )
                req.token_ids = case["token_ids"]
                await llm.process_request(req)
                assert req.output_token_ids == record["cold"]["token_ids"], (
                    case["name"],
                    args.backend,
                    req.output,
                )
                comparisons.append(
                    dict(
                        case=case["name"],
                        rank=record["rank"],
                        match=True,
                        token_ids=req.output_token_ids,
                    )
                )
            state = llm.scheduler.block_manager.snapshot()
            assert (
                state["used"] == 0
                and llm.scheduler.is_finished()
                and not llm.request_dict
            )
            args.output.write_text(
                json.dumps(
                    dict(
                        single_gpu=True,
                        short_only=True,
                        backend=args.backend,
                        semantic_checks=checks,
                        saved_reference_matches=comparisons,
                        final_state=state,
                    ),
                    ensure_ascii=False,
                    indent=2,
                )
            )
            print(
                "SHORT_MODEL_REFERENCE_MATCHES",
                args.backend,
                len(comparisons),
                flush=True,
            )
            return
        head = tokenizer.encode(
            "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nRead this reference document:\n"
        )
        paragraph = tokenizer.encode(
            "This paragraph describes a cache service. Requests reuse historical keys and values. A scheduler chooses the next batch while preserving all data.\n"
        )
        prefix = (head + paragraph * 10000)[:101376]
        tail = tokenizer.encode(
            "\nEnd of document. Please respond only with OK.<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
        )
        suffix = ([1000] * (1024 - len(tail))) + tail
        seeds = []
        for count in [25088, 50176, 75264, 101376]:
            seeds.append(await issue(prefix[:count]))
        counter = 0
        rounds = []
        for _ in range(3):
            measured = []
            for i in range(40):
                counter += 1
                unique = suffix.copy()
                unique[0] = 120000 + counter
                result = await issue(prefix + unique)
                metric = result["metrics"]["batches"][0]
                assert result["cached_tokens"] == 101376
                assert (
                    metric["input_query_tokens"] == 1024 and metric["batch_size"] == 1
                )
                assert metric["logits_finite"] and metric["remote_loaded_blocks"] == 0
                assert metric["scheduler_backend"] == args.backend
                if i >= 10:
                    measured.append(result)
            rounds.append(measured)
            print(
                "MODEL_ROUND",
                args.backend,
                len(rounds),
                summary([r["engine_request_wall_ms"] for r in measured]),
                flush=True,
            )
        state = llm.scheduler.block_manager.snapshot()
        assert (
            state["used"] == 0 and not llm.request_dict and llm.scheduler.is_finished()
        )
        all_samples = [r for rnd in rounds for r in rnd]
        data = dict(
            single_gpu=True,
            placement="local",
            http=False,
            model=args.model,
            backend=args.backend,
            total_tokens=102400,
            cached_tokens=101376,
            query_tokens=1024,
            rounds=rounds,
            seeds=seeds,
            semantic_checks=checks,
            final_state=state,
            summary={
                k: summary([r["metrics"]["batches"][0][k] for r in all_samples])
                for k in [
                    "schedule_wall_ms",
                    "model_gpu_span_ms",
                    "prepare_wall_ms",
                    "postprocess_wall_ms",
                ]
            },
        )
        data["summary"]["engine_request_wall_ms"] = summary(
            [r["engine_request_wall_ms"] for r in all_samples]
        )
        args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    finally:
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task
        llm.exit()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["python", "cpp"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--model", default="/root/llm-resource/Models/Qwen3-8B")
    p.add_argument("--dist-port", type=int, default=18440)
    p.add_argument("--short-only", action="store_true")
    args = p.parse_args()
    with tempfile.TemporaryDirectory(prefix="swiftcache-native-model-") as runtime:
        os.environ["SWIFTCACHE_RUNTIME_DIR"] = runtime
        os.environ["SWIFTCACHE_SHM_NAME"] = Path(runtime).name.replace("-", "_")
        os.environ["SWIFTCACHE_CONTROL_ADDRESS"] = "ipc://" + str(
            Path(runtime) / "control.sock"
        )
        asyncio.run(benchmark(args))


if __name__ == "__main__":
    main()
