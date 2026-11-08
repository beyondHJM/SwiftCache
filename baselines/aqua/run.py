"""Run the bounded Aqua baseline or its real-GPU validation workload."""

import argparse
import contextlib
import json
import os
from pathlib import Path
import socket
import signal
import subprocess
import sys
import time
import urllib.request


@contextlib.contextmanager
def coordinator(port, log):
    with socket.socket() as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        sock.bind(("127.0.0.1", port))
    with log.open("w") as output:
        process = subprocess.Popen(
            [sys.executable, "-m", "baselines.aqua.coordinator", "--port", str(port)],
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            for _ in range(100):
                if process.poll() is not None:
                    raise RuntimeError("Aqua coordinator failed; see " + str(log))
                try:
                    with urllib.request.urlopen(
                        f"http://127.0.0.1:{port}/health", timeout=1
                    ) as response:
                        if response.status == 200:
                            break
                except Exception:
                    time.sleep(0.1)
            else:
                raise TimeoutError("Aqua coordinator startup timed out")
            yield
        finally:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()


def main():
    p = argparse.ArgumentParser(
        description="Minimal Aqua Qwen3 baseline (upstream Aqua tensors + HF executor)"
    )
    p.add_argument("--model", default="/root/llm-resource/Models/Qwen3-8B")
    p.add_argument("--gpus", default="0,2")
    p.add_argument("--offer-mib", type=int, default=512)
    p.add_argument("--host-swap-mib", type=int, default=2048)
    p.add_argument("--chunk-size", type=int, default=128)
    p.add_argument("--token-budget", type=int, default=256)
    p.add_argument("--decode-slots", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--ignore-model-max-length", action="store_true")
    p.add_argument("--coordinator-port", type=int, default=18101)
    p.add_argument("--port", type=int, default=18100)
    p.add_argument("--no-producer-model", action="store_true")
    p.add_argument(
        "--scenario", choices=["validate", "serve", "generate"], default="validate"
    )
    p.add_argument("--prompt", default="What is 2 + 2? Answer only with the number.")
    p.add_argument("--max-tokens", type=int, default=32)
    p.add_argument(
        "--prefix-file",
        type=Path,
        help="JSON list of {id, token_ids or prompt} to cache before serving",
    )
    p.add_argument(
        "--output", type=Path, default=Path("artifacts/aqua/validation.json")
    )
    args = p.parse_args()

    def interrupt(signum, frame):
        raise KeyboardInterrupt()

    signal.signal(signal.SIGTERM, interrupt)
    devices = [int(x) for x in args.gpus.split(",")]
    if len(devices) != 2 or len(set(devices)) != 2 or min(devices) < 0:
        p.error("Provide two distinct physical GPU IDs")
    if args.offer_mib < 0 or args.host_swap_mib <= 0 or args.max_model_len <= 0:
        p.error("Invalid memory/context budgets")
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible != args.gpus:
        p.error("CUDA_VISIBLE_DEVICES conflicts with --gpus")
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpus
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.scenario == "serve":
        with socket.socket() as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", args.port))
    with coordinator(
        args.coordinator_port, args.output.with_suffix(".coordinator.log")
    ):
        from .engine import Engine

        engine = Engine(
            args.model,
            devices,
            args.coordinator_port,
            args.offer_mib << 20,
            args.host_swap_mib << 20,
            args.chunk_size,
            args.token_budget,
            args.decode_slots,
            not args.no_producer_model,
            args.max_model_len,
            ignore_model_max_length=args.ignore_model_max_length,
        )
        try:
            if args.prefix_file:
                entries = json.loads(args.prefix_file.read_text(encoding="utf-8"))
                if not isinstance(entries, list):
                    raise ValueError("prefix-file must contain a JSON list")
                for entry in entries:
                    ids = engine.consumer.ids(
                        prompt=entry.get("token_ids", entry.get("prompt"))
                    )
                    engine.cache_prefix(ids, entry["id"])
            if args.scenario == "validate":
                from .validate import validate

                result = validate(engine)
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(
                    "AQUA_VALIDATED",
                    json.dumps(result["summary"], ensure_ascii=False),
                    flush=True,
                )
            elif args.scenario == "generate":
                ids = engine.consumer.ids(
                    messages=[dict(role="user", content=args.prompt)]
                )
                result = engine.drain([engine.submit(ids, args.max_tokens)])[0]
                args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2))
                print(result["text"], flush=True)
            else:
                import uvicorn
                from .serve import make_app

                uvicorn.run(
                    make_app(engine),
                    host="127.0.0.1",
                    port=args.port,
                    log_level="warning",
                )
        finally:
            engine.close()
            state = engine.client.status()
            assert not state["allocations"] and not state["providers"]
            args.output.with_suffix(".final-ledger.json").write_text(
                json.dumps(state, indent=2)
            )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
