"""The sole model-process and HTTP lifecycle implementation."""

import argparse
import json
import multiprocessing as mp
import os
from pathlib import Path
import signal
import tempfile
import time

from server.configuration import load_config


def model_process(rank, config, events, ready):
    import torch

    torch.cuda.set_device(rank)
    from swiftcache import LLM
    from server.openai_app import make_app
    import atexit
    import uvicorn

    instance = config.models[rank]
    workers = list(range(1, len(config.models)))
    options = dict(instance.engine)
    llm = LLM(
        instance.model,
        rank=rank,
        role=instance.role,
        dist_port=config.dist_port,
        tp_group=[rank],
        master_list=[0],
        slave_list=workers,
        slave_event=events if rank == 0 else events[rank],
        slave_ready_event=ready if rank == 0 else ready[rank],
        **options,
    )
    atexit.unregister(llm.exit)
    app = make_app(llm, instance, config)
    uvicorn.run(
        app,
        host="127.0.0.1",
        port=config.internal_base_port + rank,
        log_level="warning",
        timeout_graceful_shutdown=config.shutdown_timeout,
    )


def gateway_process(config):
    from server.gateway import make_gateway
    import uvicorn

    uvicorn.run(
        make_gateway(config),
        host=config.host,
        port=config.port,
        log_level="warning",
        timeout_graceful_shutdown=config.shutdown_timeout,
    )


def launch(config):
    config.validate()
    if os.environ.get("USE_CSCHEDULER", "").lower() in ("1", "true", "yes"):
        raise ValueError("Unified elastic serving requires the Python scheduler")
    devices = ",".join(str(m.gpu) for m in config.models)
    visible = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible and visible != devices:
        raise ValueError(
            "CUDA_VISIBLE_DEVICES conflicts with configured GPUs; use --gpus or update the configuration"
        )
    os.environ["CUDA_VISIBLE_DEVICES"] = devices
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    runtime = Path(tempfile.mkdtemp(prefix="swiftcache-"))
    os.environ["SWIFTCACHE_RUNTIME_DIR"] = str(runtime)
    os.environ["SWIFTCACHE_SHM_NAME"] = runtime.name.replace("-", "_")
    os.environ["SWIFTCACHE_CONTROL_ADDRESS"] = "ipc://" + str(runtime / "control.sock")
    ctx = mp.get_context("spawn")
    events = {i: ctx.Event() for i in range(1, len(config.models))}
    ready = {i: ctx.Event() for i in events}
    children = [
        ctx.Process(
            target=model_process,
            args=(i, config, events, ready),
            name=config.models[i].name,
        )
        for i in list(events) + [0]
    ]
    children.append(
        ctx.Process(target=gateway_process, args=(config,), name="openai-gateway")
    )
    old_handler = signal.getsignal(signal.SIGTERM)

    def stop(signum, frame):
        raise KeyboardInterrupt

    signal.signal(signal.SIGTERM, stop)
    try:
        print("SWIFTCACHE_CONFIG " + json.dumps(config.as_dict()), flush=True)
        print(f"SWIFTCACHE_PARENT pid={os.getpid()} runtime={runtime}", flush=True)
        for process in children:
            process.start()
        deadline = time.monotonic() + config.startup_timeout
        next_check = 0
        started = False
        while all(p.is_alive() for p in children):
            if not started and time.monotonic() >= next_check:
                import urllib.request

                try:
                    probe_host = (
                        "127.0.0.1"
                        if config.host in ("0.0.0.0", "localhost")
                        else "::1"
                        if config.host == "::"
                        else config.host
                    )
                    if ":" in probe_host:
                        probe_host = f"[{probe_host}]"
                    probe = urllib.request.urlopen(
                        f"http://{probe_host}:{config.port}/ready", timeout=1
                    )
                    with probe:
                        started = probe.status == 200
                except Exception:
                    pass
                next_check = time.monotonic() + 1
                if started:
                    print(
                        f"SWIFTCACHE_READY http://{config.host}:{config.port}/v1",
                        flush=True,
                    )
                elif time.monotonic() > deadline:
                    raise TimeoutError(
                        "Models did not become ready before startup_timeout"
                    )
            time.sleep(0.25)
        failed = [(p.name, p.exitcode) for p in children if not p.is_alive()]
        raise RuntimeError(f"Serving process exited: {failed}")
    except KeyboardInterrupt:
        pass
    finally:
        for p in reversed(children):
            if p.is_alive():
                p.terminate()
        deadline = time.monotonic() + config.shutdown_timeout + 5
        for p in children:
            if p.pid is not None:
                p.join(timeout=max(0, deadline - time.monotonic()))
                if p.is_alive():
                    p.kill()
                    p.join()
            p.close()
        from multiprocessing.shared_memory import SharedMemory

        try:
            memory = SharedMemory(name=os.environ["SWIFTCACHE_SHM_NAME"])
            memory.close()
            memory.unlink()
        except FileNotFoundError:
            pass
        for path in runtime.iterdir():
            path.unlink()
        runtime.rmdir()
        signal.signal(signal.SIGTERM, old_handler)


def parse_config(argv=None):
    parser = argparse.ArgumentParser(
        description="SwiftCache multi-model OpenAI-compatible service"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "configs/two_qwen8b.yaml",
    )
    parser.add_argument("--host")
    parser.add_argument("--port", type=int)
    parser.add_argument("--internal-base-port", type=int)
    parser.add_argument("--dist-port", type=int)
    parser.add_argument("--model", "-m")
    parser.add_argument("--gpus")
    parser.add_argument("--max-model-len", type=int)
    parser.add_argument("--ignore-model-max-length", action="store_true", default=None)
    parser.add_argument("--print-config", action="store_true")
    args = parser.parse_args(argv)
    config = load_config(args.config)
    for key in ["host", "port", "internal_base_port", "dist_port"]:
        value = getattr(args, key)
        if value is not None:
            setattr(config, key, value)
    if args.gpus:
        gpus = [int(x) for x in args.gpus.split(",")]
        if len(gpus) != len(config.models):
            parser.error("Provide one GPU per model")
        for m, gpu in zip(config.models, gpus):
            m.gpu = gpu
    for m in config.models:
        if args.model:
            m.model = args.model
        if args.max_model_len is not None:
            m.engine["max_model_len"] = args.max_model_len
        if args.ignore_model_max_length is not None:
            m.engine["ignore_model_max_length"] = args.ignore_model_max_length
    config.validate()
    if args.print_config:
        print(json.dumps(config.as_dict(), indent=2))
        return None
    return config


def main(argv=None):
    config = parse_config(argv)
    if config is not None:
        launch(config)
