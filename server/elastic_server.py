"""Legacy CLI adapter. Process/model/API implementation lives in launcher.py."""

import argparse
from pathlib import Path
from server.configuration import load_config
from server.launcher import launch


def main():
    p = argparse.ArgumentParser(
        description="Legacy two-GPU adapter; prefer server.starter --config"
    )
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=18800)
    p.add_argument("--dist-port", type=int, default=18334)
    p.add_argument("--cache-blocks", type=int, default=32)
    p.add_argument("--initial-blocks", type=int, default=1)
    p.add_argument("--max-model-len", type=int, default=8192)
    p.add_argument("--window", type=float, default=60)
    p.add_argument("--check-interval", type=float, default=1)
    p.add_argument("--fixed", action="store_true")
    p.add_argument("--ignore-model-max-length", action="store_true")
    args = p.parse_args()
    config = load_config(
        Path(__file__).resolve().parents[1] / "configs/two_qwen8b.yaml"
    )
    config.internal_base_port = args.port
    config.port = args.port - 1
    config.dist_port = args.dist_port
    for m in config.models:
        m.model = args.model
        m.engine.update(
            max_model_len=args.max_model_len,
            max_num_batched_tokens=args.max_model_len * 2,
            cache_block_limit=args.cache_blocks,
            elastic_initial_blocks=args.initial_blocks,
            elastic_enabled=not args.fixed,
            elastic_window_seconds=args.window,
            elastic_check_seconds=args.check_interval,
            ignore_model_max_length=args.ignore_model_max_length,
        )
    launch(config)


if __name__ == "__main__":
    main()
