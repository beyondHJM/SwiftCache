"""Validated configuration shared by every serving entry point."""

from dataclasses import dataclass, field, asdict
import os
from pathlib import Path
import yaml


@dataclass
class ModelInstance:
    name: str
    model: str
    gpu: int
    role: str
    engine: dict = field(default_factory=dict)


@dataclass
class ServiceConfig:
    models: list[ModelInstance]
    host: str = "127.0.0.1"
    port: int = 18000
    internal_base_port: int = 18001
    dist_port: int = 18334
    api_key_env: str | None = None
    enable_thinking: bool = False
    shutdown_timeout: float = 10.0
    startup_timeout: float = 300.0

    def validate(self):
        if self.startup_timeout <= 0 or self.shutdown_timeout <= 0:
            raise ValueError("Startup and shutdown timeouts must be positive")
        if len(self.models) < 2 or sum(m.role == "master" for m in self.models) != 1:
            raise ValueError("Configure exactly one master and at least one worker")
        self.models.sort(key=lambda m: m.role != "master")
        if len({m.name for m in self.models}) != len(self.models):
            raise ValueError("Served model names must be unique")
        if len({m.gpu for m in self.models}) != len(self.models):
            raise ValueError("Assign a separate GPU to each model")
        ports = [self.port, self.dist_port] + list(
            range(self.internal_base_port, self.internal_base_port + len(self.models))
        )
        if len(set(ports)) != len(ports) or not all(1 <= p <= 65535 for p in ports):
            raise ValueError(
                "Public, internal and distributed ports must be distinct valid ports"
            )
        allowed = {
            "max_model_len",
            "max_num_batched_tokens",
            "max_num_seqs",
            "gpu_memory_utilization",
            "cache_block_limit",
            "elastic_enabled",
            "elastic_initial_blocks",
            "elastic_window_seconds",
            "elastic_check_seconds",
            "ignore_model_max_length",
            "warmup_tokens",
            "profile_stages",
            "control_timeout_seconds",
            "enforce_eager",
            "tensor_parallel_size",
            "scheduler_backend",
        }
        for m in self.models:
            if m.engine.get("scheduler_backend", "python") not in ("python", "cpp"):
                raise ValueError("scheduler_backend must be python or cpp")
            if m.role not in ("master", "slave") or m.gpu < 0 or not m.name:
                raise ValueError("Invalid model role, GPU or name")
            m.model = os.path.expandvars(os.path.expanduser(m.model))
            if not Path(m.model).is_dir():
                raise ValueError(f"Model directory does not exist: {m.model}")
            unknown = set(m.engine) - allowed
            if unknown:
                raise ValueError(f"Unknown engine options: {sorted(unknown)}")
            for key in [
                "max_model_len",
                "max_num_batched_tokens",
                "max_num_seqs",
                "warmup_tokens",
                "elastic_initial_blocks",
            ]:
                if key in m.engine and (
                    type(m.engine[key]) is not int or m.engine[key] <= 0
                ):
                    raise ValueError(f"{key} must be a positive integer")
            for key in [
                "elastic_enabled",
                "ignore_model_max_length",
                "profile_stages",
                "enforce_eager",
            ]:
                if key in m.engine and type(m.engine[key]) is not bool:
                    raise ValueError(f"{key} must be boolean")
            if m.engine.get("tensor_parallel_size", 1) != 1 or not m.engine.get(
                "enforce_eager", True
            ):
                raise ValueError(
                    "Current elastic serving requires TP=1 and eager execution"
                )
            initial = m.engine.get("elastic_initial_blocks", 1)
            limit = m.engine.get("cache_block_limit", -1)
            if initial < 1 or (limit > 0 and initial > limit):
                raise ValueError("Invalid cache block limits")
        return self

    def as_dict(self):
        return asdict(self)


def load_config(path):
    data = yaml.safe_load(Path(path).read_text())
    if not isinstance(data, dict):
        raise ValueError("Configuration must be a mapping")
    data = dict(data)
    data["models"] = [
        ModelInstance(**dict(m, role="slave" if m["role"] == "worker" else m["role"]))
        for m in data["models"]
    ]
    for m in data["models"]:
        value = Path(os.path.expandvars(os.path.expanduser(m.model)))
        if not value.is_absolute():
            m.model = str((Path(path).resolve().parent / value).resolve())
    return ServiceConfig(**data)
