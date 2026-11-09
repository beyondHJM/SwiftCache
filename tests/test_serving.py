import json
from pathlib import Path
from types import SimpleNamespace
import pytest
from server.openai_app import APIError, validate_options, authenticate, tokenize_request
from server.configuration import load_config


def test_options_reject_silent_inference_fallback():
    for fields in [
        dict(tools=[{"type": "function"}]),
        dict(n=2),
        dict(top_p=0),
        dict(temperature=-1),
        dict(max_tokens=1, max_completion_tokens=2),
        dict(seed=123),
        dict(seed=0),
        dict(stop=""),
        dict(temperature=True),
        dict(stream_options={"include_usage": 1}),
        dict(unrecognized=True),
    ]:
        with pytest.raises(APIError):
            validate_options(fields, True)
    assert validate_options(
        dict(max_completion_tokens=10, top_p=0.9, stop="END"), True
    ) == (10, 0.0, 0.9, ["END"])


def test_model_config_length_override(tmp_path):
    from swiftcache.config import Config

    (tmp_path / "config.json").write_text(
        json.dumps(dict(model_type="qwen3", max_position_embeddings=40960))
    )
    limited = Config(str(tmp_path), max_model_len=102400)
    extended = Config(
        str(tmp_path),
        max_model_len=102400,
        ignore_model_max_length=True,
        max_num_batched_tokens=2048,
    )
    assert limited.max_model_len == 40960
    assert (
        extended.max_model_len == extended.hf_config.max_position_embeddings == 102400
    )
    assert (
        extended.original_model_max_length == 40960
        and extended.hf_config.rope_scaling is None
    )


def test_long_pretokenized_request_queries_vocabulary_once():
    class Tokenizer:
        calls = 0

        def __len__(self):
            self.calls += 1
            return 151936

    tokenizer = Tokenizer()
    ids = [1000] * 102400
    assert tokenize_request(tokenizer, {"prompt": ids}, False, False) is ids
    assert tokenizer.calls == 1
    with pytest.raises(APIError):
        tokenize_request(tokenizer, {"prompt": [151936]}, False, False)


def test_config_conflicting_ports_and_models(tmp_path):
    p = Path(__file__).resolve().parents[1] / "configs/two_qwen8b.yaml"
    cfg = load_config(p)
    for m in cfg.models:
        m.model = str(tmp_path)
    cfg.port = cfg.internal_base_port
    with pytest.raises(ValueError, match="ports"):
        cfg.validate()
    cfg.port = 18000
    cfg.models[1].name = cfg.models[0].name
    with pytest.raises(ValueError, match="unique"):
        cfg.validate()


def test_service_bearer_key(monkeypatch):
    monkeypatch.setenv("TEST_SERVICE_KEY", "test-key")
    cfg = SimpleNamespace(api_key_env="TEST_SERVICE_KEY")
    authenticate(SimpleNamespace(headers={"authorization": "Bearer test-key"}), cfg)
    with pytest.raises(APIError) as exc:
        authenticate(SimpleNamespace(headers={}), cfg)
    assert exc.value.status == 401


def test_nucleus_sampler_and_greedy():
    import torch
    from swiftcache.layers.sampler import Sampler

    logits = torch.tensor([[0.0, 1.0, 10.0, -1.0]] * 100)
    samples = Sampler()(logits, torch.ones(100), torch.full((100,), 0.01))
    assert bool(torch.all(samples == 2))
    samples = Sampler()(logits, torch.zeros(100), torch.full((100,), 0.1))
    assert bool(torch.all(samples == 2))


def test_live_shared_prefix_admission_and_cache_plan():
    from bench_scheduler import scheduler
    from swiftcache.engine.sequence import Sequence
    from swiftcache.sampling_params import SamplingParams

    s = scheduler(448)
    a = [100 + (i % 999) for i in range(102400)]
    first = Sequence(a, SamplingParams(max_tokens=8))
    s.add(first)
    _, kind = s.schedule()
    assert kind == "prefill"
    b = a.copy()
    b[101376] = 44444
    second = Sequence(b, SamplingParams(max_tokens=8))
    s.add(second)
    seqs, kind = s.schedule()
    assert kind == "prefill" and seqs == [second]
    assert second.num_cached_tokens == 101376
    assert len(s.block_manager.used_block_ids) == 404
    s.preempt(second)
    s.preempt(first)
    assert not s.block_manager.used_block_ids
    s.block_manager.assert_invariants()


def test_fixed_worker_capacity_is_not_counted_as_borrowable():
    from test_elastic import pair

    master, worker, _, _ = pair(total=12)
    worker.request_resize(8)
    master.poll()
    worker.poll()
    worker.config.elastic_enabled = False
    master.peers[b"slave1"]["elastic_enabled"] = False
    assert worker.maximum_blocks() == 8
    assert master.maximum_blocks() == 4
    assert master.maximum_blocks() + worker.maximum_blocks() == 12
