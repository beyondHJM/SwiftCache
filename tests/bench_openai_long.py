"""Long-context end-to-end streaming through the PUBLIC OpenAI gateway."""

import argparse
import json
from pathlib import Path
import statistics
import time
import httpx
from openai import OpenAI
from transformers import AutoTokenizer


def stats(v):
    v = sorted(v)
    return dict(
        mean=statistics.mean(v),
        p50=statistics.median(v),
        p95=v[min(len(v) - 1, int(0.95 * len(v)))],
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--metrics-url", default="http://127.0.0.1:18001")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained("/root/llm-resource/Models/Qwen3-8B")
    document = (
        "This reference document describes a reliable cache service and its scheduling algorithm. "
        * 10000
    )
    document_tokens = tokenizer.encode(document)[:101376]
    prefix = tokenizer.decode(document_tokens)
    new_text = " The new request must preserve each cached token." * 100
    client = OpenAI(
        base_url="http://127.0.0.1:18000/v1",
        api_key="local",
        max_retries=0,
        timeout=900,
    )
    for count in [25000, 50000, 75000, 101376]:
        value = client.chat.completions.create(
            model="qwen3-8b-master",
            messages=[
                dict(
                    role="user",
                    content=tokenizer.decode(document_tokens[:count])
                    + "\nReply only with OK.",
                )
            ],
            max_tokens=1,
            temperature=0,
        )
        print("OPENAI_PREWARM", count, value.usage.prompt_tokens, flush=True)
    records = []
    for i in range(10 + args.samples):
        messages = [
            dict(
                role="user",
                content=prefix
                + f"\nRequest number {i}: "
                + new_text
                + "\nReply only with OK.",
            )
        ]
        tokenize = time.perf_counter()
        ids = tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True, enable_thinking=False
        )
        tokenize_ms = (time.perf_counter() - tokenize) * 1000
        start = time.perf_counter()
        first = None
        text = ""
        usage = None
        for chunk in client.chat.completions.create(
            model="qwen3-8b-master",
            messages=messages,
            max_tokens=1,
            temperature=0,
            stream=True,
            stream_options={"include_usage": True},
        ):
            if chunk.choices and chunk.choices[0].delta.content:
                first = first or time.perf_counter()
                text += chunk.choices[0].delta.content
            if chunk.usage:
                usage = chunk.usage
            rid = chunk.id
        end = time.perf_counter()
        assert usage is not None and usage.prompt_tokens == len(ids)
        cached = usage.prompt_tokens_details.cached_tokens
        if i >= 10:
            assert cached > 100000 and 500 < usage.prompt_tokens - cached < 2000, (
                usage,
                cached,
            )
            response = httpx.get(
                args.metrics_url + "/request-metrics/" + rid,
                timeout=30,
                trust_env=False,
            )
            response.raise_for_status()
            server_metrics = response.json()
            records.append(
                dict(
                    prompt_tokens=usage.prompt_tokens,
                    cached_tokens=cached,
                    uncached_tokens=usage.prompt_tokens - cached,
                    client_tokenize_check_ms=tokenize_ms,
                    first_content_ms=(first - start) * 1000 if first else None,
                    total_ms=(end - start) * 1000,
                    output=text,
                    server_metrics=server_metrics,
                )
            )
        print("OPENAI_LONG", i, len(ids), cached, (end - start) * 1000, flush=True)
    data = dict(
        samples=records,
        summary={
            k: stats([r[k] for r in records])
            for k in ["client_tokenize_check_ms", "first_content_ms", "total_ms"]
        },
    )
    data["summary"]["server_text_preprocess_ms"] = stats(
        [r["server_metrics"]["api_preprocess_ms"] for r in records]
    )
    args.output.write_text(json.dumps(data, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
