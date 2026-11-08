"""Explicit live-server SDK checks; not collected as a CPU unit test."""

import argparse
import asyncio
import json
from pathlib import Path
import time

import httpx
from openai import OpenAI, AsyncOpenAI, APIStatusError


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--url", default="http://127.0.0.1:18100")
    p.add_argument("--prefix-file", type=Path, required=True)
    p.add_argument(
        "--output", type=Path, default=Path("artifacts/aqua/http-validation.json")
    )
    args = p.parse_args()
    for _ in range(300):
        try:
            response = httpx.get(args.url + "/health", timeout=2, trust_env=False)
            if response.status_code == 200:
                break
        except httpx.HTTPError:
            pass
        time.sleep(0.5)
    else:
        raise TimeoutError("Aqua baseline API did not start")
    client = OpenAI(
        base_url=args.url + "/v1", api_key="local", max_retries=0, timeout=120
    )
    names = {m.id for m in client.models.list()}
    assert names == {"qwen3-8b-consumer", "qwen3-8b-producer"}
    results = []
    for name in sorted(names):
        r = client.chat.completions.create(
            model=name,
            messages=[
                dict(role="user", content="What is 2 + 2? Reply only with the number.")
            ],
            max_tokens=8,
            temperature=0,
        )
        assert r.choices[0].message.content.strip() == "4"
        assert r.usage.total_tokens == r.usage.prompt_tokens + r.usage.completion_tokens
        results.append(dict(test="model-routing", model=name, response=r.model_dump()))
    entry = json.loads(args.prefix_file.read_text())[0]
    ids = entry["token_ids"]
    cold = client.completions.create(
        model="qwen3-8b-consumer", prompt=ids, max_tokens=8, temperature=0
    )
    warm = client.completions.create(
        model="qwen3-8b-consumer",
        prompt=ids,
        max_tokens=8,
        temperature=0,
        extra_body={"prefix_id": entry["id"]},
    )
    assert cold.choices[0].text == warm.choices[0].text
    assert warm.usage.prompt_tokens_details.cached_tokens == len(ids)
    for fields in [
        dict(stream=True),
        dict(temperature=0.5),
        dict(n=2),
        dict(tools=[{"type": "function"}]),
    ]:
        params = dict(
            model="qwen3-8b-consumer",
            messages=[dict(role="user", content="Hello")],
            max_tokens=4,
        )
        params.update(fields)
        try:
            client.chat.completions.create(**params)
        except APIStatusError as exc:
            assert exc.status_code == 400
        else:
            raise AssertionError("Unsupported option accepted: " + str(fields))

    async def concurrent():
        async with AsyncOpenAI(
            base_url=args.url + "/v1", api_key="local", max_retries=0, timeout=120
        ) as ac:
            data = await asyncio.gather(
                *(
                    ac.chat.completions.create(
                        model="qwen3-8b-consumer",
                        messages=[
                            dict(
                                role="user",
                                content="What is 2 + 2? Reply only with the number.",
                            )
                        ],
                        max_tokens=8,
                        temperature=0,
                    )
                    for _ in range(4)
                )
            )
            assert all(r.choices[0].message.content.strip() == "4" for r in data)

    asyncio.run(concurrent())
    state = httpx.get(args.url + "/metrics", timeout=30, trust_env=False).json()
    assert state["pending"] == 0
    args.output.write_text(
        json.dumps(
            dict(
                sdk_sync=True,
                sdk_async_concurrent=4,
                unsupported_rejected=True,
                prefix_cached_tokens=len(ids),
                responses=results,
                prefix_response=warm.model_dump(),
                state=state,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print("AQUA_HTTP_VALIDATED", flush=True)


if __name__ == "__main__":
    main()
