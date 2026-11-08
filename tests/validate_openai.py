"""Live dual-Qwen tests using the official SDK and LangChain ChatOpenAI."""

import asyncio
import json
from pathlib import Path
import time
import httpx
from openai import OpenAI, AsyncOpenAI, APIStatusError

URL = "http://127.0.0.1:18000"
MODELS = ["qwen3-8b-master", "qwen3-8b-worker"]


def wait_ready():
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
        try:
            if httpx.get(URL + "/ready", timeout=5, trust_env=False).status_code == 200:
                return
        except httpx.HTTPError:
            pass
        time.sleep(1)
    raise TimeoutError("Unified serving did not become ready")


def main():
    wait_ready()
    records = []
    client = OpenAI(base_url=URL + "/v1", api_key="local", max_retries=0, timeout=120)
    assert {m.id for m in client.models.list()} == set(MODELS)
    for model in MODELS:
        response = client.chat.completions.create(
            model=model,
            messages=[
                dict(role="user", content="What is 2 + 2? Answer only with the number.")
            ],
            temperature=0,
            max_completion_tokens=16,
        )
        assert response.choices[0].message.content.strip() == "4"
        assert (
            response.usage.total_tokens
            == response.usage.prompt_tokens + response.usage.completion_tokens
        )
        records.append(dict(test="chat", model=model, response=response.model_dump()))
    messages = [
        dict(role="user", content="用中文依次列出一到二十，每个数字之间用逗号分隔。")
    ]
    reference = client.chat.completions.create(
        model=MODELS[0], messages=messages, temperature=0, max_tokens=100
    )
    started = time.perf_counter()
    first = None
    chunks = []
    text = ""
    usage = None
    stream = client.chat.completions.create(
        model=MODELS[0],
        messages=messages,
        temperature=0,
        max_tokens=100,
        stream=True,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        chunks.append(chunk)
        if chunk.choices:
            delta = chunk.choices[0].delta.content or ""
            if delta and first is None:
                first = time.perf_counter()
            text += delta
        if chunk.usage:
            usage = chunk.usage
    finished = time.perf_counter()
    assert text == reference.choices[0].message.content
    assert len({c.id for c in chunks}) == 1 and len({c.created for c in chunks}) == 1
    assert usage and len(chunks) > 4 and first is not None and finished - first > 0.01
    records.append(
        dict(
            test="streaming",
            text=text,
            ttft_ms=(first - started) * 1000,
            total_ms=(finished - started) * 1000,
            chunks=len(chunks),
            usage=usage.model_dump(),
        )
    )
    stopped = client.chat.completions.create(
        model=MODELS[0],
        messages=[
            dict(role="user", content="What is 2 + 2? Answer only with the number.")
        ],
        max_tokens=16,
        temperature=0,
        stop="4",
    )
    assert (
        stopped.choices[0].message.content == ""
        and stopped.choices[0].finish_reason == "stop"
    )
    assert stopped.usage.completion_tokens == 1
    completion = client.completions.create(
        model=MODELS[0], prompt="The capital of France is", max_tokens=8, temperature=0
    )
    assert completion.choices[0].text
    records.append(dict(test="completion", response=completion.model_dump()))
    for params in [
        dict(model="missing", messages=messages),
        dict(model=MODELS[0], messages=messages, n=2),
        dict(model=MODELS[0], messages=messages, max_tokens=10000),
        dict(model=MODELS[0], messages=messages, seed=0),
        dict(model=MODELS[0], messages=messages, stop=""),
    ]:
        try:
            client.chat.completions.create(**params)
        except APIStatusError as exc:
            assert exc.status_code in (400, 404)
        else:
            raise AssertionError(("Invalid request was accepted", params))
    for payload in [[], {"model": []}]:
        bad = httpx.post(URL + "/v1/chat/completions", json=payload, trust_env=False)
        assert bad.status_code in (400, 404) and "error" in bad.json()
    multi = client.chat.completions.create(
        model=MODELS[0],
        messages=[
            dict(role="user", content="Remember code ALPHA-42."),
            dict(role="assistant", content="I will remember ALPHA-42."),
            dict(role="user", content="What is the code? Reply only with the code."),
        ],
        temperature=0,
        max_tokens=16,
    )
    assert "ALPHA-42" in multi.choices[0].message.content
    across = client.chat.completions.create(
        model=MODELS[0],
        messages=messages,
        temperature=0,
        max_tokens=100,
        stream=True,
        stop="十三，十四",
    )
    stopped = "".join(
        chunk.choices[0].delta.content or "" for chunk in across if chunk.choices
    )
    assert stopped == reference.choices[0].message.content.split("十三，十四")[0], (
        stopped
    )
    records.append(
        dict(
            test="multi_turn_and_cross_chunk_stop",
            multi=multi.choices[0].message.content,
            stopped=stopped,
        )
    )

    async def asynchronous():
        async with AsyncOpenAI(
            base_url=URL + "/v1", api_key="local", max_retries=0
        ) as ac:
            values = await asyncio.gather(
                *(
                    ac.chat.completions.create(
                        model=MODELS[i % 2],
                        messages=[
                            dict(
                                role="user",
                                content="What is 2 + 2? Answer only with the number.",
                            )
                        ],
                        temperature=0,
                        max_tokens=8,
                    )
                    for i in range(4)
                )
            )
            assert all(v.choices[0].message.content.strip() == "4" for v in values)

    asyncio.run(asynchronous())
    from langchain_openai import ChatOpenAI

    third_party = ChatOpenAI(
        base_url=URL + "/v1",
        api_key="local",
        model=MODELS[0],
        temperature=0,
        max_tokens=16,
        use_responses_api=False,
    )
    value = third_party.invoke("What is 2 + 2? Answer only with the number.")
    assert value.content.strip() == "4"
    streamed = "".join(
        chunk.content
        for chunk in third_party.stream("What is 2 + 2? Answer only with the number.")
    )
    assert streamed.strip() == "4"
    records.append(dict(test="langchain", text=value.content, streamed=streamed))
    stream = client.chat.completions.create(
        model=MODELS[0],
        messages=[dict(role="user", content="List every integer from 1 to 1000.")],
        max_tokens=1000,
        temperature=0,
        stream=True,
    )
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            break
    stream.close()
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        state = httpx.get(URL + "/cache", timeout=30, trust_env=False).json()
        if all(
            s["request_records"] == s["scheduled_requests"] == s["used"] == 0
            for s in state.values()
        ):
            break
        time.sleep(0.1)
    else:
        raise AssertionError(("Cancelled stream retained state", state))
    Path("artifacts/phase2/openai-validation.json").write_text(
        json.dumps(
            dict(
                tests=records,
                official_sdk=True,
                async_requests=4,
                third_party="langchain_openai.ChatOpenAI",
                stop_and_error_checks=True,
                cancellation_clean=True,
                final_state=state,
            ),
            ensure_ascii=False,
            indent=2,
        )
    )
    print(
        "OpenAI SDK sync/async/chat/completions/stream/usage/errors/stop/cancel and LangChain invoke/stream passed",
        flush=True,
    )
    print(json.dumps(records[2], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
