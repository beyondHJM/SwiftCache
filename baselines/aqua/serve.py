"""Small OpenAI-shaped nonstream greedy interface for baseline workload tools."""

import asyncio
from contextlib import asynccontextmanager
import time

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse


def make_app(engine):
    names = {"qwen3-8b-consumer": "consumer"}
    if engine.producer_model:
        names["qwen3-8b-producer"] = "producer"

    @asynccontextmanager
    async def lifespan(app):
        engine.start()
        yield
        engine.close()

    app = FastAPI(lifespan=lifespan)

    @app.exception_handler(ValueError)
    async def invalid(request, exc):
        return JSONResponse(
            {
                "error": {
                    "message": str(exc),
                    "type": "invalid_request_error",
                    "code": "invalid_request_error",
                }
            },
            status_code=400,
        )

    @app.exception_handler(RuntimeError)
    async def failed(request, exc):
        return JSONResponse(
            {
                "error": {
                    "message": str(exc),
                    "type": "server_error",
                    "code": "server_error",
                }
            },
            status_code=503,
        )

    @app.get("/health")
    async def health():
        return JSONResponse(
            dict(status="ready" if not engine.fatal else "failed", error=engine.fatal),
            status_code=503 if engine.fatal else 200,
        )

    @app.get("/v1/models")
    async def models():
        return dict(
            object="list",
            data=[
                dict(id=n, object="model", created=0, owned_by="aqua-baseline")
                for n in names
            ],
        )

    @app.get("/metrics")
    async def metrics():
        ledger = await asyncio.to_thread(engine.client.status)
        return dict(
            mode=engine.mode,
            pending=len(engine.jobs),
            host_swap_bytes=engine.store.bytes,
            ledger=ledger,
            events=engine.trace.snapshot(),
        )

    async def complete(request, chat):
        data = await request.json()
        if (
            not isinstance(data, dict)
            or not isinstance(data.get("model"), str)
            or data.get("model") not in names
        ):
            raise ValueError("Unknown model")
        allowed = {
            "model",
            "messages" if chat else "prompt",
            "max_tokens",
            "max_completion_tokens",
            "temperature",
            "top_p",
            "n",
            "stream",
            "prefix_id",
        }
        if set(data) - allowed:
            raise ValueError("Unsupported fields: " + str(sorted(set(data) - allowed)))
        if (
            data.get("stream", False)
            or data.get("temperature", 0) != 0
            or data.get("top_p", 1) != 1
            or data.get("n", 1) != 1
        ):
            raise ValueError(
                "Minimal baseline supports nonstream, greedy temperature=0, top_p=1, n=1 only"
            )
        if (
            data.get("max_tokens") is not None
            and data.get("max_completion_tokens") is not None
        ):
            raise ValueError("Choose one output length field")
        length = data.get("max_completion_tokens", data.get("max_tokens", 32))
        role = names[data["model"]]
        model = engine.consumer if role == "consumer" else engine.producer_model
        ids = (
            model.ids(messages=data.get("messages"))
            if chat
            else model.ids(prompt=data.get("prompt"))
        )
        req = engine.submit(ids, length, role, data.get("prefix_id"))
        future = asyncio.wrap_future(req.future)
        try:
            while not future.done():
                await asyncio.wait([future], timeout=0.1)
                if await request.is_disconnected():
                    req.cancelled = True
                    raise asyncio.CancelledError()
            result = await future
        finally:
            if not future.done():
                req.cancelled = True
                future.cancel()
        choice = dict(index=0, finish_reason=result["finish_reason"])
        if chat:
            choice["message"] = dict(role="assistant", content=result["text"])
        else:
            choice.update(text=result["text"], logprobs=None)
        return dict(
            id="chatcmpl-" + req.ident if chat else "cmpl-" + req.ident,
            created=int(time.time()),
            object="chat.completion" if chat else "text_completion",
            model=data["model"],
            choices=[choice],
            usage=dict(
                prompt_tokens=len(ids),
                completion_tokens=len(result["token_ids"]),
                total_tokens=len(ids) + len(result["token_ids"]),
                prompt_tokens_details={"cached_tokens": result["cached_tokens"]},
            ),
        )

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await complete(request, True)

    @app.post("/v1/completions")
    async def completion(request: Request):
        return await complete(request, False)

    return app
