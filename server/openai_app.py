"""OpenAI text protocols over the same SwiftCache engine used by legacy calls."""

import asyncio
from contextlib import asynccontextmanager, suppress
import json
import os
import secrets
import time
import uuid
from collections import OrderedDict

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, StreamingResponse
from starlette.exceptions import HTTPException


class APIError(Exception):
    def __init__(self, message, status=400, param=None, code="invalid_request_error"):
        self.message = message
        self.status = status
        self.param = param
        self.code = code


def error_body(
    message,
    param=None,
    code="invalid_request_error",
    error_type="invalid_request_error",
):
    return {"error": dict(message=message, type=error_type, param=param, code=code)}


def configure_errors(app):
    @app.exception_handler(APIError)
    async def error_handler(request, exc):
        category = (
            "server_error"
            if exc.status >= 500
            else "authentication_error"
            if exc.status == 401
            else "invalid_request_error"
        )
        return JSONResponse(
            error_body(exc.message, exc.param, exc.code, category),
            status_code=exc.status,
        )

    @app.exception_handler(RuntimeError)
    async def runtime_error(request, exc):
        return JSONResponse(
            error_body(str(exc), code="server_error", error_type="server_error"),
            status_code=500,
        )

    @app.exception_handler(ValueError)
    async def value_error(request, exc):
        return JSONResponse(error_body(str(exc)), status_code=400)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        return JSONResponse(error_body(str(exc)), status_code=400)

    @app.exception_handler(HTTPException)
    async def http_error(request, exc):
        return JSONResponse(error_body(str(exc.detail)), status_code=exc.status_code)


def authenticate(request, config):
    if config.api_key_env:
        expected = os.environ.get(config.api_key_env)
        if not expected:
            raise APIError(
                "Configured service key is unavailable", 503, code="service_unavailable"
            )
        received = request.headers.get("authorization", "")
        if not secrets.compare_digest(received, "Bearer " + expected):
            raise APIError("Invalid service API key", 401, code="invalid_api_key")


def validate_options(data, chat):
    allowed = {
        "model",
        "messages" if chat else "prompt",
        "max_tokens",
        "max_completion_tokens",
        "temperature",
        "top_p",
        "stop",
        "stream",
        "stream_options",
        "n",
        "user",
        "metadata",
        "frequency_penalty",
        "presence_penalty",
        "logprobs",
        "seed",
        "tools",
        "tool_choice",
        "response_format",
        "echo",
        "best_of",
    }
    unknown = set(data) - allowed
    if unknown:
        raise APIError(
            f"Unsupported fields: {sorted(unknown)}", param=sorted(unknown)[0]
        )
    for key in ["frequency_penalty", "presence_penalty"]:
        if data.get(key, 0) not in (None, 0):
            raise APIError(f"{key} is not implemented", param=key)
    for key, supported in [
        ("tools", (None, [])),
        ("seed", (None,)),
        ("logprobs", (None, False)),
    ]:
        if data.get(key) not in supported:
            raise APIError(f"{key} is not implemented", param=key)
    if data.get("tool_choice") not in (None, "none"):
        raise APIError("Tool calls are not implemented", param="tool_choice")
    if data.get("response_format") not in (None, {"type": "text"}):
        raise APIError("Only text output is implemented", param="response_format")
    if data.get("n", 1) != 1 or data.get("best_of", 1) != 1 or data.get("echo", False):
        raise APIError("This text endpoint supports n=1, best_of=1, echo=false")
    if (
        data.get("max_tokens") is not None
        and data.get("max_completion_tokens") is not None
    ):
        raise APIError(
            "Specify only one output length parameter", param="max_completion_tokens"
        )
    length = data.get("max_completion_tokens", data.get("max_tokens", 64))
    if type(length) is not int or length < 1:
        raise APIError("Output length must be a positive integer")
    temperature = data.get("temperature", 0)
    top_p = data.get("top_p", 1)
    if type(temperature) not in (int, float) or not 0 <= temperature <= 2:
        raise APIError("temperature must be between 0 and 2", param="temperature")
    if type(top_p) not in (int, float) or not 0 < top_p <= 1:
        raise APIError("top_p must be in (0,1]", param="top_p")
    stops = [] if data.get("stop") is None else data["stop"]
    if isinstance(stops, str):
        stops = [stops]
    if (
        not isinstance(stops, list)
        or len(stops) > 4
        or any(not isinstance(x, str) or not x or len(x) > 256 for x in stops)
    ):
        raise APIError("stop must contain up to four nonempty strings", param="stop")
    if type(data.get("stream", False)) is not bool:
        raise APIError("stream must be a boolean", param="stream")
    options = data.get("stream_options") or {}
    if not isinstance(options, dict) or set(options) - {
        "include_usage",
        "include_obfuscation",
    }:
        raise APIError("Unsupported stream_options", param="stream_options")
    if options.get("include_obfuscation"):
        raise APIError("Stream obfuscation is not implemented")
    if any(type(value) is not bool for value in options.values()):
        raise APIError("stream_options values must be booleans", param="stream_options")
    return length, float(temperature), float(top_p), stops


def tokenize_request(tokenizer, data, chat, thinking):
    if chat:
        messages = data.get("messages")
        if not isinstance(messages, list) or not messages:
            raise APIError("messages must be nonempty", param="messages")
        normalized = []
        for message in messages:
            if not isinstance(message, dict) or message.get("role") not in (
                "system",
                "developer",
                "user",
                "assistant",
            ):
                raise APIError(
                    "Only text system/developer/user/assistant messages are supported",
                    param="messages",
                )
            if set(message) - {"role", "content", "name"}:
                raise APIError("Unsupported message fields", param="messages")
            content = message.get("content")
            if isinstance(content, list):
                if any(
                    not isinstance(p, dict)
                    or p.get("type") != "text"
                    or not isinstance(p.get("text"), str)
                    for p in content
                ):
                    raise APIError(
                        "Only text content parts are supported", param="messages"
                    )
                content = "".join(p["text"] for p in content)
            if not isinstance(content, str):
                raise APIError("Message content must be text", param="messages")
            normalized.append(
                dict(
                    role="system"
                    if message["role"] == "developer"
                    else message["role"],
                    content=content,
                )
            )
        return tokenizer.apply_chat_template(
            normalized,
            tokenize=True,
            add_generation_prompt=True,
            enable_thinking=thinking,
        )
    prompt = data.get("prompt")
    if isinstance(prompt, str):
        return tokenizer.encode(prompt)
    # Fast tokenizers may construct their full vocabulary in __len__.
    # Query it once per request rather than once for each of 100K tokens.
    vocab_size = len(tokenizer)
    if (
        isinstance(prompt, list)
        and prompt
        and all(type(x) is int and 0 <= x < vocab_size for x in prompt)
    ):
        return prompt
    raise APIError("prompt must be text or one token-ID list", param="prompt")


def make_app(llm, instance, config):
    from swiftcache.engine.request import Request as GenerationRequest

    task = None
    completed_metrics = OrderedDict()

    def remember(rid, req):
        completed_metrics[rid] = req.metrics
        while len(completed_metrics) > 32:
            completed_metrics.popitem(last=False)

    @asynccontextmanager
    async def lifespan(app):
        nonlocal task
        task = asyncio.create_task(llm._main_event_loop())

        def failed(future):
            if not future.cancelled() and future.exception() is not None:
                import signal, traceback

                traceback.print_exception(future.exception())
                os.kill(os.getpid(), signal.SIGTERM)

        task.add_done_callback(failed)
        yield
        task.cancel()
        with suppress(asyncio.CancelledError):
            await task

    app = FastAPI(lifespan=lifespan)
    configure_errors(app)

    def check_engine():
        if task is not None and task.done():
            raise APIError(
                "Inference loop stopped: " + str(task.exception()),
                503,
                code="service_unavailable",
            )

    @app.get("/health")
    @app.get("/ready")
    async def health():
        check_engine()
        return dict(
            status="ready",
            model=instance.name,
            role=instance.role,
            rank=llm.config.rank,
        )

    @app.get("/cache")
    async def cache(request: Request):
        authenticate(request, config)
        check_engine()
        state = llm.scheduler.block_manager.snapshot()
        state.update(
            request_records=len(llm.request_dict),
            scheduled_requests=len(llm.seqs_in_scheduler),
            context_limit=llm.config.max_model_len,
            original_context_limit=llm.config.original_model_max_length,
            ignore_model_max_length=llm.config.ignore_model_max_length,
            regular_cache_blocks=llm.config.local_num_blocks,
        )
        state["maximum_usable_blocks"] = llm.scheduler.block_manager.maximum_blocks()
        return state

    @app.get("/request-metrics/{rid}")
    async def request_metrics(rid: str, request: Request):
        authenticate(request, config)
        if rid not in completed_metrics:
            raise APIError("Request metrics not found", 404)
        return completed_metrics[rid]

    async def complete(data, request, chat):
        authenticate(request, config)
        check_engine()
        if not isinstance(data, dict):
            raise APIError("Request body must be a JSON object")
        if data.get("model") != instance.name:
            raise APIError("Unknown model", param="model", code="model_not_found")
        maximum, temperature, top_p, stops = validate_options(data, chat)
        started = time.perf_counter_ns()
        ids = tokenize_request(llm.tokenizer, data, chat, config.enable_thinking)
        if not ids or len(ids) + maximum > llm.config.max_model_len:
            raise APIError(
                "Input and output exceed runtime context limit",
                param="messages" if chat else "prompt",
                code="context_length_exceeded",
            )
        if (
            len(ids) + maximum - 1
            > llm.scheduler.block_manager.maximum_blocks()
            * llm.config.kvcache_block_size
        ):
            raise APIError(
                "Request exceeds physical KV capacity", code="capacity_exceeded"
            )
        req = GenerationRequest(None, temperature, maximum)
        req.token_ids = ids
        req.top_p = top_p
        req.stop = stops
        req.streaming = data.get("stream", False)
        req.metrics["api_preprocess_ms"] = (time.perf_counter_ns() - started) / 1e6
        rid = ("chatcmpl-" if chat else "cmpl-") + uuid.uuid4().hex
        created = int(time.time())

        def usage():
            count = len(req.output_token_ids)
            return dict(
                prompt_tokens=len(ids),
                completion_tokens=count,
                total_tokens=len(ids) + count,
                prompt_tokens_details=dict(cached_tokens=max(0, req.prefix_cached)),
            )

        def chunk(delta=None, finish=None, final_usage=False):
            choice = dict(index=0, finish_reason=finish)
            choice.update(delta=delta or {}) if chat else choice.update(
                text=delta or "", logprobs=None
            )
            return dict(
                id=rid,
                object="chat.completion.chunk" if chat else "text_completion",
                created=created,
                model=instance.name,
                choices=[] if final_usage else [choice],
                **({"usage": usage()} if final_usage else {}),
            )

        if data.get("stream", False):

            async def generate_stream():
                producer = asyncio.create_task(llm.process_request(req))
                emitted = ""
                try:
                    if chat:
                        yield (
                            "data: "
                            + json.dumps(chunk({"role": "assistant", "content": ""}))
                            + "\n\n"
                        )
                    while True:
                        if await request.is_disconnected():
                            break
                        try:
                            event = await asyncio.wait_for(req.stream_queue.get(), 0.1)
                        except asyncio.TimeoutError:
                            if producer.done():
                                producer.result()
                                break
                            continue
                        text = event["text"]
                        if not text.startswith(emitted):
                            raise RuntimeError("Nonmonotonic incremental decoding")
                        delta = text[len(emitted) :]
                        emitted = text
                        if delta:
                            yield (
                                "data: "
                                + json.dumps(
                                    chunk({"content": delta} if chat else delta),
                                    ensure_ascii=False,
                                )
                                + "\n\n"
                            )
                        if event["finished"]:
                            await producer
                            remember(rid, req)
                            yield (
                                "data: "
                                + json.dumps(chunk(finish=req.finish_reason))
                                + "\n\n"
                            )
                            if (data.get("stream_options") or {}).get("include_usage"):
                                yield (
                                    "data: "
                                    + json.dumps(chunk(final_usage=True))
                                    + "\n\n"
                                )
                            yield "data: [DONE]\n\n"
                            break
                finally:
                    if not producer.done():
                        producer.cancel()
                        llm.cancel_request(req.request_id)
                    with suppress(asyncio.CancelledError):
                        await producer

            return StreamingResponse(
                generate_stream(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": rid,
                },
            )
        producer = asyncio.create_task(llm.process_request(req))
        try:
            while not producer.done():
                await asyncio.wait([producer], timeout=0.1)
                if await request.is_disconnected():
                    raise asyncio.CancelledError()
            await producer
        finally:
            if not producer.done():
                producer.cancel()
                llm.cancel_request(req.request_id)
            with suppress(asyncio.CancelledError):
                await producer
        choice = dict(index=0, finish_reason=req.finish_reason, logprobs=None)
        remember(rid, req)
        choice.update(
            message=dict(role="assistant", content=req.output)
        ) if chat else choice.update(text=req.output)
        return JSONResponse(
            dict(
                id=rid,
                object="chat.completion" if chat else "text_completion",
                created=created,
                model=instance.name,
                choices=[choice],
                usage=usage(),
            ),
            headers={"X-Request-ID": rid},
        )

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await complete(await request.json(), request, True)

    @app.post("/v1/completions")
    async def completions(request: Request):
        return await complete(await request.json(), request, False)

    @app.get("/v1/models")
    async def models(request: Request):
        authenticate(request, config)
        return dict(
            object="list",
            data=[
                dict(id=instance.name, object="model", created=0, owned_by="swiftcache")
            ],
        )

    @app.post("/generate")
    async def legacy(request: Request):
        authenticate(request, config)
        check_engine()
        data = await request.json()
        req = GenerationRequest(
            data.get("prompt"),
            float(data.get("temperature", 0)),
            int(data.get("max_tokens", 16)),
            bool(data.get("ignore_eos", False)),
        )
        req.token_ids = data.get("token_ids")
        req.top_p = float(data.get("top_p", 1))
        req.stop = data.get("stop", [])
        await llm.process_request(req)
        return dict(
            output=req.output,
            token_ids=req.output_token_ids,
            prompt_tokens=len(req.token_ids),
            cached_tokens=req.prefix_cached,
            time_usage=req.time_usage,
            metrics=req.metrics,
        )

    @app.post("/profiling")
    async def profiling(request: Request):
        authenticate(request, config)
        data = await request.json()
        if type(data.get("enabled")) is not bool:
            raise APIError("enabled must be boolean")
        llm.config.profile_stages = data["enabled"]
        return dict(enabled=llm.config.profile_stages)

    return app
