"""Single public OpenAI endpoint, routing served model names to local instances."""

from contextlib import asynccontextmanager
import asyncio
import json
import httpx
import anyio
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, StreamingResponse
from server.openai_app import APIError, authenticate, configure_errors


def make_gateway(config):
    client = None
    routes = {
        m.name: f"http://127.0.0.1:{config.internal_base_port + i}"
        for i, m in enumerate(config.models)
    }

    @asynccontextmanager
    async def lifespan(app):
        nonlocal client
        client = httpx.AsyncClient(timeout=None, trust_env=False)
        yield
        await client.aclose()

    app = FastAPI(lifespan=lifespan)
    configure_errors(app)

    @app.get("/health")
    @app.get("/ready")
    async def ready():
        async def check(name, url):
            try:
                r = await client.get(url + "/health", timeout=30)
                return name, r.status_code == 200
            except httpx.HTTPError:
                return name, False

        states = dict(await asyncio.gather(*(check(k, v) for k, v in routes.items())))
        return JSONResponse(
            dict(status="ready" if all(states.values()) else "starting", models=states),
            status_code=200 if all(states.values()) else 503,
        )

    @app.get("/v1/models")
    async def models(request: Request):
        authenticate(request, config)
        return dict(
            object="list",
            data=[
                dict(id=m.name, object="model", created=0, owned_by="swiftcache")
                for m in config.models
            ],
        )

    @app.get("/v1/models/{name}")
    async def model(name: str, request: Request):
        authenticate(request, config)
        if name not in routes:
            raise APIError("Unknown model", 404, param="model", code="model_not_found")
        return dict(id=name, object="model", created=0, owned_by="swiftcache")

    @app.get("/cache")
    async def cache(request: Request, model: str | None = None):
        authenticate(request, config)
        headers = {"Authorization": request.headers.get("Authorization", "")}
        if model:
            if model not in routes:
                raise APIError("Unknown model", param="model")
            response = await client.get(routes[model] + "/cache", headers=headers)
            return JSONResponse(response.json(), status_code=response.status_code)
        result = {}
        for name, url in routes.items():
            response = await client.get(url + "/cache", headers=headers)
            if response.is_error:
                return JSONResponse(response.json(), status_code=response.status_code)
            result[name] = response.json()
        return result

    async def proxy(request: Request):
        authenticate(request, config)
        data = await request.json()
        if not isinstance(data, dict):
            raise APIError("Request body must be a JSON object")
        name = data.get("model")
        if name is None and request.url.path == "/generate":
            name = config.models[0].name
        if not isinstance(name, str) or name not in routes:
            raise APIError("Unknown model", 404, param="model", code="model_not_found")
        headers = {
            "Authorization": request.headers.get("Authorization", ""),
            "Content-Type": "application/json",
        }
        url = routes[name] + request.url.path
        if data.get("stream", False):
            response = await client.send(
                client.build_request("POST", url, json=data, headers=headers),
                stream=True,
            )
            if response.is_error:
                body = await response.aread()
                await response.aclose()
                return JSONResponse(json.loads(body), status_code=response.status_code)

            async def forward():
                try:
                    async for chunk in response.aiter_bytes():
                        if await request.is_disconnected():
                            break
                        yield chunk
                finally:
                    with anyio.CancelScope(shield=True):
                        await response.aclose()

            return StreamingResponse(
                forward(),
                media_type="text/event-stream",
                headers={
                    "Cache-Control": "no-cache",
                    "X-Accel-Buffering": "no",
                    "X-Request-ID": response.headers.get("X-Request-ID", ""),
                },
            )
        producer = asyncio.create_task(client.post(url, json=data, headers=headers))
        try:
            while not producer.done():
                await asyncio.wait([producer], timeout=0.1)
                if await request.is_disconnected():
                    raise asyncio.CancelledError()
            response = await producer
            return JSONResponse(
                response.json(),
                status_code=response.status_code,
                headers={"X-Request-ID": response.headers.get("X-Request-ID", "")},
            )
        finally:
            if not producer.done():
                producer.cancel()

    app.add_api_route("/v1/chat/completions", proxy, methods=["POST"])
    app.add_api_route("/v1/completions", proxy, methods=["POST"])
    app.add_api_route("/generate", proxy, methods=["POST"])
    return app
