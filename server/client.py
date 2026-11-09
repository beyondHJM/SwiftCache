"""Official OpenAI SDK client; no SwiftCache-specific request objects."""

import argparse
import os
from openai import OpenAI


def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--base-url",
        default=os.environ.get("OPENAI_BASE_URL", "http://127.0.0.1:18000/v1"),
    )
    p.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "local"))
    p.add_argument("--model", default="qwen3-8b-master")
    p.add_argument("--prompt", default="What is 2 + 2? Answer briefly.")
    p.add_argument("--max-tokens", type=int, default=64)
    p.add_argument("--stream", action="store_true")
    args = p.parse_args()
    client = OpenAI(base_url=args.base_url, api_key=args.api_key)
    result = client.chat.completions.create(
        model=args.model,
        messages=[dict(role="user", content=args.prompt)],
        max_completion_tokens=args.max_tokens,
        temperature=0,
        stream=args.stream,
    )
    if args.stream:
        for chunk in result:
            if chunk.choices and chunk.choices[0].delta.content:
                print(chunk.choices[0].delta.content, end="", flush=True)
        print()
    else:
        print(result.choices[0].message.content)


if __name__ == "__main__":
    main()
