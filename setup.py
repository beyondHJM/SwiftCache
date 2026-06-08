from setuptools import setup

setup(
    name="SwiftCache",
    version="0.0.1",
    author="Jianmin Hu",
    description="An Efficient LLM Serving for Multi-turn Conversations with Heterogeneous KV cache Sharing",
    packages=["swiftcache","server"],
    zip_safe=False,
)