import os
from setuptools import setup

extensions = []
commands = {}
if os.environ.get("SWIFTCACHE_BUILD_CPP") == "1":
    from pybind11.setup_helpers import Pybind11Extension, build_ext

    extensions = [
        Pybind11Extension(
            "swiftcache.engine._native_scheduler",
            ["csrc/native_scheduler.cpp"],
            cxx_std=17,
            extra_compile_args=["-O3"],
        )
    ]
    commands = {"build_ext": build_ext}

setup(
    name="SwiftCache",
    version="0.0.1",
    author="Jianmin Hu",
    description="An Efficient LLM Serving for Multi-turn Conversations with Heterogeneous KV cache Sharing",
    packages=["swiftcache", "server"],
    zip_safe=False,
    ext_modules=extensions,
    cmdclass=commands,
)
