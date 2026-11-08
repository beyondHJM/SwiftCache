"""SwiftCache public API (model/CUDA initialization is lazy)."""
from swiftcache.sampling_params import SamplingParams

def __getattr__(name):
    if name == "LLM":
        from swiftcache.llm import LLM
        return LLM
    raise AttributeError(name)
