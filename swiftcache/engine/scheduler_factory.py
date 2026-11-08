"""Select one scheduler without loading the obsolete pre-elastic C++ extension."""


def get_scheduler_class(backend):
    if backend == "python":
        from .scheduler import Scheduler
    elif backend == "cpp":
        from .native_scheduler import Scheduler
    else:
        raise ValueError(f"Unknown scheduler backend: {backend}")
    return Scheduler
