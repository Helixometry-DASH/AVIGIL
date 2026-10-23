"""Name -> constructor registry so configs can select a model by string."""

from __future__ import annotations

from collections.abc import Callable

_REGISTRY: dict[str, Callable] = {}


def register(name: str) -> Callable:
    def wrap(fn: Callable) -> Callable:
        if name in _REGISTRY:
            raise ValueError(f"model {name!r} already registered")
        _REGISTRY[name] = fn
        return fn

    return wrap


def build(name: str, **kwargs):
    if name not in _REGISTRY:
        raise KeyError(f"unknown model {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name](**kwargs)


def available() -> list[str]:
    return sorted(_REGISTRY)
