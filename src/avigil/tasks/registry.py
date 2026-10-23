"""Task registry -- mirrors models.registry so configs select by string."""

from __future__ import annotations

from collections.abc import Callable

_TASKS: dict[str, Callable] = {}


def register_task(name: str) -> Callable:
    def wrap(fn: Callable) -> Callable:
        if name in _TASKS:
            raise ValueError(f"task {name!r} already registered")
        _TASKS[name] = fn
        return fn

    return wrap


def build_task(name: str, cfg):
    if name not in _TASKS:
        raise KeyError(f"unknown task {name!r}; registered: {sorted(_TASKS)}")
    return _TASKS[name](cfg)


def available_tasks() -> list[str]:
    return sorted(_TASKS)
