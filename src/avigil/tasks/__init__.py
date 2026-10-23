from .base import Task
from .loc_av import AVLocalizationTask
from .registry import available_tasks, build_task, register_task

__all__ = [
    "AVLocalizationTask",
    "Task",
    "available_tasks",
    "build_task",
    "register_task",
]
