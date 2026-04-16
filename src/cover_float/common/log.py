from __future__ import annotations

import logging
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from types import TracebackType
from typing import Any, Callable, TypeVar

from rich import console
from rich.logging import RichHandler
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    ProgressColumn,
    SpinnerColumn,
    Task,
    TaskID,
    TextColumn,
)

STATUS_LEVEL_NUM = 25
logging.addLevelName(STATUS_LEVEL_NUM, "STATUS")


class ModelLogger(logging.Logger):
    status_reporter: StatusReporter

    def __init__(self, name: str) -> None:
        super().__init__(name)

    def status(self, message: str, *args: Any, **kwargs: Any) -> None:  # noqa: ANN401
        if self.isEnabledFor(STATUS_LEVEL_NUM):
            self._log(STATUS_LEVEL_NUM, message, args, **kwargs)


logging.setLoggerClass(ModelLogger)


class ExcludeStatusFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno != STATUS_LEVEL_NUM


class OnlyStatusFilter(logging.Filter):
    def filter(self, record: logging.LogRecord) -> bool:
        return record.levelno == 25


T = TypeVar("T")


class BarHandle:
    def __init__(self, progress: Progress, task_id: TaskID) -> None:
        self._progress = progress
        self._task_id = task_id

    # Match Progress.advance exactly
    def advance(self, advance: float = 1) -> None:
        self._progress.advance(self._task_id, advance)

    # Match Progress.update exactly — forward all kwargs through
    def update(
        self,
        *,
        total: float | None = None,
        completed: float | None = None,
        advance: float | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
        **fields: Any,  # noqa: ANN401
    ) -> None:
        self._progress.update(
            self._task_id,
            total=total,
            completed=completed,
            advance=advance,
            description=description,
            visible=visible,
            refresh=refresh,
            **fields,
        )

    def track(
        self,
        sequence: Iterable[T],
        *,
        total: float | None = None,
        completed: int = 0,
        description: str = "",
        update_period: float = 0.1,
    ) -> Iterable[T]:
        return self._progress.track(
            sequence,
            task_id=self._task_id,
            description=description,
            total=total,
            completed=completed,
            update_period=update_period,
        )


class OptionalColumn(ProgressColumn):
    def __init__(self, condition: Callable[[Task], bool], real_col: ProgressColumn) -> None:
        super().__init__()
        self._real_col = real_col
        self._condition = condition

    def render(self, task: Task) -> console.RenderableType:
        if not self._condition(task):
            return ""
        return self._real_col.render(task)


class StatusReporter:
    class _LoggingHandler(logging.Handler):
        def __init__(self, reporter: StatusReporter, model_name: str) -> None:
            super().__init__()
            self.reporter = reporter
            self.model_name = model_name

        def emit(self, record: logging.LogRecord) -> None:
            self.reporter.status_update(self.model_name, record.msg)

    def __init__(self) -> None:
        self.active_status_bars: dict[str, TaskID] = {}
        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold]{task.description}[/]"),
            TextColumn("[yellow]{task.fields[status]}[/]"),
            OptionalColumn(lambda t: t.total is not None, BarColumn()),
            OptionalColumn(lambda t: t.fields.get("m_of_n", False), MofNCompleteColumn()),
            OptionalColumn(lambda t: t.total is not None, TextColumn("{task.percentage:>3.0f}%")),
        )

    def __enter__(self) -> StatusReporter:
        self.progress.start()
        return self

    def __exit__(self, exc_type: type | None, exc_value: Exception | None, traceback: TracebackType | None) -> None:
        self.progress.stop()

    def start_model(self, model_name: str) -> None:
        task_id = self.progress.add_task(f"{model_name}: ", status="Test Generation", total=None)
        self.active_status_bars[model_name] = task_id
        model_logger = logging.getLogger(model_name)

        if isinstance(model_logger, ModelLogger):
            model_logger.status_reporter = self

        model_logger.handlers = []
        model_logger.propagate = False

        # Handle Status Updates
        handler = self._LoggingHandler(self, model_name)
        handler.addFilter(OnlyStatusFilter())
        model_logger.addHandler(handler)

        # Handle Other Updates
        general_handler = RichHandler()
        general_handler.addFilter(ExcludeStatusFilter())
        model_logger.addHandler(general_handler)

    def stop_model(self, model_name: str) -> None:
        self.progress.remove_task(self.active_status_bars[model_name])
        self.active_status_bars.pop(model_name)

    def status_update(self, model_name: str, status_message: str) -> None:
        task = self.active_status_bars[model_name]
        self.progress.update(task, status=status_message)

    @contextmanager
    def progress_bar(
        self, model_name: str, status: str | None = None, *, total: float | None = 100, show_m_of_n: bool = False
    ) -> Generator[BarHandle, None, None]:
        task_id = self.active_status_bars[model_name]

        if status is not None:
            self.progress.update(task_id, total=total, completed=0, status=status, m_of_n=show_m_of_n)
        else:
            self.progress.update(task_id, total=total, completed=0, m_of_n=show_m_of_n)

        try:
            yield BarHandle(self.progress, task_id)
        finally:
            self.progress.update(task_id, total=None, completed=0, m_of_n=False)
