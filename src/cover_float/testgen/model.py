import logging
from pathlib import Path
from typing import Callable, TextIO

import cover_float.common.log as log
from cover_float.scripts.parse_testvectors import auto_parse

GLOBAL_MODELS: dict[str, Callable[[Path, log.StatusReporter], None]] = {}


def register_model(
    model_name: str,
) -> Callable[[Callable[[TextIO, TextIO], None]], Callable[[Path, log.StatusReporter], None]]:
    def inner(fn: Callable[[TextIO, TextIO], None]) -> Callable[[Path, log.StatusReporter], None]:
        def wrapper(output_dir: Path, status_reporter: log.StatusReporter, post_process: bool = True) -> None:
            tv_path = output_dir / "testvectors" / f"{model_name}_tv.txt"
            cv_path = output_dir / "covervectors" / f"{model_name}_cv.txt"

            status_reporter.start_model(model_name)

            with tv_path.open("w") as test_f, cv_path.open("w") as cover_f:
                try:
                    fn(test_f, cover_f)
                except Exception as e:
                    logger = logging.getLogger(model_name)

                    logger.exception(f"[bold red]Fatal Error in {model_name}[/] ", exc_info=e, extra={"markup": True})

            if post_process:
                auto_parse(model_name, str(output_dir))

            status_reporter.stop_model(model_name)

        GLOBAL_MODELS[model_name] = wrapper
        return wrapper

    return inner
