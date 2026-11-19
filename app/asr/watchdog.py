import threading
import time
from typing import Callable


class WatchdogTimeout(RuntimeError):
    pass


def run_with_watchdog(
    target: Callable[[Callable[[], None]], object],
    timeout_seconds: float,
    poll_interval: float = 0.2,
):
    state = {"beat": time.monotonic(), "result": None, "error": None, "done": False}
    lock = threading.Lock()

    def heartbeat() -> None:
        with lock:
            state["beat"] = time.monotonic()

    def worker() -> None:
        try:
            state["result"] = target(heartbeat)
        except BaseException as exc:
            state["error"] = exc
        finally:
            state["done"] = True

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    while True:
        thread.join(poll_interval)
        if state["done"]:
            if state["error"] is not None:
                raise state["error"]
            return state["result"]
        with lock:
            beat = state["beat"]
        if time.monotonic() - beat > timeout_seconds:
            raise WatchdogTimeout(
                f"超过 {timeout_seconds:g}s 无进度，看门狗已中止该引擎任务"
            )