import time

import pytest

from app.asr.watchdog import WatchdogTimeout, run_with_watchdog


def test_stall_triggers_timeout():
    def target(heartbeat):
        time.sleep(5)
        return "late"

    start = time.monotonic()
    with pytest.raises(WatchdogTimeout, match="无进度"):
        run_with_watchdog(target, 0.5, poll_interval=0.05)
    assert time.monotonic() - start < 3


def test_heartbeat_keeps_task_alive():
    def target(heartbeat):
        for _ in range(6):
            time.sleep(0.1)
            heartbeat()
        return "ok"

    assert run_with_watchdog(target, 0.4, poll_interval=0.05) == "ok"


def test_worker_error_propagates():
    def target(heartbeat):
        raise ValueError("boom")

    with pytest.raises(ValueError, match="boom"):
        run_with_watchdog(target, 5)


def test_fast_result_returned():
    assert run_with_watchdog(lambda heartbeat: 42, 5) == 42