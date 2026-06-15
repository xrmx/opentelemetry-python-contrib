# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
import socket
import time


def _wait_for_port(host: str, port: int, timeout: float) -> None:
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.2)
            if sock.connect_ex((host, port)) == 0:
                return
        time.sleep(0.1)

    raise TimeoutError(f"Timed out waiting for {host}:{port}")


def _is_known_auto_instrumentation_violation(violation: dict) -> bool:
    context = violation.get("context") or {}

    return (
        violation.get("id") == "missing_attribute"
        and violation.get("signal_type") == "resource"
        and (context.get("attribute_key") or context.get("attribute_name"))
        == "telemetry.auto.version"
    )
