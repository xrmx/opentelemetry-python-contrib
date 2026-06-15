# Copyright The OpenTelemetry Authors
# SPDX-License-Identifier: Apache-2.0
import os
import os.path
import shutil
import subprocess
import sys
import time
from pathlib import Path

import requests

from opentelemetry.semconv.metrics import MetricInstruments
from opentelemetry.semconv.metrics.http_metrics import (
    HTTP_SERVER_REQUEST_DURATION,
)
from opentelemetry.test.weaver_live_check import WeaverLiveCheck

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils import _is_known_auto_instrumentation_violation, _wait_for_port

METRIC_EXPORT_INTERVAL_MILLIS = 1000
METRIC_EXPORT_INTERVAL_SECONDS = METRIC_EXPORT_INTERVAL_MILLIS / 1000
METRIC_EXPORT_GRACE_SECONDS = 0.2
FASTAPI_STARTUP_TIMEOUT_SECONDS = 3


class FastAPIApplication:
    def __init__(self, otlp_endpoint: str, port: int = 35000):
        self.port = port
        self.otlp_endpoint = otlp_endpoint
        wrapper = shutil.which("opentelemetry-instrument")

        opentelemetry_instrumentation = [
            wrapper,
        ]

        self.application_path = os.path.join(os.path.dirname(__file__))
        self.command = opentelemetry_instrumentation + [
            "fastapi",
            "run",
            "main.py",
            "--port",
            f"{self.port}",
        ]

    def __enter__(self):
        self.handler = subprocess.Popen(
            self.command,
            cwd=self.application_path,
            env={
                **os.environ,
                "OTEL_EXPORTER_OTLP_ENDPOINT": self.otlp_endpoint,
                "OTEL_PYTHON_DISABLED_INSTRUMENTATIONS": "system_metrics",
                "OTEL_SEMCONV_STABILITY_OPT_IN": "http",  # use stable http semconv
                "OTEL_METRIC_EXPORT_INTERVAL": str(
                    METRIC_EXPORT_INTERVAL_MILLIS
                ),
            },
        )

        try:
            _wait_for_port(
                "127.0.0.1", self.port, FASTAPI_STARTUP_TIMEOUT_SECONDS
            )
        except TimeoutError:
            self.handler.terminate()
            self.handler.wait(timeout=5)
            raise

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        time.sleep(
            METRIC_EXPORT_INTERVAL_SECONDS + METRIC_EXPORT_GRACE_SECONDS
        )

        self.handler.terminate()
        self.handler.wait(timeout=5)


def test_fastapi_request():
    with WeaverLiveCheck() as weaver:
        with FastAPIApplication(
            otlp_endpoint=weaver.otlp_endpoint
        ) as fastapi_app:
            response = requests.get(f"http://127.0.0.1:{fastapi_app.port}/")
            assert response.status_code == 200

            response = requests.get(
                f"http://127.0.0.1:{fastapi_app.port}/notfound"
            )
            assert response.status_code == 404

        report = weaver.end()

    assert [
        violation
        for violation in report.violations
        if not _is_known_auto_instrumentation_violation(violation)
    ] == []
    seen_metrics = report["statistics"]["seen_registry_metrics"]
    assert seen_metrics.get(HTTP_SERVER_REQUEST_DURATION, 0) > 0
    assert (
        seen_metrics.get(MetricInstruments.HTTP_SERVER_ACTIVE_REQUESTS, 0) > 0
    )
    seen_attributes = report["statistics"]["seen_registry_attributes"]
    assert sum(seen_attributes.values()) > 0
