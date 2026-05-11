import os
import os.path
import shutil
import subprocess
import time

import requests

from opentelemetry.test.weaver_live_check import WeaverLiveCheck


class FlaskApplication:
    def __init__(self, port=35000):
        # FIXME: double check that metrics have enough time to be exported

        self.port = port
        wrapper = shutil.which("opentelemetry-instrument")

        opentelemetry_instrumentation = [
            wrapper,
        ]

        self.application_path = os.path.join(os.path.dirname(__file__))
        self.command = opentelemetry_instrumentation + [
            "flask",
            "run",
            "-p",
            f"{self.port}",
        ]

    def __enter__(self):
        self.handler = subprocess.Popen(
            self.command,
            cwd=self.application_path,
            env={
                **os.environ,
                "OTEL_SEMCONV_STABILITY_OPT_IN": "http",  # use stable http semconv
            },
        )

        time.sleep(1)

        return self

    def __exit__(self, exc_type, exc_value, exc_traceback):
        # FIXME: double check there's enough time for metrics to be exported
        # 1 seconds are needed to avoid grpc errors
        time.sleep(1)

        self.handler.terminate()
        self.handler.wait(timeout=5)


def test_flask_request():
    with WeaverLiveCheck() as weaver:
        with FlaskApplication() as flask_app:
            response = requests.get(
                f"http://127.0.0.1:{flask_app.port}/rolldice"
            )
            assert response.status_code == 200

            response = requests.get(f"http://127.0.0.1:{flask_app.port}/")
            assert response.status_code == 404

        report = weaver.end()

    assert report.violations == []
