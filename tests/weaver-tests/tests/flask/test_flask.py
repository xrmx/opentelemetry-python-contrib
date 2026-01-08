import os
import subprocess

import pytest
import requests
from weaver_container import WeaverContainer

FLASK_PORT = "5000"


@pytest.fixture(scope="function")
def weaver_container_v1_36():
    weaver = WeaverContainer(
        schema_version="1.36.0",
    )
    yield weaver.start(timeout=20)
    weaver.stop()


@pytest.fixture
def flask_fixture(weaver_container_v1_36):
    otlp_endpoint = weaver_container_v1_36.get_otlp_endpoint()
    opentelemetry_instrumentation = [
        "opentelemetry-instrumentation",
        "--metric_export_interval",
        1,
        "--exporter_otlp_endpoint",
        otlp_endpoint,
    ]
    handler = subprocess.Popen(
        opentelemetry_instrumentation + ["flask", "run", "-p", FLASK_PORT]
    )

    os.sleep(1)

    yield

    handler.terminate()
    handler.wait(timeout=5)


def test_flask_request(flask_fixture, weaver_container_v1_36):
    response = requests.get(f"http://127.0.0.1:{FLASK_PORT}")

    assert response.status_code == 200

    full_report = weaver_container_v1_36.end_live_check()

    assert full_report
