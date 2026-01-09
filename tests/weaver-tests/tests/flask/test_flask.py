import os.path
import shutil
import subprocess
import time

import pytest
import requests
from weaver_container import WeaverContainer

FLASK_PORT = 5000


@pytest.fixture(scope="function")
def weaver_container_v1_36():
    weaver = WeaverContainer(
        schema_version="1.36.0",
    )
    yield weaver.start(timeout=20)
    weaver.stop()


@pytest.fixture
def flask_fixture():  # weaver_container_v1_36):
    # otlp_endpoint = weaver_container_v1_36.get_otlp_endpoint()
    wrapper = shutil.which("opentelemetry-instrument")
    opentelemetry_instrumentation = [
        wrapper,
        "--metric_export_interval",
        "1000",
        # "--exporter_otlp_endpoint",
        # otlp_endpoint,
    ]
    application_path = os.path.join(os.path.dirname(__file__))
    handler = subprocess.Popen(
        opentelemetry_instrumentation
        + ["flask", "run", "-p", f"{FLASK_PORT}"],
        cwd=application_path,
    )

    time.sleep(1)

    yield

    time.sleep(1)

    handler.terminate()
    handler.wait(timeout=5)


def test_flask_request(flask_fixture):  # , weaver_container_v1_36):
    response = requests.get(f"http://127.0.0.1:{FLASK_PORT}/rolldice")
    assert response.status_code == 200

    response = requests.get(f"http://127.0.0.1:{FLASK_PORT}/")
    assert response.status_code == 404

    """
    full_report = weaver_container_v1_36.end_live_check()

    assert full_report
    """
