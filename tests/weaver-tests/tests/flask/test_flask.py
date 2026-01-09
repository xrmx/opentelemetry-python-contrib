import json
import os
import os.path
import shutil
import subprocess
import time

import pytest
import requests
from weaver_container import WeaverContainer

FLASK_PORT = 5000

DIR = os.path.join(os.path.dirname(__file__))


@pytest.fixture(scope="function")
def weaver_container_v1_36():
    weaver = WeaverContainer(
        schema_version="1.36.0",
        templates_dir=os.path.join(DIR, "../templates"),
    )
    yield weaver.start(timeout=20)
    weaver.stop()


@pytest.fixture()
def weaver_binary():
    # FIXME: don't hardcode weaver bin path
    weaver_bin = "/home/rm/src/weaver/./target/release/weaver"
    application_path = os.path.join(os.path.dirname(__file__))

    weaver = subprocess.Popen(
        [
            weaver_bin,
            "registry",
            "live-check",
            "--inactivity-timeout=10",  # FIXME: don't hardcode timeout
            "--format=json",
            "--no-stats",
            "--output",
            DIR,
        ],
        cwd=application_path,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    time.sleep(2)

    ready = False
    for i in range(10):
        try:
            response = requests.post("http://localhost:4320", timeout=5)
        except Exception:
            continue
        if response.status_code == 404:
            ready = True
            break
        time.sleep(0.5)

    if not ready:
        weaver.terminate()
        weaver.wait(timeout=5)
        raise Exception("sad trombone")

    yield weaver


@pytest.fixture
def flask_fixture():  # weaver_container_v1_36):
    # otlp_endpoint = weaver_container_v1_36.get_otlp_endpoint()
    wrapper = shutil.which("opentelemetry-instrument")
    opentelemetry_instrumentation = [
        wrapper,
        # FIXME: looks like there's a race condition somewhere where metrics override the span key in the result :(
        "--metric_export_interval",
        "4000",
        # "--exporter_otlp_endpoint",
        # otlp_endpoint,
    ]

    application_path = os.path.join(os.path.dirname(__file__))
    handler = subprocess.Popen(
        opentelemetry_instrumentation
        + ["flask", "run", "-p", f"{FLASK_PORT}"],
        cwd=application_path,
        env={
            **os.environ,
            "OTEL_SEMCONV_STABILITY_OPT_IN": "http",  # comment to make tests fail
        },  # use stable http semconv
    )

    time.sleep(1)

    yield

    time.sleep(1)

    handler.terminate()
    handler.wait(timeout=5)


def test_flask_request(weaver_binary, flask_fixture):
    response = requests.get(f"http://127.0.0.1:{FLASK_PORT}/rolldice")
    assert response.status_code == 200

    response = requests.get(f"http://127.0.0.1:{FLASK_PORT}/")
    assert response.status_code == 404

    # 5 seconds are needed to avoid grpc errors
    time.sleep(5)

    # stop weaver
    requests.post("http://localhost:4320/stop")

    outs, errs = weaver_binary.communicate(timeout=5)

    report_path = os.path.join(DIR, "live_check.json")
    with open(report_path, "r") as f:
        report_content = f.read()
        report = json.loads(report_content)

    assert report

    if "span" in report:
        span_attributes_violations = [
            (advice["signal_name"], advice["message"])
            for attribute in report["span"]["attributes"]
            for advice in attribute["live_check_result"]["all_advice"]
            if advice["level"] == "violation"
        ]

        assert span_attributes_violations == []

    if "metric" in report:
        span_attributes_violations = [
            (advice["signal_name"], advice["message"])
            for data_point in report["metric"]["data_points"]
            for attribute in data_point["attributes"]
            for advice in attribute["live_check_result"]["all_advice"]
            if advice["level"] == "violation"
        ]

        assert span_attributes_violations == []
