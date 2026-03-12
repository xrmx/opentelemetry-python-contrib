import json
import os
import os.path
import shutil
import subprocess
import time

import pytest
import requests

FLASK_PORT = 5000

DIR = os.path.join(os.path.dirname(__file__))


@pytest.fixture
def flask_fixture():
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


@pytest.mark.parametrize("weaver_binary", [DIR], indirect=True)
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
        metric_attributes_violations = [
            (advice["signal_name"], advice["message"])
            for data_point in report["metric"]["data_points"]
            for attribute in data_point["attributes"]
            for advice in attribute["live_check_result"]["all_advice"]
            if advice["level"] == "violation"
        ]

        assert metric_attributes_violations == []
