import os.path
import subprocess
import time

import pytest
import requests


@pytest.fixture()
def weaver_binary(request):
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
            request.param,
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
