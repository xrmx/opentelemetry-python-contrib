import json
import logging
import os
import shutil
import time
from typing import Any, Optional

from requests import post
from requests.exceptions import ConnectionError as ReqConnectionError
from testcontainers.core.container import DockerContainer

logger = logging.getLogger(__name__)


class WeaverContainer(DockerContainer):
    def __init__(
        self,
        schema_version: Optional[str] = None,
        report_dir: Optional[str] = None,
        weaver_version: str = "v0.20.0",
        policies_dir: Optional[str] = None,
        templates_dir: Optional[str] = None,
        inactivity_timeout: int = 30,
    ):
        try:
            image = f"otel/weaver:{weaver_version}"
            super().__init__(image)  # type: ignore
            self._ready = False
            self._stopped = False
            self._clean_report_dir = report_dir is None

            self.with_exposed_ports(4317, 4320)  # type: ignore
            self.with_bind_ports(4317, 4317)
            self.with_bind_ports(4320, 4320)
            self.with_name("weaver-live-check")

            command = f"registry live-check --inactivity-timeout={inactivity_timeout} --format json"

            if report_dir is not None:
                self._report_dir = report_dir
            else:
                self._report_dir = "./weaver-report"
                self._clean_report_dir = True

            self._report_dir = os.path.abspath(self._report_dir)
            os.makedirs(self._report_dir, exist_ok=True)

            command += " --output /weaver/report"
            self.with_volume_mapping(
                self._report_dir, "/weaver/report", mode="rw"
            )
            logger.debug("Mapped report directory: %s", self._report_dir)

            if policies_dir:
                policies_dir = os.path.abspath(policies_dir)
                command += " --advice-policies /weaver/policies"
                self.with_volume_mapping(
                    policies_dir, "/weaver/policies", mode="ro"
                )
                logger.debug("Mapped policies directory: %s", policies_dir)
            if templates_dir:
                templates_dir = os.path.abspath(templates_dir)
                command += " --templates /weaver/templates"
                self.with_volume_mapping(
                    templates_dir, "/weaver/templates", mode="ro"
                )
                logger.debug("Mapped templates directory: %s", templates_dir)

            if schema_version:
                command += f" --registry https://github.com/open-telemetry/semantic-conventions/archive/refs/tags/v{schema_version}.tar.gz[model]"

            self.with_command(command)
            logger.debug("Weaver command: %s", command)
        except Exception as e:
            logger.error("Error initializing WeaverContainer: %s", e)
            raise

    def start(self, timeout: int = 60) -> "WeaverContainer":
        # remove files from report dir before starting
        if os.path.exists(self._report_dir):
            logger.debug("Cleaning up report directory: %s", self._report_dir)
            shutil.rmtree(self._report_dir)

        logger.debug("Starting Weaver container...")
        super().start()
        try:
            self._wait_for_ready(timeout=timeout)
        except Exception as e:
            logger.error(
                "Error while waiting for Weaver container to be ready, %s", e
            )
            raise
        self._ready = True
        return self

    def get_otlp_endpoint(self) -> str:
        host = self.get_container_host_ip()
        port = self.get_exposed_port(4317)  # type: ignore
        return f"http://{host}:{port}"

    def _wait_for_ready(self, timeout: int = 60) -> None:
        for i in range(timeout):
            try:
                # can't get exposed port before container is fully started
                response = post("http://localhost:4320", timeout=5)
                if response.status_code == 404:
                    return
                logger.debug(
                    "Weaver live-check container not ready yet, status %s, try %s",
                    response.status_code,
                    i,
                )
            except ReqConnectionError as e:
                logger.debug("Health check exception: %s", e)
                pass
            time.sleep(1)
        raise TimeoutError(
            "Weaver live-check container did not become ready in time"
        )

    def end_live_check(self, timeout: int = 30) -> dict[str, Any]:
        if self._stopped:
            return {}
        self._stopped = True

        try:
            if self._ready:
                response = post("http://localhost:4320/stop", timeout=5)
                response.raise_for_status()
                logger.debug("Weaver live-check stopped successfully")
                result = self.get_wrapped_container().wait(timeout=timeout)
                exit_code = str(result["StatusCode"])
            else:
                exit_code = "container could not start"

            if exit_code == "0":
                return self._read_report()

            self._clean_report_dir = False  # keep report for debugging
            logs = self._read_weaver_logs()
            violations = self._read_violations()
            error_message = (
                f"violations: {violations}" if violations else f"logs: {logs}"
            )
            raise Exception(
                f"Exited with non-zero status: {exit_code}, {error_message}"
            )
        except Exception as e:
            self._clean_report_dir = False  # keep report for debugging
            logs = self._read_weaver_logs()
            logger.error(
                "Error during weaver live-check: %s, logs: %s", e, logs
            )
            raise

    def _read_violations(self) -> str:
        try:
            violations_path = os.path.join(self._report_dir, "violations.md")
            with open(violations_path, "r") as f:
                content = f.read().strip()
                logger.debug("Weaver violations report content: %s", content)
                return content
        except Exception as e:
            logger.error("Could not read violations report: %s", e)
            raise

    def _read_report(self) -> dict[str, Any]:
        try:
            report_path = os.path.join(self._report_dir, "full_report.json")
            with open(report_path, "r") as f:
                report_content = f.read()
                return json.loads(report_content)
        except Exception as e:
            logger.error("Error checking Weaver report: %s", e)
            raise

    def _read_weaver_logs(self) -> Optional[str]:
        try:
            (err, out) = self.get_logs()
            logs = f"{err.decode('utf-8')}\n{out.decode('utf-8')}"
            logger.debug("Weaver live-check logs: %s", logs)
            return logs
        except Exception as e:
            logger.error("Could not get weaver logs: %s", e)
            return None

    def stop(self, force: bool = True, delete_volume: bool = True) -> None:
        try:
            self.end_live_check()
        finally:
            super().stop(force, delete_volume)
            if self._clean_report_dir:
                try:
                    shutil.rmtree(self._report_dir)
                except Exception:
                    pass
