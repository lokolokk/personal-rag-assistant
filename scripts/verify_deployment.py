"""Build and verify disposable local Compose projects; zero provider dispatches.

Run from any directory: python scripts/verify_deployment.py [--keep]
Failures preserve resources and evidence. Success removes only this run's projects.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def request(url: str) -> tuple[int, str, bytes]:
    # Ignore workstation HTTP proxy settings for strictly localhost requests.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(url, timeout=10) as response:
            return response.status, response.headers.get("Content-Type", ""), response.read()
    except urllib.error.HTTPError as error:
        return error.code, error.headers.get("Content-Type", ""), error.read()


def verify_http(url: str) -> None:
    for endpoint, expected in (("/health", "ok"), ("/ready", "ready")):
        status, _, body = request(url + endpoint)
        require(status == 200 and json.loads(body)["status"] == expected, endpoint)
    status, _, index = request(url + "/")
    require(status == 200 and b'<div id="root"' in index, "SPA root unavailable")
    require(request(url + "/conversations/synthetic")[2] == index, "SPA deep link failed")
    assets = re.findall(rb'(?:src|href)="(/assets/[^"?]+)', index)
    require(bool(assets), "No frontend assets")
    for asset in assets:
        status, content_type, body = request(url + asset.decode())
        require(status == 200 and "html" not in content_type, "Asset fallback/error")
        require(b"http://backend" not in body, "Docker hostname leaked to browser")
    require(request(url + "/assets/missing.js")[0] == 404, "Missing asset fallback")
    for endpoint in ("/api/v1/documents", "/api/v1/conversations"):
        status, content_type, body = request(url + endpoint)
        require(status == 200 and "json" in content_type, "API proxy failed")
        require(isinstance(json.loads(body), list), "API response shape")
    status, content_type, _ = request(url + "/api/v1/documents/does-not-exist")
    require(status == 404 and "json" in content_type, "API error became SPA HTML")
    require(request(url + "/api/does-not-exist")[0] == 404, "Unknown API became SPA")


def verify_sse(url: str) -> dict[str, float]:
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    start = time.monotonic()
    with opener.open(url + "/api/v1/_verification/events", timeout=10) as response:
        require("text/event-stream" in response.headers["Content-Type"], "SSE content type")
        first = response.readline()
        first_at = time.monotonic() - start
        tail = response.read()
        finished_at = time.monotonic() - start
    require(first == b"event: token\n" and b"event: done" in tail, "SSE events missing")
    require(finished_at - first_at >= 1.5, "SSE buffered until completion")
    return {"first_event_seconds": first_at, "completion_seconds": finished_at}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--keep", action="store_true", help="Keep disposable projects for browser inspection"
    )
    args = parser.parse_args()
    project = "ragverify-" + uuid.uuid4().hex[:12]
    evidence = ROOT / "tmp" / project
    evidence.mkdir(parents=True)
    empty_env = evidence / "empty.env"
    empty_env.write_text("", encoding="utf-8")
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(("COMPOSE_", "OPENAI_", "RAG_")):
            del env[key]
    env["RAG_PORT"] = "0"
    report: dict[str, Any] = {"project": project, "status": "incomplete"}
    projects: dict[str, list[str]] = {}

    def run(*command: str, timeout: int = 120, stdin: str | None = None) -> str:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            input=stdin,
            capture_output=True,
            text=True,
            encoding="utf-8",
            timeout=timeout,
        )
        with (evidence / "commands.log").open("a", encoding="utf-8") as log:
            log.write(json.dumps(list(command)) + "\n" + result.stdout + result.stderr)
            log.write(f"\nexit={result.returncode}\n")
        require(
            result.returncode == 0,
            f"Command failed ({result.returncode}): {command[:4]}; see commands.log",
        )
        return result.stdout.strip()

    def compose(name: str, override: Path | None = None) -> list[str]:
        command = [
            "docker",
            "compose",
            "--project-directory",
            str(ROOT),
            "--env-file",
            str(empty_env),
            "-p",
            name,
            "-f",
            str(ROOT / "compose.yaml"),
        ]
        if override:
            command += ["-f", str(override)]
        projects[name] = command
        return command

    def start(command: list[str]) -> str:
        run(
            *command,
            "up",
            "-d",
            "--no-build",
            "--force-recreate",
            "--wait",
            "--wait-timeout",
            "120",
            timeout=180,
        )
        address = run(*command, "port", "frontend", "8080")
        require(address.startswith("127.0.0.1:"), "Port is not loopback-only")
        url = "http://" + address
        verify_http(url)
        return url

    try:
        require(
            shutil.which("docker") is not None,
            "Docker CLI unavailable; install/start Docker Desktop with Linux containers",
        )
        require(
            run("docker", "info", "--format", "{{.OSType}}") == "linux",
            "Linux Docker daemon required",
        )
        require(
            int(run("docker", "compose", "version", "--short").lstrip("v").split(".")[0]) >= 2,
            "Compose v2 or newer required",
        )
        command = compose(project)
        run(*command, "config", "--quiet")
        print(f"Building isolated project {project}; evidence: {evidence}", flush=True)
        run(*command, "build", timeout=1800)
        url = start(command)
        images = {}
        for service in ("backend", "frontend"):
            container = run(*command, "ps", "-q", service)
            details = json.loads(run("docker", "inspect", container))[0]
            images[service] = details["Image"]
            uid = int(run(*command, "exec", "-T", service, "id", "-u"))
            require(uid != 0, f"{service} runs as root")
            require(
                details["Config"]["Labels"]["com.docker.compose.project"] == project,
                "Ownership mismatch",
            )
            if service == "backend":
                require(
                    not any(details["NetworkSettings"]["Ports"].values()), "Backend port published"
                )
                for network in details["NetworkSettings"]["Networks"]:
                    require(
                        json.loads(run("docker", "network", "inspect", network))[0]["Internal"],
                        "Backend has egress network",
                    )
                volume = next(
                    m["Name"] for m in details["Mounts"] if m["Destination"] == "/app/storage"
                )
                require(volume == project + "_rag-storage", "Unexpected storage volume")
                require(
                    not any(
                        v
                        for v in details["Config"]["Env"]
                        if v.startswith("OPENAI_API_KEY=") and v != "OPENAI_API_KEY="
                    ),
                    "Provider key configured",
                )
                run(
                    *command,
                    "exec",
                    "-T",
                    "backend",
                    "python",
                    "-c",
                    "from pathlib import Path; p=Path('/app/storage/.write-check'); "
                    "p.write_text('ok'); assert p.read_text()=='ok'; p.unlink(); "
                    "assert not Path('/verification').exists(); "
                    "assert not Path('/app/backend/.env').exists(); "
                    "assert not Path('/app/backend/tests').exists()",
                )
                # Non-provider public IP: verify the isolated network cannot open TCP.
                run(
                    *command,
                    "exec",
                    "-T",
                    "backend",
                    "python",
                    "-c",
                    "import socket; s=socket.socket(); s.settimeout(2); "
                    "result=s.connect_ex(('1.1.1.1',443)); s.close(); "
                    "assert result != 0, 'External egress available'",
                )
        report["images"] = images
        run(*command, "stop", "backend")
        fixture = ROOT / "scripts/deployment/fixture.py"
        run(
            *command,
            "run",
            "--rm",
            "--no-deps",
            "-T",
            "backend",
            "python",
            "-",
            "seed",
            stdin=fixture.read_text(),
        )
        url = start(command)
        require(
            json.loads(request(url + "/api/v1/documents")[2])[0]["id"] == "synthetic",
            "Document lost on recreation",
        )
        require(
            json.loads(request(url + "/api/v1/conversations")[2])[0]["id"]
            == "synthetic-conversation",
            "Conversation lost",
        )
        override_data: dict[str, Any] = {
            "services": {
                "backend": {
                    "command": ["python", "/verification/fixture.py", "serve"],
                    "environment": {"PYTHONPATH": "/app/backend"},
                    "volumes": [
                        {
                            "type": "bind",
                            "source": str(fixture.parent),
                            "target": "/verification",
                            "read_only": True,
                        }
                    ],
                }
            }
        }
        override = evidence / "fixture.compose.json"
        override.write_text(json.dumps(override_data), encoding="utf-8")
        command = compose(project, override)
        url = start(command)
        status, _, body = request(url + "/api/v1/_verification/state")
        require(status == 200, "Vector/PDF/registry/BM25 recovery failed")
        original_state = json.loads(body)
        report["state"] = original_state
        report["sse"] = verify_sse(url)
        run(*command, "stop")
        # Quiescent whole-volume backup: no live Chroma/JSON writers.
        run(
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "10001:10001",
            "-v",
            f"{volume}:/source:ro",
            "-v",
            f"{evidence}:/backup",
            images["backend"],
            "tar",
            "-cf",
            "/backup/storage.tar",
            "-C",
            "/source",
            ".",
        )
        restore = project + "-restore"
        restore_volume = restore + "_rag-storage"
        run(
            "docker",
            "volume",
            "create",
            "--label",
            f"com.docker.compose.project={restore}",
            "--label",
            "com.docker.compose.volume=rag-storage",
            restore_volume,
        )
        run(
            "docker",
            "run",
            "--rm",
            "--network",
            "none",
            "--user",
            "0:0",
            "-v",
            f"{restore_volume}:/restore",
            "-v",
            f"{evidence}:/backup:ro",
            images["backend"],
            "sh",
            "-c",
            "tar -xf /backup/storage.tar -C /restore && chown -R 10001:10001 /restore",
        )
        override_data["services"]["backend"]["image"] = images["backend"]
        override_data["services"]["frontend"] = {"image": images["frontend"]}
        restore_override = evidence / "restore.compose.json"
        restore_override.write_text(json.dumps(override_data), encoding="utf-8")
        restore_command = compose(restore, restore_override)
        restored_url = start(restore_command)
        status, _, body = request(restored_url + "/api/v1/_verification/state")
        require(status == 200 and json.loads(body) == original_state, "Restored data differs")
        report["restore"] = "passed"
        report["restored_url"] = restored_url
        report["status"] = "passed"
        report["browser_acceptance"] = "Open restored_url with --keep; inspect browser separately"
        if not args.keep:
            for name, owned in projects.items():
                volumes = json.loads(run("docker", "volume", "inspect", name + "_rag-storage"))
                require(
                    volumes[0]["Labels"]["com.docker.compose.project"] == name,
                    "Refusing cleanup of unowned volume",
                )
                run(*owned, "down", "--volumes", "--remove-orphans")
        print(json.dumps(report, indent=2), flush=True)
        return 0
    except (
        OSError,
        RuntimeError,
        ValueError,
        KeyError,
        IndexError,
        subprocess.TimeoutExpired,
    ) as error:
        report["status"] = "failed"
        report["error"] = str(error)
        print(f"FAILED: {error}\nEvidence/resources preserved: {evidence}", file=sys.stderr)
        return 1
    finally:
        (evidence / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
