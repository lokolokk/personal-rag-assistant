"""Real local proxy, persistence, cold backup and fresh-volume restoration checks."""

import copy
import json
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    project = "raginvited-" + uuid.uuid4().hex[:12]
    evidence = ROOT / "tmp" / project
    evidence.mkdir()
    env = {
        k: v for k, v in os.environ.items() if not k.startswith(("LIVE_", "OPENAI_", "COMPOSE_"))
    }
    (evidence / "empty.env").write_text("", encoding="utf-8")

    def run(*command: str, record: bool = True) -> str:
        result = subprocess.run(
            command,
            cwd=ROOT,
            env=env,
            text=True,
            encoding="utf-8",
            capture_output=True,
            timeout=1800,
        )
        if record:
            with (evidence / "commands.log").open("a", encoding="utf-8") as log:
                log.write(
                    json.dumps(command)
                    + "\n"
                    + result.stdout
                    + result.stderr
                    + f"\nexit={result.returncode}\n"
                )
        if result.returncode:
            raise RuntimeError(f"Command failed: {command[:4]}; see evidence")
        return result.stdout.strip()

    def compose(name: str, override: Path) -> list[str]:
        return [
            "docker",
            "compose",
            "--project-directory",
            str(ROOT),
            "--env-file",
            str(evidence / "empty.env"),
            "-p",
            name,
            "-f",
            str(ROOT / "compose.live.yaml"),
            "-f",
            str(override),
        ]

    fixture = ROOT / "scripts/deployment"
    config: dict[str, Any] = {
        "services": {
            "backend": {
                "networks": ["private"],
                "environment": {"LIVE_PROVIDER_ENABLED": "false", "PYTHONPATH": "/app/backend"},
                "volumes": [f"{fixture}:/verification:ro", f"{evidence}:/verification-output"],
                "command": ["python", "/verification/invited_fixture.py", "serve"],
            },
            "proxy": {"ports": ["127.0.0.1:8182:8080"]},
        }
    }
    override = evidence / "fixture.compose.json"

    def write_override(path: Path, selected: dict[str, Any] | None = None) -> None:
        value = json.dumps(selected if selected is not None else config)
        value = value.replace('"networks": ["private"]', '"networks": !override ["private"]')
        value = value.replace(
            '"ports": ["127.0.0.1:8182:8080"]', '"ports": !override ["127.0.0.1:8182:8080"]'
        )
        path.write_text(value, encoding="utf-8")

    write_override(override)
    command = compose(project, override)
    run(*command, "config", "--quiet")
    print(f"Disposable invited runtime: {project}; evidence: {evidence}", flush=True)
    run(*command, "build", "backend")
    image = run(*command, "images", "-q", "backend")
    # Compose images returns empty before creation; resolve the task-specific build tag.
    if not image:
        image = run("docker", "image", "inspect", project + "-backend", "--format", "{{.Id}}")
    volume = project + "_live-storage"
    run(
        "docker",
        "volume",
        "create",
        "--label",
        f"com.docker.compose.project={project}",
        "--label",
        "com.docker.compose.volume=live-storage",
        volume,
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
        volume + ":/app/storage",
        image,
        "chown",
        "10001:10001",
        "/app/storage",
    )
    run(
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        "10001:10001",
        "-v",
        volume + ":/app/storage",
        "-v",
        f"{fixture}:/verification:ro",
        "-v",
        f"{evidence}:/verification-output",
        "-e",
        "PYTHONPATH=/app/backend",
        image,
        "python",
        "/verification/invited_fixture.py",
        "seed",
    )
    capabilities = json.loads((evidence / "capabilities.json").read_text(encoding="utf-8"))
    production = copy.deepcopy(config)
    del production["services"]["backend"]["command"]
    del production["services"]["backend"]["volumes"]
    production_override = evidence / "production.compose.json"
    write_override(production_override, production)
    production_command = compose(project, production_override)
    run(*production_command, "up", "-d", "--no-build", "--wait", "--wait-timeout", "120")
    production_check = (
        "import json,urllib.request,urllib.error; "
        "assert json.load(urllib.request.urlopen('http://127.0.0.1:8000/ready'))"
        "['status']=='ready'; "
        "from pathlib import Path; from app.services.access import AccessStore; "
        "from app.services.workspaces import WorkspaceManager; "
        "a=AccessStore(Path('/app/storage/live/control.sqlite3')); "
        "m=WorkspaceManager(Path('/app/storage/live'),a,None); "
        "assert all(m.get(w).vectors.count()==0 for w in a.workspace_ids())"
    )
    run(*production_command, "exec", "-T", "backend", "python", "-c", production_check)
    run(*production_command, "stop")
    run(*command, "up", "-d", "--no-build", "--wait", "--wait-timeout", "120")
    backend = run(*command, "ps", "-q", "backend")
    uid = run(*command, "exec", "-T", "backend", "id", "-u")
    if uid != "10001":
        raise RuntimeError("Backend not non-root")
    details = json.loads(run("docker", "inspect", backend))[0]
    if details["Config"]["Labels"]["com.docker.compose.project"] != project:
        raise RuntimeError("Project ownership mismatch")
    # No paid provider/egress permitted in this fixture.
    run(
        *command,
        "exec",
        "-T",
        "backend",
        "python",
        "-c",
        "import socket; s=socket.socket(); s.settimeout(2); "
        "assert s.connect_ex(('1.1.1.1',443)) != 0",
    )
    url = "http://127.0.0.1:8182"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(
        path: str,
        token: str | None = None,
        data: bytes | None = None,
        content_type: str = "application/json",
        method: str | None = None,
    ) -> tuple[int, bytes]:
        headers = {"Content-Type": content_type}
        if token:
            headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(url + path, data=data, headers=headers, method=method)
        try:
            with opener.open(req, timeout=180) as response:
                return response.status, response.read()
        except urllib.error.HTTPError as error:
            return error.code, error.read()

    assert request("/api/v1/documents")[0] == 401
    assert request("/api/v1/documents", capabilities["revoked"]["session"])[0] == 401
    pdf = (evidence / "synthetic.pdf").read_bytes()
    body = (
        b'--synthetic-boundary\r\nContent-Disposition: form-data; name="file"; '
        b'filename="same.pdf"\r\nContent-Type: application/pdf\r\n\r\n'
        + pdf
        + b"\r\n--synthetic-boundary--\r\n"
    )
    documents: dict[str, str] = {}
    for name in ("a", "b"):
        token = capabilities[name]["session"]
        status, result = request(
            "/api/v1/documents/upload",
            token,
            body,
            "multipart/form-data; boundary=synthetic-boundary",
        )
        assert status == 201, result.decode()
        documents[name] = json.loads(result)["id"]
    assert documents["a"] != documents["b"]
    assert request("/api/v1/documents/" + documents["a"], capabilities["b"]["session"])[0] == 404
    token = capabilities["a"]["session"]
    _, raw = request("/api/v1/conversations", token, b"{}")
    conversation = json.loads(raw)["conversationId"]
    req = urllib.request.Request(
        url + f"/api/v1/conversations/{conversation}/messages",
        data=json.dumps({"content": "quartz", "documentIds": []}).encode(),
        headers={"Content-Type": "application/json", "Authorization": "Bearer " + token},
    )
    started = time.monotonic()
    with opener.open(req, timeout=180) as response:
        assert "text/event-stream" in response.headers["Content-Type"]
        first = response.readline()
        first_at = time.monotonic() - started
        tail = response.read()
        finished = time.monotonic() - started
    assert first.startswith(b"event:") and finished - first_at >= 1.5
    assert b'"finishReason": "stop"' in tail and documents["b"].encode() not in tail
    status, history = request(f"/api/v1/conversations/{conversation}/messages", token)
    assert status == 200 and len(json.loads(history)) == 2
    delayed_request = urllib.request.Request(
        url + "/api/v1/_verification/delayed", headers={"Authorization": "Bearer " + token}
    )
    delayed_started = time.monotonic()
    with opener.open(delayed_request, timeout=10) as response:
        delayed_first = response.readline()
        delayed_first_at = time.monotonic() - delayed_started
        delayed_tail = response.read()
        delayed_finished = time.monotonic() - delayed_started
    assert delayed_first == b"event: token\n" and b"event: done" in delayed_tail
    assert delayed_first_at >= 1.8 and delayed_finished - delayed_first_at >= 1.8
    run(*command, "restart", "backend")
    run(
        *command,
        "up",
        "-d",
        "--no-build",
        "--force-recreate",
        "proxy",
        "--wait",
        "--wait-timeout",
        "120",
    )
    assert request("/api/v1/documents", token)[0] == 200
    run(*command, "stop")
    run(
        "docker",
        "run",
        "--rm",
        "--network",
        "none",
        "--user",
        "10001:10001",
        "-v",
        volume + ":/source:ro",
        "-v",
        f"{evidence}:/backup",
        image,
        "tar",
        "-cf",
        "/backup/storage.tar",
        "-C",
        "/source",
        ".",
    )
    restored = project + "-restore"
    restored_volume = restored + "_live-storage"
    run(
        "docker",
        "volume",
        "create",
        "--label",
        f"com.docker.compose.project={restored}",
        "--label",
        "com.docker.compose.volume=live-storage",
        restored_volume,
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
        restored_volume + ":/restore",
        "-v",
        f"{evidence}:/backup:ro",
        image,
        "sh",
        "-c",
        "tar -xf /backup/storage.tar -C /restore && chown -R 10001:10001 /restore",
    )
    config["services"]["backend"]["image"] = image
    restore_override = evidence / "restore.compose.json"
    write_override(restore_override)
    restored_command = compose(restored, restore_override)
    run(*restored_command, "up", "-d", "--no-build", "--wait", "--wait-timeout", "120")
    assert request("/api/v1/documents", token)[0] == 200
    assert request("/api/v1/documents/" + documents["a"], capabilities["b"]["session"])[0] == 404
    assert request("/api/v1/documents", capabilities["revoked"]["session"])[0] == 401
    assert request(f"/api/v1/conversations/{conversation}/messages", token)[1] == history
    run(
        *restored_command,
        "exec",
        "-T",
        "backend",
        "python",
        "-c",
        "import json; from pathlib import Path; from app.services.access import AccessStore; "
        "from app.services.workspaces import WorkspaceManager; "
        "from app.services.retrieval.vector_store import DeterministicEmbeddingFunction; "
        "r=Path('/app/storage/live'); a=AccessStore(r/'control.sqlite3'); "
        "m=WorkspaceManager(r,a,None); "
        "c=json.loads(Path('/verification-output/capabilities.json').read_text()); "
        "v=[float(x) for x in DeterministicEmbeddingFunction(1024)(['quartz'])[0]]; "
        "bundles=[m.get(c[n]['workspace']) for n in ('a','b')]; "
        "assert all(b.vectors.query('quartz',query_embedding=v) and "
        "b.bm25.query('quartz') for b in bundles); "
        "pdf=Path('/verification-output/synthetic.pdf').read_bytes(); "
        "assert all(next((b.root/'uploads').glob('*.pdf')).read_bytes()==pdf for b in bundles); "
        "db=a.transaction(); conn=db.__enter__(); "
        'assert conn.execute("SELECT SUM(amount) FROM usage '
        "WHERE kind='chat' AND scope!='*'\").fetchone()[0]==1; "
        "db.__exit__(None,None,None)",
    )
    report = {
        "project": project,
        "restored_project": restored,
        "image": image,
        "backend_url": url,
        "first_event_seconds": first_at,
        "completion_seconds": finished,
        "delayed_first_event_seconds": delayed_first_at,
        "delayed_completion_seconds": delayed_finished,
        "restore": "passed",
        "documents": documents,
        "conversation": conversation,
        "offline_fixture": True,
        "production_entrypoint": "passed, provider disabled",
        "capabilities_file": "capabilities.json",
        "override": str(restore_override),
    }
    (evidence / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in report.items() if k != "capabilities_file"}), flush=True)


if __name__ == "__main__":
    main()
