"""Build a sanitized static invitation frontend locally; never publish it."""

import argparse
import os
import shutil
import subprocess
from pathlib import Path
from urllib.parse import urlsplit

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backend-origin", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--ngrok", action="store_true")
    args = parser.parse_args()
    parsed = urlsplit(args.backend_origin)
    if (
        (
            parsed.scheme != "https"
            and not (parsed.scheme == "http" and parsed.hostname in {"127.0.0.1", "localhost"})
        )
        or not parsed.netloc
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
        or parsed.username
    ):
        parser.error("Specify an HTTPS origin (or localhost for disposable acceptance)")
    output = args.output.resolve()
    if output.exists():
        parser.error("Output must be a fresh directory")
    # Vite loads .env files itself: refuse potentially private frontend env files.
    if list((ROOT / "frontend").glob(".env*")):
        parser.error("Frontend .env files must be removed from this build's source first")
    env = {
        key: value
        for key, value in os.environ.items()
        if not key.startswith(("VITE_", "OPENAI_", "DASHSCOPE_", "LIVE_QWEN_"))
    }
    origin = args.backend_origin.rstrip("/")
    env.update(
        VITE_INVITED_ACCESS="1",
        VITE_API_URL=origin + "/api/v1",
        VITE_NGROK_BYPASS="1" if args.ngrok else "0",
    )
    npm = shutil.which("npm.cmd" if os.name == "nt" else "npm")
    if npm is None:
        raise RuntimeError("npm unavailable")
    subprocess.run([npm, "run", "type-check"], cwd=ROOT / "frontend", env=env, check=True)
    subprocess.run(
        [npm, "exec", "--", "vite", "build", "--outDir", str(output)],
        cwd=ROOT / "frontend",
        env=env,
        check=True,
    )
    headers = (ROOT / "frontend/pages/_headers").read_text(encoding="utf-8")
    (output / "_headers").write_text(
        headers.replace("connect-src 'self'", f"connect-src 'self' {origin}"), encoding="utf-8"
    )
    shutil.copyfile(ROOT / "frontend/pages/_redirects", output / "_redirects")
    print(f"Local invited build: {output}")


if __name__ == "__main__":
    main()
