"""Synthetic test provider mounted only into a disposable acceptance container."""

import asyncio
import json
import sys
import time
from collections.abc import AsyncIterator, Iterator
from pathlib import Path

import fitz
import uvicorn
from fastapi.responses import StreamingResponse

from app.api.v1.live import create_live_application
from app.services.access import AccessStore
from app.services.evaluation.offline import offline_network
from app.services.retrieval.vector_store import DeterministicEmbeddingFunction
from app.services.workspaces import WorkspaceManager

ROOT = Path("/app/storage/live")


class SyntheticProvider:
    def embeddings(self, texts: list[str]) -> list[list[float]]:
        return [[float(x) for x in v] for v in DeterministicEmbeddingFunction(1024)(texts)]

    def reformulate(self, prompt: str) -> str:
        return prompt.split("User Follow-up:\n")[-1].split("\n\nStandalone Search Query:")[0]

    def generate(self, prompt: str) -> Iterator[str]:
        _ = prompt
        yield "Synthetic quartz evidence "
        time.sleep(2)
        yield "[1]."


def seed() -> None:
    if ROOT.exists():
        raise ValueError("Refusing to seed existing live storage")
    access = AccessStore(ROOT / "control.sqlite3")
    manager = WorkspaceManager(ROOT, access, None)
    values: dict[str, object] = {}
    for name in ("a", "b", "revoked"):
        workspace, invitation = access.create_invitation()
        manager.get(workspace)
        session = access.redeem(invitation)
        values[name] = {
            "workspace": workspace,
            "invitation": invitation,
            "session": session.session_token,
        }
        if name == "revoked":
            access.revoke(workspace)
    with fitz.open() as pdf:
        pdf.new_page().insert_text((72, 72), "Synthetic quartz evidence for workspace isolation.")
        Path("/verification-output/synthetic.pdf").write_bytes(pdf.tobytes())
    # Raw test capabilities are saved only to the explicit private host evidence mount.
    Path("/verification-output/capabilities.json").write_text(json.dumps(values), encoding="utf-8")


if __name__ == "__main__":
    with offline_network():
        if sys.argv[1] == "seed":
            seed()
        else:
            app = create_live_application(ROOT, SyntheticProvider(), ["http://127.0.0.1:8183"])

            @app.get("/api/v1/_verification/delayed")
            async def delayed() -> StreamingResponse:
                async def events() -> AsyncIterator[str]:
                    await asyncio.sleep(2)
                    yield 'event: token\ndata: {"token":"synthetic"}\n\n'
                    await asyncio.sleep(2)
                    yield 'event: done\ndata: {"finishReason":"stop"}\n\n'

                return StreamingResponse(
                    events(),
                    media_type="text/event-stream",
                    headers={"X-Accel-Buffering": "no", "Cache-Control": "no-store"},
                )

            uvicorn.run(app, host="0.0.0.0", port=8000, access_log=False, workers=1)
