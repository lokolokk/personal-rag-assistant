# Personal RAG Assistant

[![CI](https://github.com/lokolokk/personal-rag-assistant/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/lokolokk/personal-rag-assistant/actions/workflows/ci.yml)
[![Deploy backend](https://github.com/lokolokk/personal-rag-assistant/actions/workflows/deploy.yml/badge.svg?branch=main)](https://github.com/lokolokk/personal-rag-assistant/actions/workflows/deploy.yml)

A personal assistant for working with PDF documents.

The interface supports PDF uploads, document selection, saved conversations
and inspection of source citations.

## System architecture

```mermaid
flowchart TD
    UI["React + TypeScript interface"] --> API["FastAPI backend"]

    API --> PDF["PDF text, layout and table extraction"]
    PDF --> Chunks["Structural chunks with source metadata"]
    Chunks --> Index["Chroma vector index + BM25 index"]

    API --> Flow["LangGraph conversation flow"]
    Flow --> Query["First-turn query or rewritten follow-up"]
    Query --> Search["Document-scoped retrieval + RRF"]
    Index --> Search
    Search --> Gate{"Any passages found?"}
    Gate -->|No| Fallback["Unable-to-answer response"]
    Gate -->|Yes| Generate["Grounded generation"]
    Generate --> Citations["Response with source citations"]
    Fallback --> API
    Citations --> API
    API --> UI
```

The backend preserves PDF layout and tables, then stores chunks with their
document and page metadata. Retrieval combines Chroma and BM25 results using
Reciprocal Rank Fusion (RRF).

LangGraph coordinates the conversation. The first turn uses the original
question; later turns use a rewritten search query. Empty retrieval returns
an inability response. Otherwise, the generation path uses retrieved passages
and maps citation references to their sources.

## Technology

- Backend: Python and FastAPI
- Frontend: React and TypeScript
- PDF processing: PyMuPDF and pdfplumber
- Retrieval: Chroma and BM25
- Conversation flow: LangGraph
- Cloud: AWS (EC2, ECR, S3, Systems Manager), Docker Compose, Cloudflare Pages
- Delivery: GitHub Actions and Terraform

## Deployment

```mermaid
flowchart TD
    User["Browser"] --> Pages["Cloudflare Pages<br/>React frontend"]
    Pages -->|HTTPS| Public["ngrok public address"]

    subgraph AWS["AWS (Sydney)"]
        subgraph Server["EC2 server, Docker Compose"]
            Agent["ngrok agent<br/>(connects out only)"] --> Proxy["nginx"]
            Proxy --> Backend["FastAPI backend"]
        end
        Public -.->|tunnel| Agent
        Backend --> Disk[("Encrypted disk<br/>documents, Chroma, SQLite")]
        Registry["ECR image registry"] --> Server
        Params["Parameter Store<br/>settings and secrets"] --> Server
        Disk -.->|weekly snapshot,<br/>on-demand backup| Backups[("S3 backups")]
    end

    Backend -->|HTTPS| Model["Qwen API"]
```

```mermaid
flowchart LR
    PR["Pull request"] --> CI["CI: backend, frontend<br/>and docker checks"]
    CI -->|merge to main| Build["Build and push image"]
    Build --> Registry["ECR"]
    Registry --> Deploy["Update the server<br/>through Systems Manager"]
```

- **No open ports.** The server has no inbound rules and no SSH. The tunnel
  connects outward, and administration goes through AWS Systems Manager.
- **Infrastructure as code.** Terraform defines the server, registry, backup
  bucket, snapshot policy and least-privilege access.
- **Continuous integration.** Three required checks (backend, frontend, docker)
  must pass before a pull request can merge into `main`.
- **Continuous delivery.** Each merge to `main` builds an immutable image tagged
  with the commit, pushes it to the registry and updates the running server.
- **On-demand server.** Start and stop buttons (GitHub Actions) keep the cost low.
  The server also stops itself after three hours.
- **Secrets stay out of the repository.** API keys live in Parameter Store and are
  read by the server at start-up.
- **Backups.** The data disk is snapshotted weekly, and a script saves a
  checksummed copy to a private bucket that can be restored on a laptop.

## Status

The backend runs on AWS and the frontend on Cloudflare Pages.

**Website:** [Personal RAG Assistant](https://personal-rag-assistant-ymwon04.pages.dev/)

**Live demo available on request.** The server runs on demand to keep costs low,
and access requires an invitation. Contact the administrator or the person who
shared this website to have the demo started and to receive an invitation link.
