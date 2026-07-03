# 💡 IdeaLens

**A multi-agent AI system that evaluates any product or business idea through four expert lenses — in parallel.**

Built for the Kaggle 5-Day AI Agents Intensive Capstone (Agents for Business Track).

---

## The Problem

Launching a product, service, or business idea in a new market usually means running four separate research processes before you can be confident it'll work: is this culturally appropriate and well-received here? Does the business case actually hold up? What's the environmental and social footprint? And can everyone — including people with disabilities — actually use it? Each of these normally requires its own research, its own experts, and its own time. Most founders and teams only get to a couple of them, if any, before launch — usually cultural fit and business case, while sustainability and accessibility get skipped entirely due to time and cost.

## The Solution

IdeaLens runs all four evaluations **simultaneously** instead of sequentially, using four specialist AI agents grounded in their own retrieval-augmented knowledge bases, and synthesizes the results into one consolidated, actionable report.

## What It Does

Describe a product, service, or business idea and a target culture/market. IdeaLens runs four specialist AI agents **simultaneously**, each powered by Gemini 2.5 Flash and grounded in its own retrieval-augmented knowledge base:

| Lens | Evaluates |
|---|---|
| 🌍 **Cultural Localisation** | Language & tone, visual identity, cultural references, values alignment, timing, and taboos to avoid |
| 💼 **Business Case** | Market opportunity (TAM/SAM/SOM), revenue models, top risks, competitive edge, and the first validation step |
| 🌱 **Sustainability** | Environmental & social impact, quick wins, long-term circularity roadmap, and relevant certifications |
| ♿ **Accessibility** | Visual, motor, cognitive, and hearing accessibility, WCAG 2.2 compliance path, and inclusive language |

Results are synthesized into a single consolidated report with actionable next steps.

---

## Architecture

```
                     ┌─────────────────┐
                     │   index.html     │  (frontend/)
                     │  HTML/CSS/JS UI  │
                     └────────┬─────────┘
                              │ POST /analyze
                              ▼
                     ┌─────────────────┐
                     │   backend.py     │  FastAPI server
                     │  (rate limit,    │
                     │   sanitization,  │
                     │  static hosting) │
                     └────────┬─────────┘
                              │
                              ▼
                   ┌────────────────────┐
                   │  orchestrator.py    │
                   │  IdeaLensOrchestrator│
                   └──────────┬──────────┘
                              │ ThreadPoolExecutor
              ┌───────┬───────┼───────┬───────┐
              ▼       ▼       ▼       ▼
          culture  business sustain. access.  ← agents/*.py
              │       │       │       │           (Gemini 2.5 Flash)
              └───────┴───────┴───────┘
                              │
                              ▼
                    ┌──────────────────┐
                    │      rag.py       │  In-memory vector RAG
                    │  (text-embedding- │  (knowledge_base/*)
                    │   004, cosine sim)│
                    └──────────────────┘
```

Two additional entry points wrap the same orchestrator:
- **`app.py`** — a CLI for running analyses from the terminal (development/testing)
- **`mcp_server.py`** — exposes IdeaLens as MCP tools so AI coding assistants (e.g. Claude Desktop) can call it directly

All three entry points (web, CLI, MCP) share the same underlying agent logic — nothing is duplicated per interface.

---

## Tech Stack

- **LLM:** Google Gemini 2.5 Flash (via `google-genai` SDK)
- **Embeddings:** `text-embedding-004` (with automatic fallback to `gemini-embedding-2`)
- **Backend:** FastAPI + Uvicorn
- **Agent protocol:** FastMCP (Model Context Protocol)
- **Frontend:** Vanilla HTML/CSS/JS
- **Parallelism:** Python `ThreadPoolExecutor`
- **Deployment:** Google Cloud Run (`us-central1`)

---

## Project Structure

```text
idealens/
├── .env                  # GEMINI_API_KEY and local config (not committed)
├── .gitignore
├── requirements.txt
├── README.md
├── app.py                # CLI entry point
├── backend.py             # FastAPI server — serves POST /analyze, GET /health
├── orchestrator.py        # Parallel execution and report synthesis
├── rag.py                 # In-memory vector RAG (embed, chunk, retrieve)
├── mcp_server.py          # MCP tool server for AI coding assistants
├── security.py            # Input sanitization, rate limiting, prompt-injection & output checks
├── skills.py               # Shared deterministic calculations (market sizing, CO2 estimate, WCAG check)
├── agents/
│   ├── culture.py
│   ├── business.py
│   ├── sustainability.py
│   └── accessibility.py
├── frontend/
│   ├── index.html          # Served as static files by backend.py at "/"
│   └── assets/
│       └── hero.png
└── knowledge_base/
    ├── culture/
    ├── business/
    ├── sustainability/
    └── accessibility/
```

---

## Setup

**1. Clone and install dependencies**

```bash
git clone <your-repo-url>
cd idealens
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

**2. Configure your API key**

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_key_here
```

**3. Add knowledge base documents (optional but recommended)**

Drop `.txt` reference documents into the relevant `knowledge_base/<domain>/` folder for each lens. If a folder is empty, that lens still runs — just without grounded RAG context.

---

## Usage

### Web app (primary)

```bash
python backend.py
```

Then open `http://localhost:8000` in a browser — `backend.py` serves `frontend/index.html` directly, or open the file yourself if running the backend separately.

### CLI

```bash
python app.py analyze --idea "A subscription meal-kit service" --regions "Japan" --population 500000 --spend 30
python app.py info
```

### MCP (for AI coding assistants)

```bash
python mcp_server.py
```

Exposes `analyze_new_idea` and `query_knowledge_base` as MCP tools.

---

## Security & Guardrails

- **Input sanitization** — strips control characters, enforces a 3,000-character max on submitted ideas
- **Prompt-injection detection** — regex-based screening for common injection patterns before any idea reaches Gemini
- **Rate limiting** — 15 requests/minute per IP, in-memory sliding window
- **Output safety check** — flags empty, refused, or degenerate agent output, and catches RAG source-filename leakage, before a report reaches the frontend

---

## Course Concepts Demonstrated

Per the capstone evaluation rubric (minimum 3 of 6 required):

| Concept | Where |
|---|---|
| **MCP Server** | `mcp_server.py` — exposes `analyze_new_idea` and `query_knowledge_base` as MCP tools for AI coding assistants |
| **Security features** | `security.py` — input sanitization, prompt-injection detection, per-IP rate limiting, and post-generation output safety checks (all wired into `backend.py` and `orchestrator.py`) |
| **Antigravity** | Entire project scaffolded and iteratively built in Google Antigravity — demonstrated in the walkthrough video |
| **Deployability** | Fully containerizable and deployable to Google Cloud Run (`us-central1`); optional for judging per capstone guidelines |

IdeaLens is also architecturally a genuine multi-agent system — four specialist agents (culture, business, sustainability, accessibility) run in parallel via `ThreadPoolExecutor` in `orchestrator.py`, each independently grounded with its own RAG-retrieved context — though the agents are built directly on the `google-genai` SDK rather than the ADK `LlmAgent`/`Runner` framework specifically.

---

## Deployment

This project can be run fully locally by following the Setup and Usage instructions above — no live deployment is required to evaluate it. Per the capstone guidelines, deploying to a public endpoint is optional for judging purposes.

If deployed, IdeaLens runs on Google Cloud Run (`us-central1`), live at: `<add your Cloud Run URL here if deployed>`. Deploying it yourself only requires setting `GEMINI_API_KEY` as an environment variable on the Cloud Run service and updating `API_URL` in `frontend/index.html` to point to the deployed URL instead of `localhost:8000`.

---

## Built With

Scaffolded using [Google Antigravity](https://antigravity.google) and developed as part of the [Kaggle 5-Day AI Agents Intensive](https://www.kaggle.com/) Capstone, Agents for Business Track.

