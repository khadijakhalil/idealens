"""
IdeaLens FastAPI Backend.
Serves POST /analyze, GET /health, and mounts the frontend static directory.
"""

import os
import logging
from typing import List
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from dotenv import load_dotenv

from security import sanitise, check_rate, SecurityGuard
from orchestrator import IdeaLensOrchestrator

# Load environment configuration
load_dotenv()

# Setup Logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("idealens.backend")

# Initialize FastAPI App
app = FastAPI(
    title="IdeaLens API",
    description="Backend API for the IdeaLens multi-agent parallel analysis platform.",
    version="1.0.0"
)

# Enable CORS (Cross-Origin Resource Sharing) for all origins — fine for the Kaggle
# demo where frontend/backend may be on different hosts; tighten to your real
# frontend origin if this ever becomes a genuinely public production service.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Single orchestrator instance, created once at process startup and reused across
# every request. This matters: RAGService's embedding cache (_cached_kbs) lives on
# the orchestrator instance, so reusing one instance means the knowledge base is
# only ever embedded once per server process — not once per user request.
orchestrator = IdeaLensOrchestrator()


@app.on_event("startup")
async def startup_event():
    """Pre-ingest all knowledge bases when the server boots, so the first real
    user request isn't the one paying the full embedding cost. Non-fatal if it
    fails (e.g. missing API key at build time) — requests will just ingest lazily."""
    try:
        stats = orchestrator.ingest_all_knowledge_bases()
        logger.info("Startup knowledge base ingestion complete: %s", stats)
    except Exception as e:
        logger.warning("Startup ingestion failed (will retry lazily per-request): %s", e)

# Pydantic request body model for the /analyze endpoint — matches exactly what
# index.html's fetch() call sends as JSON
class AnalysisRequest(BaseModel):
    idea: str = Field(..., description="The user's business, product, or technology idea.")
    target_culture: str = Field(..., description="The culture or region to localize the idea for.")
    selected_lenses: List[str] = Field(
        default=["culture", "business", "sustainability", "accessibility"],
        description="The specific specialist agent perspectives to analyze."
    )

@app.post("/analyze")
async def analyze_idea_endpoint(request_body: AnalysisRequest, request: Request):
    """
    POST endpoint to run multi-agent parallel analysis on a user prompt.
    Performs rate-limiting and prompt-injection sanitization checks.
    """
    client_ip = request.client.host if request.client else "unknown_ip"
    
    # 1. Rate Limiting Check — 15 requests/min per IP, see security.py's check_rate()
    if not check_rate(client_ip):
        logger.warning("Request rejected: Rate limit exceeded for IP %s", client_ip)
        raise HTTPException(
            status_code=400,
            detail="Rate limit exceeded. Please wait a moment before trying again."
        )

    # 2. Input Sanitization — strips control chars, enforces max length.
    # Note: run_parallel_lenses() below re-sanitizes internally too (belt-and-braces,
    # harmless since sanitizing already-clean text is idempotent) — this pass here
    # exists so we can validate/reject *before* touching the orchestrator at all.
    sanitized_idea = sanitise(request_body.idea)
    sanitized_culture = sanitise(request_body.target_culture)
    
    if not sanitized_idea:
        raise HTTPException(
            status_code=400,
            detail="Request rejected: The provided idea is empty or invalid after sanitization."
        )
        
    if not sanitized_culture:
        raise HTTPException(
            status_code=400,
            detail="Request rejected: The target culture field is empty or invalid."
        )

    # 3. Prompt Injection Security Check
    if SecurityGuard.inspect_prompt_injection(sanitized_idea):
        logger.warning("Request rejected: Prompt injection detected from IP %s", client_ip)
        raise HTTPException(
            status_code=400,
            detail="Security violation: Suspicious input pattern detected. Request rejected."
        )

    # Validate selected lenses — silently drops anything not in the known set rather
    # than erroring, so a stray/unexpected value doesn't fail the whole request
    valid_lenses = {"culture", "business", "sustainability", "accessibility"}
    active_lenses = [l for l in request_body.selected_lenses if l in valid_lenses]
    if not active_lenses:
        raise HTTPException(
            status_code=400,
            detail="Request rejected: No valid lenses selected. Choose from: culture, business, sustainability, accessibility."
        )

    logger.info("IP %s running analysis on '%s' for culture '%s'", client_ip, sanitized_idea, sanitized_culture)

    try:
        # 4. Invoke Multi-agent parallel execution using the single shared orchestrator
        # (created once at module load, not per-request — see comment near its
        # definition above). This also runs the per-lens output safety check
        # (SecurityGuard.check_safety_guidelines) internally via run_parallel_lenses.
        results = orchestrator.run_parallel_lenses(
            idea=sanitized_idea,
            target_culture=sanitized_culture,
            selected_lenses=active_lenses
        )
        return results
    except Exception as e:
        logger.error("Error during parallel run: %s", e)
        raise HTTPException(
            status_code=500,
            detail=f"An error occurred while generating analysis reports: {str(e)}"
        )

@app.get("/health")
async def health_check_endpoint():
    """
    Simple health check endpoint to verify backend status — this is what Cloud Run
    pings to confirm the container started successfully.
    """
    return {"status": "ok", "service": "IdeaLens"}

# Mount the frontend directory to serve the static index.html and assets at root.
# This MUST be declared last to prevent catching custom routes (e.g. /analyze, /health) —
# StaticFiles registers a catch-all route, so anything mounted after it would be
# unreachable (shadowed by the static file handler matching first).
frontend_dir = os.path.abspath("frontend")
if os.path.exists(frontend_dir):
    app.mount("/", StaticFiles(directory=frontend_dir, html=True), name="frontend")
    logger.info("Frontend directory mounted at root successfully.")
else:
    # Not fatal — the API still works standalone (e.g. if index.html is opened
    # directly as a local file, or served from a different static host)
    logger.warning("Frontend directory not found. Static file serving disabled.")
