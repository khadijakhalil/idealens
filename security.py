"""
IdeaLens Security Module.
Provides input validation, prompt sanitization, rate limiting, and output verification.
"""

import re
import time
import logging

logger = logging.getLogger("idealens.security")

# Simple in-memory request store for rate limiting.
# NOTE: this dict lives in one process's memory, so it resets on every deploy/restart
# and isn't shared across Cloud Run instances if it ever scales beyond one — fine for
# a single-instance capstone demo, but not a durable rate limiter for production.
_rate_limits = {}

def sanitise(text: str) -> str:
    """
    Cleans and sanitizes user input to prevent injection and remove suspicious characters.
    
    Args:
        text (str): Raw input string.
        
    Returns:
        str: Sanitized input string.
    """
    if not text:
        return ""
        
    # Strip leading/trailing whitespaces
    sanitized = text.strip()
    
    # Remove control characters and non-printable sequences (keeps newlines/tabs,
    # which are legitimate in a multi-line idea description)
    sanitized = "".join(ch for ch in sanitized if ch.isprintable() or ch in "\n\r\t")
    
    # Enforce maximum length constraint — caps both cost (fewer tokens sent to Gemini)
    # and risk (long inputs are a common vector for prompt injection padding)
    MAX_LENGTH = 3000
    if len(sanitized) > MAX_LENGTH:
        logger.warning("Input truncated due to exceeding maximum allowed length of %d", MAX_LENGTH)
        sanitized = sanitized[:MAX_LENGTH]
        
    return sanitized

def check_rate(client_ip: str) -> bool:
    """
    Provides a simple sliding-window rate limit (e.g., max 15 requests per minute per IP).
    
    Args:
        client_ip (str): IP address of the requester.
        
    Returns:
        bool: True if request is allowed, False if rate-limited.

    ⚠️ NOTE: this function isn't called anywhere in orchestrator.py or the agent files
    reviewed so far. If the FastAPI /analyze endpoint doesn't call it either, your
    deployed Cloud Run app currently has NO rate limiting — worth wiring this in at
    the endpoint level (pass request.client.host as client_ip) before making the
    URL public, since each request costs real Gemini API calls.
    """
    now = time.time()
    if client_ip not in _rate_limits:
        _rate_limits[client_ip] = []
        
    # Remove logs older than 60 seconds — sliding window, not a fixed bucket
    _rate_limits[client_ip] = [t for t in _rate_limits[client_ip] if now - t < 60]
    
    # Check limit (15 requests per minute)
    if len(_rate_limits[client_ip]) >= 15:
        logger.warning("Rate limit exceeded for IP: %s", client_ip)
        return False
        
    _rate_limits[client_ip].append(now)
    return True

class SecurityGuard:
    """Handles security checks, sanitization, and compliance logic for IdeaLens inputs and outputs."""

    @staticmethod
    def sanitize_input(text: str) -> str:
        return sanitise(text)

    @staticmethod
    def inspect_prompt_injection(text: str) -> bool:
        """
        Detects potential prompt injection attempts.

        Simple regex pattern match — not exhaustive (a determined attacker could phrase
        around these patterns), but catches the common/obvious "ignore previous
        instructions" style attempts, which is a reasonable bar for a capstone demo.
        """
        patterns = [
            r"ignore\s+(?:all\s+)?previous\s+instructions",
            r"system\s*prompt",
            r"disregard\s+(?:all\s+)?prior",
            r"you\s+are\s+now\s+a\s+different\s+agent",
            r"new\s+role",
            r"bypass\s+restrictions",
        ]
        
        for pattern in patterns:
            if re.search(pattern, text, re.IGNORECASE):
                logger.warning("Potential prompt injection pattern detected: %s", pattern)
                return True
                
        return False

    @staticmethod
    def check_safety_guidelines(output_text: str) -> bool:
        """
        Lightweight post-generation check on an agent's output before it's returned
        to the frontend.

        This is NOT a full content-moderation system — Gemini's own built-in safety
        filters handle that. This is a fast, deterministic sanity check for the
        failure modes most likely to actually show up in an IdeaLens report:
        empty/degenerate output, the model refusing outright instead of analyzing,
        or leaking RAG source filenames/citations that the agent prompts explicitly
        instruct it to avoid (see the "CRITICAL INSTRUCTIONS" block in each agent file).

        Args:
            output_text (str): Raw text returned by a lens agent (Gemini response.text).

        Returns:
            bool: True if output passes checks, False if it should be flagged/replaced.
        """
        if not output_text or not output_text.strip():
            logger.warning("Safety check failed: output is empty.")
            return False

        # Catch obvious model refusals — if one of these fires, something caused
        # Gemini to decline instead of producing an analysis report
        refusal_patterns = [
            r"\bI cannot (help|assist|provide|generate)\b",
            r"\bI['’]m not able to\b",
            r"\bI['’]m sorry,?\s*(but\s*)?I can['’]t\b",
            r"\bas an AI (language model|assistant)\b",
        ]
        for pattern in refusal_patterns:
            if re.search(pattern, output_text, re.IGNORECASE):
                logger.warning("Safety check failed: possible model refusal detected.")
                return False

        # Catch RAG source leakage — agent prompts explicitly tell Gemini not to
        # mention source filenames or add a "Sources Referenced" section. If this
        # shows up anyway, those instructions were bypassed or ignored.
        leakage_patterns = [
            r"\bSources?\s+Referenced\b",
            r"\[Source:\s*[\w\-. ]+\]",
            r"\.txt\b",
        ]
        for pattern in leakage_patterns:
            if re.search(pattern, output_text, re.IGNORECASE):
                logger.warning("Safety check failed: possible RAG source/filename leakage detected.")
                return False

        # Catch degenerate/too-short output — each lens is prompted to produce a
        # multi-section markdown report (5-6 headed sections), so anything this
        # short almost certainly means generation failed silently rather than
        # genuinely being a valid short analysis
        MIN_LENGTH = 200
        if len(output_text.strip()) < MIN_LENGTH:
            logger.warning("Safety check failed: output too short (%d chars).", len(output_text.strip()))
            return False

        return True
