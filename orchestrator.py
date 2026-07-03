"""
IdeaLens Orchestrator Module.
Manages the parallel execution of the specialist agents and coordinates the synthesis of the final report.
"""

import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, Any, List

from rag import RAGService              # Handles knowledge base ingestion + retrieval (RAG)
from security import SecurityGuard      # Input sanitization + prompt-injection detection

# Import the lens functions directly — each is a standalone agent function that takes
# (idea, target_culture, rag_context) and returns a markdown analysis string
from agents.culture import run_culture_lens
from agents.business import run_business_lens
from agents.sustainability import run_sustainability_lens
from agents.accessibility import run_accessibility_lens

logger = logging.getLogger("idealens.orchestrator")

# Central registry mapping lens name -> agent function. Adding a 5th lens later just
# means adding one entry here (plus its RAG query in lens_queries below).
LENS_AGENTS = {
    "culture": run_culture_lens,
    "business": run_business_lens,
    "sustainability": run_sustainability_lens,
    "accessibility": run_accessibility_lens
}

class IdeaLensOrchestrator:
    """Orchestrates RAG ingestion, parallel agent runs, and report synthesis."""

    def __init__(self, kb_root: str = "knowledge_base"):
        # Single RAG service shared across all four lenses; kb_root points to the
        # folder structure expected by RAGService (e.g. knowledge_base/culture/*.pdf)
        self.rag_service = RAGService(kb_root)

    def ingest_all_knowledge_bases(self) -> Dict[str, int]:
        """Ingests all files inside the knowledge_base directories."""
        # One subdirectory per lens; each domain's docs are embedded/indexed independently
        domains = ["culture", "business", "sustainability", "accessibility"]
        results = {}
        for domain in domains:
            results[domain] = self.rag_service.ingest_directory(domain)
        return results

    def run_parallel_lenses(self, idea: str, target_culture: str, selected_lenses: List[str]) -> Dict[str, Any]:
        """
        Runs the selected lens agents in parallel.
        
        Args:
            idea (str): The product/business idea description.
            target_culture (str): The target culture (e.g., Japan, Germany).
            selected_lenses (List[str]): List of active lenses (e.g. ['culture', 'business']).
            
        Returns:
            Dict[str, Any]: Dictionary of analysis reports keyed by lens name.
        """
        # Security Gate — reject empty/malicious input before spending any API calls
        sanitized_idea = SecurityGuard.sanitize_input(idea)
        if not sanitized_idea:
            raise ValueError("Input idea is empty or invalid.")
            
        if SecurityGuard.inspect_prompt_injection(sanitized_idea):
            raise ValueError("Security violation: Potential prompt injection attempt detected.")

        # 1. Retrieve RAG contexts by calling rag.retrieve() with the appropriate lens name and query
        #
        # These are built dynamically from target_culture and sanitized_idea so retrieval
        # actually adapts to whatever culture/idea the user submits, rather than being
        # biased toward one fixed example.
        lens_queries = {
            "culture": f"Cultural norms, business etiquette, and localisation considerations for {target_culture} relevant to: {sanitized_idea}",
            "business": f"Market sizing, monetization strategy, customer retention, and competitive risk in {target_culture} relevant to: {sanitized_idea}",
            "sustainability": f"Environmental sustainability, ESG standards, and impact assessment considerations for {target_culture} relevant to: {sanitized_idea}",
            "accessibility": f"WCAG POUR principles and inclusive design considerations relevant to: {sanitized_idea}"
        }

        logger.info("Retrieving RAG contexts for selected lenses: %s", selected_lenses)
        contexts = {}
        for lens in selected_lenses:
            if lens in lens_queries:
                query = lens_queries[lens]
                contexts[lens] = self.rag_service.retrieve(lens, query)
            else:
                # Lens not in our query map (shouldn't happen given LENS_AGENTS keys match) —
                # fall back to empty context rather than crashing the whole run
                contexts[lens] = ""

        # 2. Run all selected agents simultaneously in parallel using ThreadPoolExecutor
        # One worker per selected lens so a 4-lens run only takes as long as the slowest
        # single agent call, not the sum of all four.
        results = {}
        logger.info("Starting parallel execution of lens agents...")

        # ThreadPoolExecutor (not asyncio/multiprocessing): these are I/O-bound Gemini API
        # calls, not CPU-bound work, so threads give the concurrency win without the
        # complexity of async/await or process-based overhead.
        with ThreadPoolExecutor(max_workers=len(selected_lenses)) as executor:
            future_to_lens = {}
            for lens in selected_lenses:
                if lens in LENS_AGENTS:
                    agent_fn = LENS_AGENTS[lens]
                    # Submit the job with idea, target_culture, and RAG context
                    future = executor.submit(agent_fn, sanitized_idea, target_culture, contexts[lens])
                    future_to_lens[future] = lens
                    
            for future in as_completed(future_to_lens):
                lens = future_to_lens[future]
                try:
                    data = future.result()
                    # Post-generation safety/quality check — catches empty output,
                    # model refusals, and RAG source-filename leakage before this
                    # ever reaches the frontend (see SecurityGuard.check_safety_guidelines)
                    if not SecurityGuard.check_safety_guidelines(data):
                        logger.warning("Lens '%s' output failed safety/quality check; replacing with error message.", lens)
                        results[lens] = (
                            f"Error: The '{lens}' analysis did not pass output validation "
                            "and could not be displayed. Please try rephrasing your idea and running again."
                        )
                    else:
                        results[lens] = data
                        logger.info("Lens '%s' finished successfully.", lens)
                except Exception as exc:
                    logger.error("Lens '%s' generated an exception: %s", lens, exc)
                    # Handle agent errors gracefully: return the error message in the results
                    # so one failing agent doesn't take down the whole /analyze response —
                    # the frontend already knows to render strings starting with "Error" specially
                    results[lens] = f"Error running analysis for '{lens}' agent: {str(exc)}"

        return results

    def analyze_idea(self, raw_idea: str, parameters: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Backward-compatible analyze_idea wrapper calling run_parallel_lenses under the hood.
        """
        # Used by the CLI (app.py), which doesn't expose per-lens selection — so this
        # always runs all four lenses, unlike the web /analyze endpoint which lets the
        # user pick a subset via checkboxes.
        params = parameters or {}
        regions = params.get("regions", ["Japan"])
        target_culture = regions[0] if regions else "Japan"  # only the first region is used
        selected_lenses = ["culture", "business", "sustainability", "accessibility"]
        
        agent_reports = self.run_parallel_lenses(raw_idea, target_culture, selected_lenses)
        
        # Package reports to match the expected format for output rendering
        formatted_reports = {}
        for lens, content in agent_reports.items():
            formatted_reports[lens] = {
                "agent": lens.capitalize(),
                "analysis": content
            }
            
        synthesis = self._synthesize_report(raw_idea, formatted_reports)
        
        return {
            "idea": raw_idea,
            "agent_reports": formatted_reports,
            "synthesized_report": synthesis
        }

    def _synthesize_report(self, idea: str, agent_reports: Dict[str, Any]) -> str:
        """
        Synthesizes the individual reports into a final cohesive document.
        """
        # Pull each lens's analysis text out of the formatted_reports structure;
        # .get() with defaults means a missing/failed lens just renders as an empty section
        culture_summary = agent_reports.get("culture", {}).get("analysis", "")
        business_summary = agent_reports.get("business", {}).get("analysis", "")
        sustainability_summary = agent_reports.get("sustainability", {}).get("analysis", "")
        accessibility_summary = agent_reports.get("accessibility", {}).get("analysis", "")
        
        # Static markdown template stitching all four lens outputs together, plus a
        # fixed set of "Next Steps" — these are generic placeholders, not generated
        # per-idea, so consider having an agent (or Gemini call) generate this section
        # dynamically if genuinely idea-specific recommendations matter for the demo.
        synthesis = f"""# 💡 IdeaLens Comprehensive Analysis: "{idea}"
        
## 📌 Executive Summary
IdeaLens has evaluated the proposed idea through four dimensions in parallel. This consolidated report highlights crucial recommendations for adaptation, monetization, eco-friendliness, and accessibility compliance.

---

## 🌍 Cultural Localisation
{culture_summary}

---

## 💼 Business Case & Feasibility
{business_summary}

---

## 🌱 Environmental Sustainability
{sustainability_summary}

---

## ♿ Inclusivity & Accessibility
{accessibility_summary}

---

## 🎯 Next Steps & Strategic Recommendations
1. **Optimize Materials**: Ensure all physical hardware aligns with the local sustainability index recommendations.
2. **Inclusivity First**: Integrate screen-reader compatibility and local voice synthesis during the early wireframing phase.
3. **Regional Launch**: Focus the initial business pilot on high-impact low-resource areas using localized community partnerships.
"""
        return synthesis

def run_all_lenses(idea: str, target_culture: str, selected_lenses: List[str]) -> Dict[str, Any]:
    """
    Convenience helper function to instantiate orchestrator and run parallel analyses.
    """
    # Creates a fresh orchestrator (and RAGService) per call — fine for one-off scripts/tests,
    # but the FastAPI server should reuse a single orchestrator instance across requests
    # rather than calling this helper, to avoid re-initializing RAG on every /analyze hit.
    orch = IdeaLensOrchestrator()
    return orch.run_parallel_lenses(idea, target_culture, selected_lenses)

if __name__ == '__main__':
    # Manual smoke test: run this file directly (`python orchestrator.py`) to exercise
    # the full parallel pipeline against a fixed example, without needing the CLI or server.
    print("=== Testing Orchestrator Parallel Execution ===")
    import sys
    # For emoji support on windows terminals
    sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
    
    orchestrator = IdeaLensOrchestrator()
    test_idea = "Design a loyalty app for a coffee chain"
    test_culture = "Japan"
    selected = ["culture", "business", "sustainability", "accessibility"]
    
    print(f"Idea: '{test_idea}'")
    print(f"Target Culture: '{test_culture}'")
    print(f"Active Lenses: {selected}\n")
    
    results = run_all_lenses(test_idea, test_culture, selected)
    
    for lens in selected:
        print("\n" + "="*80)
        print(f"LENS: {lens.upper()}")
        print("="*80)
        print(results.get(lens, "No result"))

