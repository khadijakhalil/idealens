"""
IdeaLens MCP (Model Context Protocol) Server.
Enables LLMs and AI coding assistants to interact with IdeaLens agents and analysis tools directly.
"""

import os
import sys
import logging
from typing import Optional

# Setup standard logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger("idealens.mcp")

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:
    logger.error("The 'mcp' package is not installed. Please run: pip install mcp")
    sys.exit(1)

# Initialize FastMCP Server
mcp = FastMCP("IdeaLens")

# Initialize Orchestrator lazily
orchestrator = None

def get_orchestrator():
    global orchestrator
    if orchestrator is None:
        from orchestrator import IdeaLensOrchestrator
        # Default knowledge base root path
        kb_root = os.getenv("IDEALENS_KB_ROOT", "knowledge_base")
        orchestrator = IdeaLensOrchestrator(kb_root)
    return orchestrator

@mcp.tool()
def analyze_new_idea(idea: str, regions: Optional[str] = None) -> str:
    """
    Analyzes any product or business idea across Cultural, Business, Sustainability, and Accessibility perspectives in parallel.
    
    Args:
        idea: The detailed description of the product/business idea.
        regions: Comma-separated list of target regions (e.g., 'Southeast Asia, East Africa').
    """
    try:
        orch = get_orchestrator()
        
        # Parse regions
        regions_list = [r.strip() for r in regions.split(",")] if regions else ["Global"]
        parameters = {"regions": regions_list}
        
        # Run orchestrator
        result = orch.analyze_idea(idea, parameters)
        
        return result["synthesized_report"]
    except Exception as e:
        logger.error("Error analyzing idea via MCP: %s", e)
        return f"Error: Failed to analyze idea. Details: {str(e)}"

@mcp.tool()
def query_knowledge_base(domain: str, query: str) -> str:
    """
    Retrieves snippets from the RAG knowledge base of a specific specialist agent.
    
    Args:
        domain: One of 'culture', 'business', 'sustainability', 'accessibility'.
        query: The search term or topic.
    """
    valid_domains = ["culture", "business", "sustainability", "accessibility"]
    if domain not in valid_domains:
        return f"Error: Invalid domain. Must be one of {valid_domains}"
        
    try:
        orch = get_orchestrator()
        hits = orch.rag_service.retrieve(domain, query)
        if not hits:
            return f"No matching documents found in the '{domain}' knowledge base."
            
        response_lines = [f"### Relevant RAG results for '{domain}':"]
        for hit in hits:
            response_lines.append(f"\n- **Source:** {hit['source']} (Relevance: {hit.get('score', 0):.2f})")
            response_lines.append(f"  {hit['content']}")
            
        return "\n".join(response_lines)
    except Exception as e:
        logger.error("Error querying RAG via MCP: %s", e)
        return f"Error: Failed to query knowledge base. Details: {str(e)}"

if __name__ == "__main__":
    # Ingest knowledge base on startup
    try:
        get_orchestrator().ingest_all_knowledge_bases()
    except Exception as err:
        logger.warning("Could not pre-ingest knowledge bases: %s", err)
        
    # Start the FastMCP server
    logger.info("Starting IdeaLens MCP server...")
    mcp.run()
