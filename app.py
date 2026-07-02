"""
IdeaLens Main CLI Application.
Provides a rich command-line interface to interact with the multi-agent analysis system.

Note: this is the *local/terminal* entry point (`python app.py analyze ...`), used for
development, debugging, and testing the orchestrator directly. It is separate from the
FastAPI web server (main.py) that powers the browser frontend's "Run IdeaLens" button —
both call into the same `IdeaLensOrchestrator`, so agent logic only needs to live in one place.
"""

import os
import sys
import click
from dotenv import load_dotenv
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.theme import Theme

# Load environment variables (e.g. GOOGLE_API_KEY / Gemini credentials) from .env
load_dotenv()

# Custom color theme for Rich console output — keeps CLI status messages
# (info/warning/danger/success/title) visually consistent and easy to scan
custom_theme = Theme({
    "info": "dim cyan",
    "warning": "magenta",
    "danger": "bold red",
    "success": "bold green",
    "title": "bold yellow"
})
console = Console(theme=custom_theme)

@click.group()
def cli():
    """IdeaLens - Multi-Agent AI System for Competition & Business Idea Analysis."""
    pass  # Entry point group; subcommands (analyze, info) are registered below via @cli.command()

@cli.command()
@click.option("--idea", "-i", required=True, help="Detailed product, service, or business idea to analyze.")
@click.option("--regions", "-r", default="Global", help="Comma-separated target regions.")
@click.option("--population", "-p", default=1000000, type=int, help="Target population size.")
@click.option("--spend", "-s", default=25.0, type=float, help="Expected annual customer spend.")
def analyze(idea: str, regions: str, population: int, spend: float):
    """Runs parallel analysis using Cultural, Business, Sustainability, and Accessibility agents.

    This mirrors what the FastAPI /analyze endpoint does for the web frontend, but runs
    synchronously in the terminal with Rich-formatted output — handy for testing agent
    changes without needing the server or browser running.
    """
    
    console.print(Panel("[title]💡 IdeaLens Multi-Agent Analysis Engine[/title]\n[info]Initializing agents and ingesting RAG knowledge bases...[/info]", border_style="cyan"))
    
    try:
        # Imported here (not top-level) so `python app.py info` and `--help` stay fast
        # and don't pay the cost of loading the full orchestrator/agent stack.
        from orchestrator import IdeaLensOrchestrator
        
        # Initialize orchestrator — this wires up the 4 lens agents (culture, business,
        # sustainability, accessibility) and their Gemini 2.5 Flash / ADK configuration.
        orchestrator = IdeaLensOrchestrator()
        
        # Pre-ingest knowledge base — loads/embeds RAG source documents (e.g. cultural
        # norms, WCAG guidelines, ESG data) used by agents to ground their analysis.
        ingest_stats = orchestrator.ingest_all_knowledge_bases()
        console.print(f"[success]✓[/success] Knowledge base check: {ingest_stats}")
        
        # Prepare parameters passed through to every agent
        regions_list = [r.strip() for r in regions.split(",")]
        parameters = {
            "regions": regions_list,
            "estimated_population": population,
            "average_spend": spend
        }
        
        # Perform analysis — runs all four lens agents (in parallel, per orchestrator
        # implementation) and returns a combined report dict
        console.print(f"[info]Analyzing: '{idea}' in progress...[/info]")
        report = orchestrator.analyze_idea(idea, parameters)
        
        console.print("\n" + "=" * 80 + "\n")
        
        # Render the orchestrator's synthesized summary (combining all 4 lenses) as markdown
        md_content = report["synthesized_report"]
        console.print(Markdown(md_content))
        
        console.print("\n" + "=" * 80 + "\n")
        console.print("[success]✓ Analysis completed successfully![/success]")
        
    except Exception as e:
        # Catch-all so CLI failures (missing API key, network errors, malformed agent
        # output, etc.) print a clean message instead of a raw traceback
        console.print(f"[danger]Error running analysis:[/danger] {str(e)}", err=True)
        sys.exit(1)

@cli.command()
def info():
    """Displays information about the IdeaLens system and specialist agents.

    Static/reference command — doesn't touch the orchestrator, safe to run without
    an API key configured. Useful as a quick sanity check that the CLI is installed correctly.
    """
    console.print(Panel(
        "[bold yellow]IdeaLens System Status[/bold yellow]\n\n"
        "[bold]Specialist Agents Ready:[/bold]\n"
        "1. 🌍 [cyan]Cultural Localisation[/cyan] - Evaluates local adaptation & barriers\n"
        "2. 💼 [cyan]Business Case[/cyan] - Analyzes monetization, TAM/SAM/SOM, and threats\n"
        "3. 🌱 [cyan]Sustainability[/cyan] - Calculates CO2 equivalents & ESG goals\n"
        "4. ♿ [cyan]Accessibility[/cyan] - Evaluates WCAG compliance and design barriers\n\n"
        "[bold]CLI Environment:[/bold] Active",
        border_style="yellow"
    ))

if __name__ == "__main__":
    cli()  # Dispatches to the appropriate subcommand based on sys.argv (e.g. `analyze`, `info`)
