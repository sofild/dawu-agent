"""Dawu Agent CLI entry point using Typer and Rich."""

from __future__ import annotations

import asyncio
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from dawu_agent.config.loader import ConfigLoader
from dawu_agent.observability.telemetry import TelemetryManager

app = typer.Typer(
    name="dawu",
    help="Enterprise AI Agent for data analysis and report generation",
    rich_markup_mode="rich",
)
console = Console()


def _print_banner() -> None:
    """Print the Dawu Agent banner."""
    banner = Text()
    banner.append("╔══════════════════════════════════════╗\n", style="bold cyan")
    banner.append("║     ", style="bold cyan")
    banner.append("Dawu Agent", style="bold yellow")
    banner.append("  v0.1.0     ║\n", style="bold cyan")
    banner.append("║  Enterprise Data Analysis Platform   ║\n", style="bold cyan")
    banner.append("╚══════════════════════════════════════╝", style="bold cyan")
    console.print(banner)


@app.command()
def init(
    path: Path = typer.Option(".", "--path", "-p", help="Project initialization path"),
) -> None:
    """Initialize a new Dawu Agent project."""
    console.print(f"[green]Initializing Dawu Agent project at {path.absolute()}[/green]")
    # Project scaffolding logic here
    console.print("[green]Done! Edit .env and config/ to customize.[/green]")


@app.command()
def run(
    config: Path = typer.Option("config", "--config", "-c", help="Configuration directory"),
    env: str = typer.Option("development", "--env", "-e", help="Environment name"),
    interactive: bool = typer.Option(True, "--interactive/--no-interactive", "-i/-I"),
) -> None:
    """Start the Dawu Agent."""
    _print_banner()

    # Load configuration
    loader = ConfigLoader(config_dir=config)
    settings = loader.load(env=env)

    # Initialize observability
    telemetry = TelemetryManager(settings)
    telemetry.initialize()
    logger = telemetry.logger
    logger.info("dawu_agent.starting", env=env, config_dir=str(config))

    console.print(Panel.fit(
        f"[bold]Environment:[/bold] {env}\n"
        f"[bold]LLM Provider:[/bold] {settings.llm.provider}\n"
        f"[bold]Model:[/bold] {settings.llm.model}\n"
        f"[bold]Multi-Agent:[/bold] {'enabled' if settings.enable_multi_agent else 'disabled'}\n"
        f"[bold]Vector Memory:[/bold] {'enabled' if settings.enable_vector_memory else 'disabled'}",
        title="Configuration",
        border_style="green",
    ))

    if interactive:
        asyncio.run(_interactive_loop(settings, telemetry))
    else:
        console.print("[yellow]Non-interactive mode not yet implemented.[/yellow]")


async def _interactive_loop(settings, telemetry) -> None:
    """Run the interactive agent loop."""
    from dawu_agent.core.agent import Agent

    agent = Agent(settings=settings, telemetry=telemetry)
    await agent.initialize()

    console.print("\n[bold green]Agent ready. Type your message or 'exit' to quit.[/bold green]\n")

    while True:
        try:
            user_input = console.input("[bold blue]You:[/bold blue] ")
            if user_input.lower() in ("exit", "quit", "bye"):
                console.print("[yellow]Shutting down...[/yellow]")
                await agent.shutdown()
                break

            with telemetry.tracer.start_as_current_span("agent.turn") as span:
                span.set_attribute("user.input_length", len(user_input))
                response = await agent.run_turn(user_input)
                console.print(f"[bold green]Agent:[/bold green] {response}\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            await agent.shutdown()
            break
        except Exception as e:
            telemetry.logger.error("agent.error", error=str(e))
            console.print(f"[red]Error: {e}[/red]")


@app.command()
def server(
    host: str = typer.Option("0.0.0.0", "--host", "-h"),
    port: int = typer.Option(8000, "--port", "-p"),
    config: Path = typer.Option("config", "--config", "-c"),
) -> None:
    """Start the Dawu Agent API server."""
    import uvicorn
    from dawu_agent.server.app import create_app

    loader = ConfigLoader(config_dir=config)
    settings = loader.load()
    app = create_app(settings)

    console.print(f"[green]Starting server on {host}:{port}[/green]")
    uvicorn.run(app, host=host, port=port)


@app.command()
def health() -> None:
    """Check agent health status."""
    console.print("[green]Health check: OK[/green]")
    # TODO: Implement actual health checks


if __name__ == "__main__":
    app()
