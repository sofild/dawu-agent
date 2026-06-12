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
    """Run the interactive agent loop with streaming event output."""
    from dawu_agent.core.agent import Agent
    from dawu_agent.core.events import (
        AssistantTextEvent,
        CompactionEvent,
        ErrorEvent,
        FinalResponseEvent,
        StateChangeEvent,
        ToolResultEvent,
        ToolUseEvent,
        TurnStartEvent,
    )

    agent = Agent(settings=settings, telemetry=telemetry)
    await agent.initialize()

    session_id = (
        agent._session_log.session_id
        if agent._session_log is not None
        else "n/a"
    )
    detail_log_path = (
        agent._session_log.detail_log_file
        if agent._session_log is not None
        else "n/a"
    )
    console.print(
        f"[dim]session:[/dim] [bold]{session_id}[/bold]    "
        f"[dim]详细日志:[/dim] [bold]{detail_log_path}[/bold]"
    )
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
                final_text = await _stream_to_console(agent, user_input, telemetry)

            # Final aggregated response, kept visually distinct from the
            # intermediate streaming text already printed.
            console.print(f"\n[bold green]Agent:[/bold green] {final_text}\n")

        except KeyboardInterrupt:
            console.print("\n[yellow]Interrupted. Goodbye![/yellow]")
            await agent.shutdown()
            break
        except Exception as e:
            telemetry.logger.error("agent.error", error=str(e))
            console.print(f"[red]Error: {e}[/red]")


async def _stream_to_console(agent, user_input: str, telemetry) -> str:
    """Consume `agent.run_stream()` and print key events as they happen.

    Returns the final aggregated response text (joined from all is_final=True
    AssistantTextEvent and FinalResponseEvent payloads). Final response is
    NOT printed here so the caller can render it with its own framing.
    """
    from dawu_agent.core.events import (
        AssistantTextEvent,
        CompactionEvent,
        ErrorEvent,
        FinalResponseEvent,
        StateChangeEvent,
        ToolResultEvent,
        ToolUseEvent,
        TurnStartEvent,
    )

    final_parts: list[str] = []
    printed_first_assistant_text = False
    spinner_ctx = console.status("[bold cyan]Agent 思考中...[/bold cyan]", spinner="dots")

    with spinner_ctx:
        try:
            async for event in agent.run_stream(user_input):
                if isinstance(event, StateChangeEvent):
                    # Quiet — just show transitions to running/idle/error.
                    if event.new in ("running", "idle", "error", "expired", "paused"):
                        spinner_ctx.update(f"[bold cyan]Agent {event.new}...[/bold cyan]")

                elif isinstance(event, TurnStartEvent):
                    console.print(f"[bold cyan]── Turn {event.turn} ──[/bold cyan]")

                elif isinstance(event, AssistantTextEvent):
                    if event.is_final:
                        final_parts.append(event.content)
                    else:
                        # Stop the spinner for the first streamed chunk so
                        # text appears in-line. Re-render status updates are
                        # not safe with Live, so we keep status as-is.
                        console.print(event.content, end="", highlight=False)
                        printed_first_assistant_text = True

                elif isinstance(event, ToolUseEvent):
                    arg_preview = str(event.tool_input)
                    if len(arg_preview) > 120:
                        arg_preview = arg_preview[:120] + "…"
                    console.print(
                        f"\n[yellow]🔧 调用工具:[/yellow] "
                        f"[bold]{event.tool_name}[/bold] [dim]({arg_preview})[/dim]"
                    )

                elif isinstance(event, ToolResultEvent):
                    content = event.content or ""
                    if event.is_error:
                        console.print(
                            f"[red]❌ 工具失败:[/red] {content[:200]}"
                        )
                    else:
                        preview = content[:200]
                        if len(content) > 200:
                            preview += "…"
                        console.print(f"[dim]📦 结果(前 200 字): {preview}[/dim]")

                elif isinstance(event, CompactionEvent):
                    console.print(f"[dim]🗜 上下文压缩: {event.detail}[/dim]")

                elif isinstance(event, ErrorEvent):
                    detail = event.detail or ""
                    console.print(
                        f"[red]⚠ {event.error_type}[/red] "
                        f"[dim](turn={event.turn}, action={event.action})[/dim] "
                        f"{detail[:200]}"
                    )
                    telemetry.logger.error(
                        "agent.event_error",
                        error_type=event.error_type,
                        detail=detail,
                        turn=event.turn,
                    )

                elif isinstance(event, FinalResponseEvent):
                    if event.text:
                        final_parts.append(event.text)
        except Exception as e:
            # Re-raise so the outer loop's handler can log/display it; we
            # also surface a single line so the user knows the run aborted.
            console.print(f"[red]⚠ 流式中断: {e}[/red]")
            telemetry.logger.error("agent.stream_error", error=str(e))
            raise

    if printed_first_assistant_text:
        # Ensure we are on a new line before caller prints "Agent: …"
        console.print("")

    return "".join(final_parts)


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
