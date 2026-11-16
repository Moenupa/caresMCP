"""Command-line entry point for listing registries and running the MCP server."""

from pathlib import Path
from typing import Annotated, Literal

import typer
from rich.console import Console

from caresmcp.api.io.table import function_help, registry_table, rich_table_to_df
from caresmcp.registry import REGISTRIES

app = typer.Typer(add_completion=False)
console = Console()

HELP_KEYWORDS = {"help", "?"}


@app.command("list")
def list_functions(
    category: str | None = typer.Argument(None),
    output: Path | None = None,
) -> None:
    """Print registered tools and prompts grouped by catalogue category."""
    try:
        table = registry_table(category)
    except ValueError as error:
        raise typer.BadParameter(str(error), param_hint="category") from error
    console.print(table)
    if output is not None:
        rich_table_to_df(table).to_csv(output, sep="\t", index=False)


@app.command("debug")
def debug_function(
    category: str, name: str, kwargs: str, save_dir: Path = Path(".")
) -> None:
    """Run a registry function with JSON kwargs and adapted byte inputs.

    kwargs may be inline JSON, a .json or .jsonl file, or help to print the
    function's signature. Byte inputs may be paths, base64 data URIs, or HTTP URLs.
    Outputs are saved under save_dir as category-name files, with a
    numeric suffix when multiple entries are supplied.
    """
    from caresmcp.api.decorators import adapt_byte_inputs
    from caresmcp.api.decorators.file_io import save_results
    from caresmcp.api.io.function import load_kwargs

    registry = REGISTRIES.get(category)
    if registry is None:
        raise typer.BadParameter(
            f"Choose one of: {', '.join(REGISTRIES)}", param_hint="category"
        )
    function = registry.get_all().get(name)
    if function is None:
        raise typer.BadParameter(
            f"Function {name!r} not found in category {category!r}. "
            f"Available: {', '.join(sorted(registry.get_all()))}",
            param_hint="name",
        )

    if kwargs.strip().lower() in HELP_KEYWORDS:
        console.print(f"[bold]{category}.{name}[/]")
        console.print(function_help(function))
        return

    try:
        entries = load_kwargs(kwargs)
    except (OSError, ValueError) as error:
        raise typer.BadParameter(str(error), param_hint="kwargs") from error

    console.print(f"Debugging {category!r} {name!r}", style="dim")
    function = adapt_byte_inputs(function)
    for index, entry in enumerate(entries, start=1):
        console.print(f"{index:02d}. {entry}")
        output = function(**entry)
        suffix = "" if len(entries) == 1 else f"_{index}"
        saved = save_results(output, save_dir / f"{category}-{name}{suffix}")
        console.print(
            " " * len(str(f"{index:02d}")) + f"  Results: {saved!r}", style="dim"
        )


@app.command("mcp")
def mcpserve(
    host: str | None = typer.Option(
        None, help="Host address to bind to (default: 127.0.0.1)."
    ),
    port: int | None = typer.Option(
        None, min=0, max=65535, help="Port to bind to (default: 8000)."
    ),
    path: str | None = typer.Option(None, help="HTTP endpoint path (default: /mcp)."),
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] | None = (
        typer.Option(None, help="Server log level (default: INFO).")
    ),
    show_banner: bool = typer.Option(False, help="Display the server banner."),
    json_response: bool | None = typer.Option(
        None, help="Return JSON responses instead of SSE streams."
    ),
    stateless_http: bool | None = typer.Option(
        None, help="Handle each request in a new HTTP transport session."
    ),
    allowed_hosts: Annotated[
        list[str] | None,
        typer.Option(
            help="Additional Host header values accepted by the request guard."
        ),
    ] = None,
    allowed_origins: Annotated[
        list[str] | None,
        typer.Option(help="Additional browser origins trusted by the request guard."),
    ] = None,
) -> None:
    """Start the FastMCP server over streamable HTTP."""
    from caresmcp.mcpserver import mcp

    mcp.run(
        transport="streamable-http",
        show_banner=show_banner,
        host=host,
        port=port,
        path=path,
        log_level=log_level,
        json_response=json_response,
        stateless_http=stateless_http,
        allowed_hosts=allowed_hosts,
        allowed_origins=allowed_origins,
    )


@app.command("mcpdev")
def mcpinspect(
    ui_port: int | None = typer.Option(
        None, min=1, max=65535, help="Port for the MCP Inspector web UI."
    ),
    server_port: int | None = typer.Option(
        None, min=1, max=65535, help="Port for the MCP Inspector proxy server."
    ),
    inspector_version: str | None = typer.Option(
        None,
        help="Version of @modelcontextprotocol/inspector to run (default: latest).",
    ),
    reload: bool = typer.Option(True, help="Restart the server when files change."),
) -> None:
    """Launch MCP Inspector against the caresmcp server over stdio.

    Requires Node.js and npx. The Inspector UI is served by
    @modelcontextprotocol/inspector, and commands are forwarded to a caresmcp
    server subprocess for interactive debugging.
    """
    import os
    import subprocess
    import sys

    server_spec = Path(__file__).with_name("mcpserver.py")
    command = [sys.executable, "-m", "fastmcp.cli", "dev", "inspector"]
    command.append(str(server_spec))
    command.append("--reload" if reload else "--no-reload")
    if ui_port is not None:
        command.extend(["--ui-port", str(ui_port)])
    if server_port is not None:
        command.extend(["--server-port", str(server_port)])
    if inspector_version is not None:
        command.extend(["--inspector-version", inspector_version])

    # FastMCP starts the server with the `fastmcp` console script, which may
    # live in an isolated tool environment rather than on PATH.
    env = os.environ.copy()
    env["PATH"] = os.pathsep.join(
        [str(Path(sys.executable).parent), env.get("PATH", "")]
    )
    console.print(f"Starting MCP Inspector for {server_spec}", style="dim")
    raise typer.Exit(subprocess.run(command, check=False, env=env).returncode)
