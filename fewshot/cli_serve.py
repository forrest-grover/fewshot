"""`fewshot serve`: the REST API and web UI, with settings read as ``uvicorn fewshot.api:app`` reads them."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any

import typer

from .cli_common import ROOT_OPT
from .cli_common import fail as _fail
from .cli_common import workspace as _ws


def _from_env(var: str, default: str) -> str:
    """How a ``serve`` option's help says where its default comes from."""
    return f"[default: ${var}, else {default}]"


def serve(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals  # one per CLI option
    host: str = typer.Option("127.0.0.1", help="Address to listen on; one other than loopback needs a token "
                                               "(or --no-token)"),
    port: int = typer.Option(8000, help="Port to listen on"),
    token: str | None = typer.Option(None, help="Require this token for the API and UI (prefer setting FEWSHOT_TOKEN: "
                                                "an option is visible to other users in the process list) "
                                                + _from_env("FEWSHOT_TOKEN", "none")),
    no_token: bool = typer.Option(False, "--no-token", help="Serve other machines without a token (without one, "
                                                           "only a loopback --host is allowed) "
                                                           + _from_env("FEWSHOT_ALLOW_NO_TOKEN", "no")),
    allow_any_model: bool | None = typer.Option(None, "--allow-any-model/--presets-only",
                                                help="Let API callers load any embedder spec, not just presets "
                                                     + _from_env("FEWSHOT_ALLOW_ANY_MODEL", "presets only")),
    warm: bool | None = typer.Option(None, "--warm/--no-warm", help="Load the models tasks use at startup "
                                                                    + _from_env("FEWSHOT_WARM", "warm")),
    allow_urls: bool | None = typer.Option(
        None, "--allow-urls/--no-urls", help="Let API callers send image URLs for the server to download (public "
                                             "addresses only) " + _from_env("FEWSHOT_ALLOW_URLS", "no URLs")),
    max_upload_mb: float | None = typer.Option(None, min=0.001, help="Largest image accepted per upload "
                                                                     + _from_env("FEWSHOT_MAX_UPLOAD_MB", "50")),
    max_request_mb: float | None = typer.Option(
        None, min=0.001, help="Largest upload request accepted (other request bodies: 1 MB) "
                              + _from_env("FEWSHOT_MAX_REQUEST_MB", "1024")),
    allowed_host: list[str] = typer.Option(
        [], "--allowed-host", help="Host name clients use to reach the server; others are refused (IP addresses "
                                   "always work). Repeatable; '*' allows any. [default: $FEWSHOT_ALLOWED_HOSTS "
                                   "(comma-separated), else localhost, plus this machine's names when not serving "
                                   "locally]"),
    forwarded_allow_ips: str | None = typer.Option(
        None, help="Comma-separated addresses of reverse proxies whose X-Forwarded-For/-Proto headers are trusted "
                   "('*': any), so limits on wrong tokens apply per client, not to the proxy "
                   + _from_env("FEWSHOT_FORWARDED_ALLOW_IPS", "127.0.0.1")),
    workers: int | None = typer.Option(None, min=1, max=256, help="Threads for heavy work (predicting, evaluating, "
                                                                  "uploads) " + _from_env("FEWSHOT_WORKERS", "4")),
    root: Path | None = ROOT_OPT,
) -> None:
    """Run the REST API and web UI.

    Settings not given as options are read from the environment as for `uvicorn fewshot.api:app`
    (see fewshot.server_settings).
    """
    # uvicorn and the API are only needed here; importing them lazily keeps other commands fast.
    import uvicorn  # pylint: disable=import-outside-toplevel

    from .access import exposure_problem  # pylint: disable=import-outside-toplevel
    from .api import create_app  # pylint: disable=import-outside-toplevel
    from .embedders import report_loading  # pylint: disable=import-outside-toplevel
    from .server_settings import LOCAL_HOSTS, ServerSettings  # pylint: disable=import-outside-toplevel

    given: dict[str, Any] = {k: v for k, v in {
        "home": root, "token": token, "allow_no_token": no_token or None, "allow_any_model": allow_any_model,
        "warm": warm, "allow_urls": allow_urls,
        "allowed_hosts": frozenset(allowed_host) or None, "max_upload_mb": max_upload_mb,
        "max_request_mb": max_request_mb, "forwarded_allow_ips": forwarded_allow_ips, "workers": workers,
    }.items() if v is not None}
    env = {k: v for k, v in os.environ.items() if k not in {f"FEWSHOT_{name.upper()}" for name in given}}
    try:  # the environment, read as uvicorn fewshot.api:app reads it, under the options given
        settings = ServerSettings.from_env(env, host=host)
    except ValueError as e:
        _fail(str(e))
    try:
        settings = dataclasses.replace(settings, **given)
    except ValueError as e:  # a token too weak; the other options are checked by their types
        _fail(f"{e} (--token)")
    if problem := exposure_problem(host, settings.token, settings.allow_no_token):
        _fail(f"{problem} (--token, --no-token)")
    local = host in LOCAL_HOSTS
    if not settings.token and not local:
        typer.secho(f"Warning: serving on {host} without a token: anyone who can reach this port can read and "
                    "change every task.", fg="yellow", err=True)
    hosts = settings.allowed_hosts
    if not local and "*" not in hosts:
        listed = [*(h for h in LOCAL_HOSTS if h in hosts), *sorted(hosts - set(LOCAL_HOSTS))]
        typer.echo(f"Answering requests by IP address or for: {', '.join(listed)}. Reaching it by another name "
                   "needs --allowed-host NAME.", err=True)
    report_loading(None)  # the server loads models on several threads: no load reporting (see report_loading)
    server = create_app(settings, _ws(root, create=True), allow_no_token=settings.allow_no_token)
    uvicorn.run(server, host=host, port=port, proxy_headers=True, forwarded_allow_ips=settings.forwarded_allow_ips)
