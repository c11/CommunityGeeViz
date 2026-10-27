"""The MCP server's contract with its callers survived the move from the
SDK's ``mcp.server.fastmcp`` to standalone FastMCP 4.

Two things changed underneath it that nothing at the tool level shows:

- **Unknown arguments are now refused.** The SDK's argument model ignored
  keys a tool did not declare, so the geeViz agent could stamp
  ``_workload_tag`` / ``_tenant`` / ``_agent_user_email`` /
  ``_agent_session_id`` into ANY tool call and the old ``call_tool``
  wrapper popped them at leisure. FastMCP 4 validates against the
  function's signature and fails the call with
  ``unexpected_keyword_argument``. The fields now come off in
  ``_WorkloadTagMiddleware``, before validation — and still have to land
  in the ContextVars that route and bill the call.
- **"http" became a transport name.** FastMCP calls streamable HTTP
  ``"http"``. ``main()`` matched only ``"streamable-http"``, so
  ``MCP_TRANSPORT=http`` (the spelling FastMCP's own docs use) fell to
  the stdio branch: sandbox defaulted OFF, then ``app.run("http")``
  served HTTP anyway — a network-reachable, unsandboxed ``run_code``.
  Found while porting, before it shipped.
"""
import asyncio
from types import SimpleNamespace

import pytest
from fastmcp import FastMCP

from geeViz.eeAuth.client import (
    CURRENT_SESSION_ID,
    CURRENT_TENANT,
    CURRENT_USER_EMAIL,
)


@pytest.fixture(scope="module")
def srv():
    import geeViz.mcp.server as _s
    return _s


_STAMP = {
    "_workload_tag": "agent__probe__acme__ana-example-org__sess1",
    "_tenant": "ACME",
    "_agent_user_email": "ana@example.org",
    "_agent_session_id": "sess-1",
}


# ── agent-stamped arguments ───────────────────────────────────────────

def test_stamped_fields_reach_the_contextvars_not_the_tool(srv):
    """A probe server running only geeViz's workload-tag layer: the tool
    sees its own arguments, and the routing fields arrive as context."""
    probe = FastMCP("probe", middleware=[srv._WorkloadTagMiddleware()])
    seen = {}

    @probe.tool
    def whoami(x: int) -> str:
        seen.update(x=x, tenant=CURRENT_TENANT.get(),
                    user=CURRENT_USER_EMAIL.get(),
                    session=CURRENT_SESSION_ID.get())
        return "ok"

    result = asyncio.run(probe.call_tool("whoami", {"x": 7, **_STAMP}))
    assert result.structured_content == {"result": "ok"}
    assert seen == {"x": 7, "tenant": "acme",
                    "user": "ana@example.org", "session": "sess-1"}


def test_a_stamped_call_to_the_real_app_is_not_refused(srv):
    """The same, through geeViz's own app and a tool that needs no Earth
    Engine. A refusal comes back as a ValidationError naming the field."""
    out = asyncio.run(srv.app.call_tool(
        "search_fia", {"query": "volume", "kind": "attributes", **_STAMP}))
    assert not out.is_error
    assert '"results"' in out.content[0].text


def test_the_contextvars_do_not_outlive_the_call(srv):
    """They are reset after the call, or the next caller is routed and
    billed as this one."""
    probe = FastMCP("probe", middleware=[srv._WorkloadTagMiddleware()])

    @probe.tool
    def noop() -> str:
        return "ok"

    async def two_calls():
        await probe.call_tool("noop", dict(_STAMP))
        return CURRENT_TENANT.get(), CURRENT_USER_EMAIL.get()

    assert asyncio.run(two_calls()) == ("", "")


# ── transport selection in main() ─────────────────────────────────────

class _NoThread:
    """Stands in for the catalog-prewarm thread main() starts."""
    def __init__(self, *a, **k):
        pass

    def start(self):
        pass


def _run_main(srv, monkeypatch, **env):
    calls = []
    monkeypatch.setattr(srv.app, "run", lambda **kw: calls.append(kw))
    # The server module's own reference only; threading.Thread is untouched.
    monkeypatch.setattr(srv, "threading", SimpleNamespace(Thread=_NoThread))
    monkeypatch.setattr(srv, "_SANDBOX_ENABLED", None)
    for k in ("MCP_TRANSPORT", "MCP_HOST", "MCP_PORT", "MCP_PATH"):
        monkeypatch.delenv(k, raising=False)
    for k, v in env.items():
        monkeypatch.setenv(k, v)
    srv.main()
    assert len(calls) == 1
    return calls[0]


@pytest.mark.parametrize("name", ["streamable-http", "http"])
def test_both_http_spellings_take_the_http_branch(srv, monkeypatch, name):
    kw = _run_main(srv, monkeypatch, MCP_TRANSPORT=name, MCP_HOST="0.0.0.0")
    assert kw["transport"] == "http"
    assert srv._SANDBOX_ENABLED is True, (
        f"MCP_TRANSPORT={name} on 0.0.0.0 left run_code unsandboxed")


def test_http_settings_go_to_run(srv, monkeypatch):
    """FastMCP 4 raises TypeError on host/port/json_response in the
    constructor; they are run() arguments now."""
    kw = _run_main(srv, monkeypatch, MCP_TRANSPORT="streamable-http",
                   MCP_HOST="127.0.0.1", MCP_PORT="8123", MCP_PATH="mcp")
    assert (kw["host"], kw["port"], kw["path"]) == ("127.0.0.1", 8123, "/mcp")
    assert kw["json_response"] is True


def test_rebinding_guard_on_loopback_off_on_wildcard(srv, monkeypatch):
    """Loopback keeps Host/Origin validation. 0.0.0.0 (Cloud Run,
    Databricks Apps) turns it off, as the SDK setting did: the platform
    hands in external hostnames, possibly via a proxy on loopback."""
    kw = _run_main(srv, monkeypatch, MCP_TRANSPORT="http", MCP_HOST="127.0.0.1")
    assert kw["host_origin_protection"] == "auto"
    kw = _run_main(srv, monkeypatch, MCP_TRANSPORT="http", MCP_HOST="0.0.0.0")
    assert kw["host_origin_protection"] is False


def test_stdio_stays_stdio_and_unsandboxed_by_default(srv, monkeypatch):
    kw = _run_main(srv, monkeypatch)
    assert kw["transport"] == "stdio"
    assert srv._SANDBOX_ENABLED is False
