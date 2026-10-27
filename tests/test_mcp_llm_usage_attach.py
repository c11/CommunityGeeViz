"""The usage record has to cross a process boundary, and land where the
agent looks for it.

The MCP server runs as a subprocess of the agent. It is the only layer
that knows both things: that geeViz just spent tokens, and that a host
exists to tell. It collects what :mod:`geeViz.llmUsage` announces during
one tool call and attaches it to that call's result.

"Where the agent looks" is not a free choice. ``after_tool_callback``
unwraps MCP results in one specific order — ``structuredContent`` first,
then the first JSON ``content[].text``, then an inner
``{"result": "<json string>"}`` envelope — and reads its fields off
whatever that lands on. Attaching the field anywhere else means it is
technically present and never found.

The collecting layer is FastMCP middleware: it sees the ``ToolResult``
the dispatch produces, whose ``structured_content`` FastMCP puts on the
wire as ``structuredContent``.
"""
import asyncio
import json
from types import SimpleNamespace

import pytest
from fastmcp.tools.base import ToolResult
from mcp.types import TextContent

import geeViz.llmUsage as lu


@pytest.fixture(scope="module")
def srv():
    import geeViz.mcp.server as _s
    return _s


def _rec(rid="rid-1"):
    return {"prompt_tokens": 10, "candidates_tokens": 2,
            "response_id": rid, "source": "geeviz.reports",
            "model": "gemini-3.8-flash"}


def _text(text):
    return TextContent(type="text", text=text)


def _usage_layer(srv, inner):
    """Run ``inner`` as the next layer under the real usage middleware."""
    async def call_next(context):
        return await inner()
    return srv._LLMUsageMiddleware().on_call_tool(
        SimpleNamespace(message=None), call_next)


# ── where the field lands ─────────────────────────────────────────────

def test_structured_content_is_preferred(srv):
    result = ToolResult(content=[], structured_content={"success": True})
    srv._attach_llm_usage(result, [_rec()])
    assert result.structured_content["_llm_usage"] == [_rec()]


def test_the_result_envelope_is_written_inside(srv):
    """FastMCP wraps a tool's JSON string as {"result": "<json>"}. The
    agent unwraps that before reading fields, so writing beside it
    rather than inside it puts the record somewhere nothing looks."""
    inner = {"success": True, "message": "done"}
    result = ToolResult(content=[],
                        structured_content={"result": json.dumps(inner)})
    srv._attach_llm_usage(result, [_rec()])
    back = json.loads(result.structured_content["result"])
    assert back["_llm_usage"] == [_rec()]
    assert back["message"] == "done", "the tool's own payload was damaged"


def test_a_text_content_part_is_used_when_there_is_no_structured(srv):
    """A tool with no output schema (one returning text and images)
    has no structured_content to write into."""
    result = ToolResult(content=[_text(json.dumps({"success": True}))])
    srv._attach_llm_usage(result, [_rec()])
    assert json.loads(result.content[0].text)["_llm_usage"] == [_rec()]


def test_non_json_text_parts_are_skipped(srv):
    """A tool returning plain prose must not have JSON spliced into it."""
    result = ToolResult(content=[_text("just some text"),
                                 _text(json.dumps({"success": True}))])
    srv._attach_llm_usage(result, [_rec()])
    assert result.content[0].text == "just some text"
    assert "_llm_usage" in json.loads(result.content[1].text)


# ── what must not happen ──────────────────────────────────────────────

def test_no_records_means_no_field(srv):
    """A tool that spent nothing must return exactly what it returned
    before — an empty accounting field is still noise in the model's
    context."""
    result = ToolResult(content=[], structured_content={"success": True})
    srv._attach_llm_usage(result, [])
    assert "_llm_usage" not in result.structured_content


def test_an_unrecognized_result_shape_is_returned_unharmed(srv):
    """Better to lose the accounting than the tool's answer."""
    weird = SimpleNamespace(structured_content=None, content=None)
    assert srv._attach_llm_usage(weird, [_rec()]) is weird
    assert srv._attach_llm_usage(None, [_rec()]) is None


def test_the_field_name_matches_what_the_agent_drains(srv):
    """Two constants, two repos, one string. If they drift, the field
    is attached and never drained — so it reaches the model AND the
    events table, which is the one outcome the drain exists to stop."""
    assert srv._LLM_USAGE_FIELD == "_llm_usage"


# ── the collector spans the whole call ────────────────────────────────

def test_the_usage_layer_is_the_outermost_wrapper(srv):
    """It has to see everything the tool did. Wrapped INSIDE the scrub
    or workload-tag layers it would miss whatever those call.

    FastMCP nests middleware in list order, first outermost."""
    order = [type(m).__name__ for m in srv.app.middleware]
    ours = [n for n in order if n in (
        "_LLMUsageMiddleware", "_ScrubMiddleware", "_WorkloadTagMiddleware")]
    assert ours == ["_LLMUsageMiddleware", "_ScrubMiddleware",
                    "_WorkloadTagMiddleware"], order


def test_a_tool_call_collects_and_attaches(srv):
    """Through the real middleware: a tool that reports usage the way
    geeViz does comes back with it attached."""
    async def fake_inner():
        lu.report(_rec("from-the-tool"))
        return ToolResult(content=[], structured_content={"success": True})

    out = asyncio.run(_usage_layer(srv, fake_inner))
    attached = out.structured_content["_llm_usage"]
    assert [r["response_id"] for r in attached] == ["from-the-tool"]


def test_a_tool_that_spends_nothing_is_untouched(srv):
    async def fake_inner():
        return ToolResult(content=[], structured_content={"success": True})

    out = asyncio.run(_usage_layer(srv, fake_inner))
    assert "_llm_usage" not in out.structured_content


def test_the_collector_does_not_leak_into_the_next_call(srv):
    """A tool that raises must not leave the collector installed — the
    next call would inherit it and bill one caller for another's work."""
    async def boom():
        lu.report(_rec("before-the-error"))
        raise RuntimeError("tool failed")

    with pytest.raises(RuntimeError):
        asyncio.run(_usage_layer(srv, boom))

    async def quiet():
        return ToolResult(content=[], structured_content={"success": True})

    out = asyncio.run(_usage_layer(srv, quiet))
    assert "_llm_usage" not in out.structured_content


# ── the deadlock this layer caused once ───────────────────────────────

def test_the_usage_module_is_imported_at_module_scope(srv):
    """A call-time ``from geeViz import llmUsage`` inside the wrapper
    hung the server, and left no error anywhere to find it by.

    ``main()`` starts a daemon thread that prewarms EE and the geeViz
    imports; it holds the ``geeViz`` package import lock for tens of
    seconds. A tool call arriving in that window ran the import ON THE
    EVENT LOOP, which blocked on that lock. The tool itself finished —
    the server logged ``"status": "OK"`` — but its result was never
    serialized back over stdio, so the agent sat on a spinner with no
    traceback, no timeout and no clue.

    The module must therefore be bound before any request is served.
    """
    assert hasattr(srv, "_lu"), (
        "llmUsage is not bound at module scope — if it is imported "
        "inside the wrapper instead, a tool call racing the prewarm "
        "thread hangs the server")


def test_the_wrapper_does_not_import_anything(srv):
    """The general form of the bug. ANY import on this path can block on
    the same lock, not just this one."""
    import inspect
    src = inspect.getsource(srv._LLMUsageMiddleware)
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.strip().startswith("#"))
    assert "import " not in code, (
        "the call_tool wrapper imports at request time; it runs on the "
        "event loop while the prewarm thread may hold the import lock")


def test_the_reporting_call_sites_do_not_import_either(srv):
    """Same hazard in the libraries — they are imported BY the prewarm
    thread, so a call-time import from inside one contends for exactly
    the lock that thread is holding."""
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1]
    offenders = []
    for path in (root / "googleMapsLib.py", root / "inventoryLib.py",
                 root / "outputLib" / "reports.py"):
        for i, line in enumerate(
                path.read_text(encoding="utf-8").splitlines(), start=1):
            if "from geeViz import llmUsage" not in line:
                continue
            if line.startswith("from geeViz import llmUsage"):
                continue          # module scope — correct
            offenders.append(f"  {path.name}:{i} {line.strip()}")
    assert not offenders, (
        "llmUsage imported inside a function:\n" + "\n".join(offenders))


# ── the shape the REAL dispatch produces ──────────────────────────────

def test_run_code_attaches_through_the_real_dispatch(srv):
    """End to end on the actual call FastMCP makes, with user code
    reporting from run_code's worker thread.

    The shape is the one that bit before. Under the SDK's
    ``mcp.server.fastmcp`` the real dispatch returned a plain
    ``(content_list, structured_dict)`` TUPLE, not a ToolResult. The
    first version of ``_attach_llm_usage`` only knew ToolResult, so it
    returned the tuple untouched: every geeViz LLM call went unbilled,
    with no error anywhere. Its unit tests all passed, because their
    fake inner returned a ToolResult-shaped object the real dispatch
    never produced.

    Everything between — the ContextVar collector, ``run_in_context``
    carrying it onto the exec thread, the result shape the middleware
    sees — has to line up, and each of those was wrong at some point.
    Only a test that makes the real call can tell, so this one goes
    through ``app.call_tool``, middleware and all.
    """
    import asyncio

    code = ("import geeViz.llmUsage as lu\n"
            "lu.report({'source': 'geeviz.test', 'response_id': 'rc-1',\n"
            "           'prompt_tokens': 5})\n"
            "print('done')\n")
    out = asyncio.run(srv.app.call_tool(
        "run_code", {"code": code, "session_id": "attach-test"}))

    assert isinstance(out, ToolResult), (
        f"FastMCP dispatch returned {type(out).__name__}, not ToolResult "
        "— re-check _attach_llm_usage")
    structured = out.structured_content
    inner = structured
    if len(structured) == 1 and isinstance(structured.get("result"), str):
        inner = json.loads(structured["result"])
    got = inner.get("_llm_usage") or []
    assert [r["response_id"] for r in got] == ["rc-1"], (
        "usage reported inside run_code did not reach the tool result")


# ── the prewarm deadlock, removed rather than locked ──────────────────

def test_there_is_no_prewarm_thread():
    """Three rounds of this decision are recorded in the source.

    A prewarm thread and the first tool call do the SAME heavy
    initialization. Round 1 they raced; round 2 added _init_lock to
    serialize them; round 3 the lock turned the race into a deadlock —
    prewarm holding the lock, wedged inside a numpy C-extension import,
    tool thread blocked acquiring it, neither ever moving.

    Serializing two threads that duplicate each other's work only
    decides which one hangs. One code path is the fix.
    """
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "mcp" / "server.py").read_text(encoding="utf-8")
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "geeviz-mcp-prewarm" not in code, (
        "the EE prewarm thread is back; it deadlocks against the first "
        "tool call under _init_lock")


def test_the_catalog_prewarm_waits_for_init():
    """The one background thread that remains must not overlap the
    geeViz/numpy import cascade — that concurrency is what wedged the
    other one."""
    import pathlib
    src = (pathlib.Path(__file__).resolve().parents[1]
           / "mcp" / "server.py").read_text(encoding="utf-8")
    i = src.index("def _prewarm_catalogs():")
    body = src[i:i + 500]
    code = "\n".join(ln for ln in body.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "_init_done.wait()" in code, (
        "the catalog prewarm runs alongside global init again")


def test_the_init_gate_is_set_after_the_work_not_during():
    """Setting it mid-cascade would release the catalog thread into
    exactly the concurrent-import window it is meant to avoid."""
    import inspect
    import geeViz.mcp.server as srv
    src = inspect.getsource(srv._ensure_initialized)
    code = "\n".join(ln for ln in src.splitlines()
                     if not ln.lstrip().startswith("#"))
    assert "_init_done.set()" in code, "the gate is never set"
    assert code.index("_ensure_initialized_locked") < code.index("_init_done.set()"), (
        "the gate is set before the init work completes")
