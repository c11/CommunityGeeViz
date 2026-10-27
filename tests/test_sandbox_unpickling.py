"""Unpickling is arbitrary code execution, and the sandbox hands out a
file-write primitive.

Found while auditing whether scipy / scikit-learn / statsmodels could be
added safely. They can — but the audit turned up an escape that needed
NONE of them, using only what the sandbox already provides:

    save_file('x.pkl', <bytes whose __reduce__ calls anything>)
    pd.read_pickle(path)              -> attacker code ran
    np.load(path, allow_pickle=True)  -> attacker code ran

Both halves are legitimate in isolation. ``save_file`` exists so the
agent can hand the user a GIF or a report; pandas and numpy are the
point of the REPL. The ``pickle`` MODULE was already blocked, which is
what made this look covered — the hole is that several available
libraries will unpickle on your behalf from a trusted stack frame, so
neither the import blocklist nor the audit hook ever sees it.

Blocking the READ side is the narrow fix. Nothing in a geospatial
analysis needs to unpickle a file the same session just wrote.
"""
import ast
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SRC = (ROOT / "mcp" / "server.py").read_text(encoding="utf-8")


def _code_only(text):
    out = []
    for line in text.splitlines():
        in_s, q, buf = False, "", []
        for i, ch in enumerate(line):
            if in_s:
                buf.append(ch)
                if ch == q and (i == 0 or line[i - 1] != "\\"):
                    in_s = False
            elif ch in "\"'":
                in_s, q = True, ch
                buf.append(ch)
            elif ch == "#":
                break
            else:
                buf.append(ch)
        out.append("".join(buf))
    s = "\n".join(out)
    try:
        tree = ast.parse(s)
    except SyntaxError:
        return s
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef,
                             ast.ClassDef, ast.Module)):
            b = getattr(node, "body", None)
            if (b and isinstance(b[0], ast.Expr)
                    and isinstance(b[0].value, ast.Constant)
                    and isinstance(b[0].value.value, str)):
                s = s.replace(b[0].value.value, "")
    return s


CODE = _code_only(SRC)


def test_the_stripper_works():
    assert "read_pickle" not in _code_only("x = 1  # read_pickle danger\n")


def test_unpickling_call_names_are_blocked():
    assert "_UNPICKLING_CALLS" in CODE, "no deserialization blocklist exists"
    m = re.search(r"_UNPICKLING_CALLS = frozenset\(\{(.*?)\}\)", CODE, re.S)
    assert m, "could not read the blocklist"
    names = set(re.findall(r'"([a-z_]+)"', m.group(1)))
    assert "read_pickle" in names, "pandas read_pickle is not blocked"


def test_the_check_is_wired_into_the_ast_pass():
    """A blocklist nothing consults is decoration."""
    assert "_fname in _UNPICKLING_CALLS" in CODE, (
        "the blocklist is defined but never consulted")


def test_allow_pickle_true_is_refused():
    """np.load is legitimate; np.load(allow_pickle=True) is not."""
    assert 'allow_pickle' in CODE
    assert re.search(r'_kw\.arg == "allow_pickle"', CODE), (
        "allow_pickle= is not inspected on load() calls")


def test_allow_pickle_false_is_still_permitted():
    """Explicitly passing False is the safe form and must not be broken —
    the check has to look at the VALUE, not just the keyword's presence."""
    assert "_kw.value.value is False" in CODE, (
        "the guard refuses allow_pickle=False as well, which blocks the "
        "safe spelling people write to be explicit")


def test_the_guard_is_sandbox_scoped():
    """Local/stdio use is unrestricted by design; this must not leak into
    it and break someone's notebook."""
    i = CODE.index("_fname in _UNPICKLING_CALLS")
    window = CODE[max(0, i - 700):i]
    assert "_SANDBOX_ENABLED" in window, (
        "the deserialization guard is not gated on sandbox mode")


def test_pickle_module_is_still_blocked_too():
    """Defense in depth: the module blocklist is the first door, this is
    the second. Removing either leaves the other."""
    assert '"pickle"' in CODE


# ── the HTTP stack under requests/urllib ───────────────────────────────
#
# Asked directly: "are urllib3, or the agent making its own package, a
# possibility?" Both were tested against a live sandboxed run_code.
#
# urllib3: YES, it was a full escape. `import requests` and `import
# urllib.request` were both refused while
#
#     urllib3.PoolManager().request('GET', 'http://example.com')
#
# returned status 200 with 559 bytes. The blocklist matched on the
# TOP-LEVEL module name, and "urllib3" != "urllib", so blocking the
# friendly wrappers left the transport they sit on wide open. httpx,
# aiohttp, httplib2 and websocket were equally importable — all present
# as transitive dependencies nobody chose.
#
# Writing its own package: NO. save_file reduces the name with
# os.path.basename so it cannot traverse, the output directory is not on
# sys.path, and `sys` is blocked so it cannot be put there. Verified:
# `import evilmod` -> ModuleNotFoundError, and the sys.path workaround
# was refused at the import of `sys`.

def _frozenset_names(var):
    m = re.search(rf"{var} = frozenset\(\{{(.*?)\}}\)", CODE, re.S)
    assert m, f"could not find {var}"
    return set(re.findall(r'"([a-zA-Z0-9_]+)"', m.group(1)))


# httpx2 / httpcore2 arrived with FastMCP 4: httpx and httpcore under
# new names, so the same full outbound client.
HTTP_STACK = ["urllib3", "httpx", "httpcore", "httpx2", "httpcore2",
              "aiohttp", "httplib2", "websocket", "websockets"]


@pytest.mark.parametrize("mod", HTTP_STACK)
def test_the_real_http_stack_is_blocked_for_user_code(mod):
    assert mod in _frozenset_names("_BLOCKED_MODULES"), (
        f"{mod} is importable from run_code — blocking `requests` while "
        f"leaving {mod} open blocks nothing")


@pytest.mark.parametrize("mod", HTTP_STACK)
def test_the_real_http_stack_is_blocked_at_the_audit_hook(mod):
    """Second layer. The file states these two lists MUST agree — an
    attacker who gets past one still has to clear the other."""
    if mod in ("websockets",):          # audit list omits a few aliases
        return
    assert mod in _frozenset_names("_AUDIT_BLOCKED_IMPORTS"), (
        f"{mod} is blocked for user code but not at the audit hook")


def test_the_two_blocklists_do_not_drift():
    """The comment at the top of _BLOCKED_MODULES promises they are kept
    in sync. Anything in the audit list but NOT the module list would be
    reachable by a plain import statement."""
    mods = _frozenset_names("_BLOCKED_MODULES")
    audit = _frozenset_names("_AUDIT_BLOCKED_IMPORTS")
    missing = sorted(audit - mods)
    assert not missing, (
        f"audit hook blocks {missing} but the module blocklist does not")


def test_save_file_cannot_escape_its_directory():
    """The self-written-package route depends on landing a .py somewhere
    importable. basename() is what stops it."""
    i = CODE.index("def save_file") if "def save_file" in CODE else CODE.index("safe_name")
    window = CODE[max(0, i - 400):i + 400]
    assert "os.path.basename" in window, (
        "save_file no longer strips directory components, so a written "
        "file could land on sys.path")


# ── reaching a blocked module through another module ───────────────────
#
# Asked: "what about gil.os, gil.blah — a blacklisted package reached
# through one that is allowed?" Yes, that worked.
#
# Any module doing `import os` at the top re-exports it, so gil.os,
# gv.os and cl.os are all the real os module. The import blocklist only
# inspects `import` statements, so it never saw them. Verified against a
# live sandbox BEFORE the fix:
#
#   gv.os.environ    -> listed GEMINI_API_KEY and
#                       GOOGLE_MAPS_PLATFORM_API_KEY, values present
#   cl.os.listdir(.) -> real directory listing
#   gil.os.getcwd()  -> real path
#
# gil.os.system() was already refused, but only because the audit hook
# watches that one SYSCALL. Reading os.environ raises no audit event, so
# nothing fired.
#
# The list is deliberately NARROWER than _BLOCKED_MODULES: `select` is
# ee.Image.select (the most used call in the codebase), `signal` is
# scipy.signal, `code` is a plausible response field. Blocking those
# would break real work, so they are excluded by design.

def test_module_reexports_are_blocked_as_attributes():
    names = _frozenset_names("_BLOCKED_MODULE_ATTRS")
    for mod in ("os", "sys", "subprocess", "socket", "requests", "urllib3",
                "pickle", "importlib", "ctypes"):
        assert mod in names, f".{mod} is reachable through another module"


def test_the_attr_list_does_not_break_ordinary_analysis_code():
    """Guards the other direction. Adding these would make
    ee.Image.select() and scipy.signal unusable, which is a worse
    outcome than the hole."""
    names = _frozenset_names("_BLOCKED_MODULE_ATTRS")
    for legit in ("select", "signal", "code", "resource", "glob", "io"):
        assert legit not in names, (
            f"'.{legit}' is blocked as an attribute — that breaks "
            f"legitimate calls (ee.Image.select, scipy.signal, ...)")


def test_the_attr_check_is_wired_in():
    assert "_BLOCKED_MODULE_ATTRS" in CODE
    assert "node.attr in _BLOCKED_MODULE_ATTRS" in CODE, (
        "the attribute blocklist is defined but never consulted")


def test_getattr_stays_blocked_so_the_static_check_cannot_be_dodged():
    """The AST check only sees `x.os`. If getattr() were available,
    getattr(gil, 'os') would walk straight past it."""
    assert "getattr" in _frozenset_names("_BLOCKED_BUILTINS")


# ── the allowlist that is not one ──────────────────────────────────────
#
# _ALLOWED_MODULE_PREFIXES is defined, extended by
# MCP_EXTRA_ALLOWED_MODULES, and never consulted. Verified against a live
# sandbox: uuid, hashlib, base64, csv, random, typing and warnings are
# all off the list and all import fine. It is also why scipy and sklearn
# worked the instant they were installed.
#
# This test does not demand it be enforced — that is a live design
# decision, and enforcing it would make the agent MORE constrained, which
# is the opposite of what is wanted. It demands only that the file not
# claim a control it does not have, so nobody sets
# MCP_EXTRA_ALLOWED_MODULES believing it does something.

def test_the_inert_allowlist_is_labelled_as_inert():
    i = CODE.index("_ALLOWED_MODULE_PREFIXES = (")
    header = SRC[max(0, SRC.index("_ALLOWED_MODULE_PREFIXES = (") - 1600):
                 SRC.index("_ALLOWED_MODULE_PREFIXES = (")]
    assert "NOT ENFORCED" in header, (
        "_ALLOWED_MODULE_PREFIXES reads as an active allowlist but nothing "
        "consults it — say so, or wire it up")
    assert i > 0


def test_it_is_still_genuinely_unreferenced():
    """If someone wires it up, this fails and the comment above must be
    rewritten — the label would then be a lie in the other direction."""
    uses = [ln for ln in CODE.splitlines()
            if "_ALLOWED_MODULE_PREFIXES" in ln]
    # definition + the _EXTRA_ALLOWED extension, nothing else
    assert len(uses) <= 2, (
        f"the allowlist is now referenced {len(uses)} times — if it is "
        f"enforced, remove the NOT ENFORCED banner")
