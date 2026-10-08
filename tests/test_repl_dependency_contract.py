"""The run_code REPL has one dependency list, and it lives here.

geeViz and geeViz_agent had drifted to 353 packages against 234, with 117
version mismatches — numpy 2.2.4 vs 2.4.4, pandas 2.2.3 vs 2.3.3 — on the
very libraries an agent's generated code imports. A snippet tested in one
was not the same snippet run in the other.

The cause was two lists: geeViz declared some, the agent's Dockerfile
declared others, and nothing compared them. The fix is that geeViz's
``mcp`` extra IS the REPL contract, and the agent installs ``geeviz[mcp]``
rather than restating it.

These packages are deliberately in an extra rather than in
``[project].dependencies``: nothing in geeViz imports them — verified by
AST across both trees. They exist so USER CODE can import them.
"""
import ast
import re
from pathlib import Path

import pytest

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
PYPROJECT = ROOT / "pyproject.toml"
DOCKERFILE = ROOT / "geeViz_agent" / "Dockerfile"

#: What an agent's generated code is promised. Adding to the REPL means
#: adding here AND to pyproject.toml's ``mcp`` extra; the tests below
#: enforce that the two agree.
REPL_CONTRACT = {
    "scipy", "scikit-learn", "statsmodels",
    "seaborn", "tabulate",
    "geopandas", "pyproj", "pyogrio",
}

_NAME = re.compile(r"[><=\[]")
_SELF = re.compile(r"geeviz\[([^\]]+)\]", re.IGNORECASE)


def _extras():
    """Resolve ``[project.optional-dependencies]`` from pyproject.toml.

    ``all`` is composed from the other extras by self-reference
    (``geeviz[mcp,...]``), deliberately, so it cannot drift out of sync
    the way the old hand-copied literal had. Expand those references so
    each extra lists the requirements it actually pulls in.
    """
    if not PYPROJECT.exists():
        pytest.skip("pyproject.toml not in this checkout")
    extras = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))[
        "project"]["optional-dependencies"]

    def expand(name):
        out = []
        for spec in extras[name]:
            m = _SELF.fullmatch(spec.strip())
            if m:
                for sub in m.group(1).split(","):
                    out += expand(sub.strip())
            else:
                out.append(spec)
        return out

    return {name: expand(name) for name in extras}


def _names(specs):
    return {_NAME.split(s)[0].strip().lower() for s in specs}


def test_the_mcp_extra_is_the_repl_contract():
    missing = REPL_CONTRACT - _names(_extras()["mcp"])
    assert not missing, (
        f"{sorted(missing)} are promised to user code but not declared in "
        f"geeViz's mcp extra")


def test_every_repl_package_has_an_upper_bound():
    """Three outages here came from an unpinned dependency crossing a
    major version on a rebuild: starlette-admin 0.x -> 1.0 twice, and
    kaleido 0.x -> 1.0, which dropped its bundled Chromium and broke
    every chart export at once."""
    unbounded = [
        s for s in _extras()["mcp"]
        if _NAME.split(s)[0].strip().lower() in REPL_CONTRACT and "<" not in s
    ]
    assert not unbounded, f"no upper bound on {unbounded}"


def test_all_extra_is_composed_not_copied():
    """The literal had already drifted — it listed websocket-client and
    none of the packages added beside it."""
    e = _extras()
    assert _names(e["mcp"]) <= _names(e["all"]), (
        "extras['all'] does not contain everything in extras['mcp']")


@pytest.mark.parametrize("pkg", sorted(REPL_CONTRACT))
def test_the_agent_does_not_restate_the_repl_packages(pkg):
    """Two lists is how the drift happened. The agent installs
    geeviz[mcp]; restating a package there lets pip resolve a different
    version for the same REPL."""
    if not DOCKERFILE.exists():
        pytest.skip("agent Dockerfile not in this checkout")
    for line in DOCKERFILE.read_text(encoding="utf-8").splitlines():
        s = line.strip()
        if s.startswith("#"):
            continue
        assert not re.match(rf'"?{re.escape(pkg)}[><=]', s), (
            f"{pkg} is pinned in the agent Dockerfile as well as in "
            f"geeViz's mcp extra")


def test_the_agent_still_installs_the_extra():
    """If this stops being true the REPL loses its libraries entirely.

    Comments are stripped first. The Dockerfile explains the extra in
    prose above the install line, so a naive search matched that comment
    and passed even with ``[mcp,segmentation]`` changed to
    ``[segmentation]`` — caught by mutation testing, in the very file
    meant to enforce this discipline.
    """
    if not DOCKERFILE.exists():
        pytest.skip("agent Dockerfile not in this checkout")
    code = "\n".join(
        ln for ln in DOCKERFILE.read_text(encoding="utf-8").splitlines()
        if not ln.lstrip().startswith("#")
    )
    assert re.search(r"geeviz[^\n]*\[[^]]*\bmcp\b", code), (
        "the agent image no longer installs geeviz[mcp], so the REPL "
        "loses scipy/sklearn/statsmodels/seaborn/tabulate/geopandas")


def test_nothing_in_geeviz_imports_the_repl_packages():
    """They are an extra, not a core dependency, precisely because the
    library does not need them. If geeViz starts importing one it must
    move to ``[project].dependencies``, or a plain ``pip install geeviz``
    breaks.

    Uses AST rather than grep: a first pass with grep matched a docstring
    example (``>>> import seaborn as sns``) and would have moved a
    dependency the library never imports.
    """
    offenders = {}
    for f in (ROOT / "geeViz").rglob("*.py"):
        s = str(f).replace("\\", "/")
        if "/tests/" in s or "generated_scripts" in s or "__pycache__" in s:
            continue
        try:
            tree = ast.parse(f.read_text(encoding="utf-8", errors="replace"))
        except SyntaxError:
            continue
        for n in ast.walk(tree):
            mods = []
            if isinstance(n, ast.Import):
                mods = [a.name.split(".")[0] for a in n.names]
            elif isinstance(n, ast.ImportFrom) and n.module and n.level == 0:
                mods = [n.module.split(".")[0]]
            for m in mods:
                key = "scikit-learn" if m == "sklearn" else m
                if key in REPL_CONTRACT:
                    offenders.setdefault(key, set()).add(f.name)
    assert not offenders, (
        f"geeViz now imports {offenders} — these live in an extra, so a "
        f"plain 'pip install geeviz' would break")
