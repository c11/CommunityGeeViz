"""FIA-only entry point for the geeViz MCP server."""

import asyncio

from . import server

_ALLOWED_TOOLS = frozenset({
    "search_fia",
    "fia_estimate",
})


async def _restrict_tools() -> None:
    tools = await server.app.list_tools()

    for tool in tools:
        if tool.name not in _ALLOWED_TOOLS:
            server.app.remove_tool(tool.name)


def main() -> None:
    asyncio.run(_restrict_tools())
    server.main()


if __name__ == "__main__":
    main()