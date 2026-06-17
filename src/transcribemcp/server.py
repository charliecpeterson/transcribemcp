from mcp.server.fastmcp import FastMCP

mcp = FastMCP("transcribemcp")


@mcp.tool()
def ping() -> dict:
    """Sanity check that the MCP server is reachable."""
    from . import __version__
    return {"ok": True, "version": __version__}
