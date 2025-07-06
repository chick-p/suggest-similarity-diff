from mcp.server.fastmcp import FastMCP

mcp = FastMCP("similarity-suggest-taro")

@mcp.tool()
def sum(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b

if __name__ == "__main__":
    mcp.run()
