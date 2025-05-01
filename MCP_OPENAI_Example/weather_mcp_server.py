from typing import List
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("Weather")

@mcp.tool()
async def get_weather(location: str) -> str:
    """Get weather for location."""
    return "It's always sunny in New York in the morning, but cloudy in the afternoon"


# @mcp.prompt()
# def get_weather(location: str) -> str:
#     return f"Please get the today's weather of {location}"

if __name__ == "__main__":
    mcp.run(transport="sse")
