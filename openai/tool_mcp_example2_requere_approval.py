from openai import OpenAI
import pprint

from dotenv import load_dotenv
# Load environment variables
load_dotenv()

client = OpenAI()

resp = client.responses.create(
    model="gpt-5",
    tools=[{
        "type": "mcp",
        "server_label": "dmcp",
        "server_description": "A Dungeons and Dragons MCP server to assist with dice rolling.",
        "server_url": "https://dmcp-server.deno.dev/sse",
        "require_approval": "always",
    }],
    input="Roll 2d4+1",
    # previous_response_id="resp_682d498bdefc81918b4a6aa477bfafd904ad1e533afccbfa",
    # input=[{
    #     "type": "mcp_approval_response",
    #     "approve": True,
    #     "approval_request_id": "mcpr_682d498e3bd4819196a0ce1664f8e77b04ad1e533afccbfa"
    # }],
)

# If resp is a dict or has a 'message' attribute
if hasattr(resp, "message"):
    message = resp.message
elif isinstance(resp, dict) and "message" in resp:
    message = resp["message"]
else:
    message = None

if message and hasattr(message, "tool_calls"):
    for tool_call in message.tool_calls:
        if getattr(tool_call, "type", None) == "mcp_approval_request":
            print("Approval Request Content:", getattr(tool_call, "content", tool_call))
elif isinstance(resp, dict):
    pprint.pprint(resp)
else:
    print("Unknown response structure:", type(resp))

print(resp)