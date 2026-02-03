from openai import OpenAI
from time import sleep
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

client = OpenAI()

resp = client.responses.create(
  model="o3",
  input="Write a very long novel about otters in space.",
  background=True,
)

while resp.status in {"queued", "in_progress"}:
  print(f"Current status: {resp.status}")
  sleep(2)
  resp = client.responses.retrieve(resp.id)

print(f"Final status: {resp.status}\nOutput:\n{resp.output_text}")