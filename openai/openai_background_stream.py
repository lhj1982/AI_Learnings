from openai import OpenAI
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

client = OpenAI()

# Fire off an async response but also start streaming immediately
stream = client.responses.create(
  model="o3",
  input="Write a very long novel about otters in space.",
  background=True,
  stream=True,
)

cursor = None
for event in stream:
  print(event)
  cursor = event.sequence_number

# If your connection drops, the response continues running and you can reconnect:
# SDK support for resuming the stream is coming soon.
# for event in client.responses.stream(resp.id, starting_after=cursor):
#     print(event)