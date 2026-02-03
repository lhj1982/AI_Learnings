from openai import OpenAI
from dotenv import load_dotenv
# Load environment variables
load_dotenv()

client = OpenAI() 

"""
response = client.responses.create(
    model="gpt-5",
    reasoning={"effort": "low"},
    input=[
        {
            "role": "developer",
            "content": "Talk like a pirate."
        },
        {
            "role": "user",
            "content": "Are semicolons optional in JavaScript?"
        }
    ]
)
"""
response = client.chat.completions.create(
            model="gpt-5",
            messages=[
                {"role": "developer", "content": "Talk like a pirate."},
                {"role": "user", "content": "Are semicolons optional in JavaScript?"}
            ]
        )


print(response.choices[0].message)