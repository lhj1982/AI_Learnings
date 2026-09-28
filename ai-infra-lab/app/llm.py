from openai import OpenAI

from app.config import settings


class LLMClient:

    def __init__(self):
        self.client = OpenAI(
            base_url=settings.llm_base_url,
            api_key=settings.llm_api_key,
        )

        self.model = settings.llm_model

    def chat(self, message: str) -> str:

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful AI assistant."
                },
                {
                    "role": "user",
                    "content": message
                }
            ],
        )

        return response.choices[0].message.content


llm = LLMClient()