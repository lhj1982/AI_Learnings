import time

from openai import OpenAI

from app.config import settings
from app.metrics import (
    AI_INPUT_TOKENS,
    AI_OUTPUT_TOKENS,
    AI_REQUEST_LATENCY,
    AI_REQUESTS,
)


class LLMProvider:

    def __init__(
        self,
        provider_name: str,
        base_url: str,
        api_key: str,
        model: str,
    ):
        self.provider_name = provider_name
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key,
        )
        self.model = model

    def chat(self, message: str) -> str:
        start_time = time.perf_counter()

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {
                        "role": "user",
                        "content": message,
                    }
                ],
            )

            duration = time.perf_counter() - start_time

            # Request succeeded
            AI_REQUESTS.labels(
                provider=self.provider_name,
                model=self.model,
                status="success",
            ).inc()

            AI_REQUEST_LATENCY.labels(
                provider=self.provider_name,
                model=self.model,
            ).observe(duration)

            # Token usage
            if response.usage:
                AI_INPUT_TOKENS.labels(
                    provider=self.provider_name,
                    model=self.model,
                ).inc(response.usage.prompt_tokens or 0)

                AI_OUTPUT_TOKENS.labels(
                    provider=self.provider_name,
                    model=self.model,
                ).inc(response.usage.completion_tokens or 0)

            return response.choices[0].message.content

        except Exception:
            duration = time.perf_counter() - start_time

            # Request failed
            AI_REQUESTS.labels(
                provider=self.provider_name,
                model=self.model,
                status="error",
            ).inc()

            AI_REQUEST_LATENCY.labels(
                provider=self.provider_name,
                model=self.model,
            ).observe(duration)

            raise


providers = {
    "deepseek": LLMProvider(
        provider_name="deepseek",
        base_url="https://api.deepseek.com",
        api_key=settings.deepseek_api_key,
        model="deepseek-flash",
    ),

    "qwen": LLMProvider(
        provider_name="qwen",
        base_url=settings.llm_base_url,
        api_key=settings.llm_api_key,
        model="Qwen/Qwen2.5-1.5B-Instruct",
    ),
}