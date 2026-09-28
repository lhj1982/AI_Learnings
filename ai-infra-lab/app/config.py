from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    llm_provider: str = "deepseek"
    llm_base_url: str
    llm_api_key: str
    llm_model: str
    deepseek_api_key: str

    class Config:
        env_file = ".env"


settings = Settings()