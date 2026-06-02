from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    llm_provider: str = "gapgpt"
    llm_api_key: str = ""
    llm_model: str = "gpt-4o"
    llm_base_url: str = "https://api.gapgpt.app/v1"

    embedding_model: str = "all-MiniLM-L6-v2"
    chroma_persist_dir: str = "app/data/chroma"
    raw_data_dir: str = "app/data/raw"

    top_k: int = 4
    chunk_size: int = 700
    chunk_overlap: int = 100

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
