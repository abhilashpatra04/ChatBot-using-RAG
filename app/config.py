from pydantic_settings import BaseSettings
from pydantic import Field

class Settings(BaseSettings):
    google_client_id: str | None = Field(default=None, alias="GOOGLE_CLIENT_ID")
    google_client_secret: str | None = Field(default=None, alias="GOOGLE_CLIENT_SECRET")
    google_oauth_redirect: str | None = Field(default=None, alias="GOOGLE_REDIRECT_URI")
    session_secret: str = Field(default="dev-secret", alias="SESSION_SECRET")
    gemini_api_key: str | None = Field(default=None, alias="GEMINI_API_KEY")
    gemini_model: str = Field(default="gemini-1.5-pro", alias="GEMINI_MODEL")
    groq_api_key: str | None = Field(default=None, alias="GROQ_API_KEY")
    groq_model: str = Field(default="openai/gpt-oss-20b", alias="GROQ_MODEL")

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
