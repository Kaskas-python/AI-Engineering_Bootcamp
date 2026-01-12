from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):

    API_URL: str = "http://api:8100"

    model_config = SettingsConfigDict(env_file=".env")

config = Config()