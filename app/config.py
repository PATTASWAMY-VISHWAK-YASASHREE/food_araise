from typing import List
from pydantic_settings import BaseSettings, SettingsConfigDict
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # API Keys
    GOOGLE_API_KEY: str = ""
    SERPAPI_API_KEY: str = ""
    SERPAPI_MIN_INTERVAL: float = 3.6  # seconds between requests
    SERPAPI_MAX_RETRIES: int = 2
    SERPAPI_BACKOFF_FACTOR: float = 1.5

    # Application Settings
    APP_NAME: str = "Food Agent Server"
    DEBUG: bool = True
    LOG_LEVEL: str = "INFO"

    # Local Failsafe Models
    LOCAL_MODEL_LIGHT_REPO: str = "moondream/moondream2-gguf"
    LOCAL_MODEL_LIGHT_FILENAME: str = "moondream2-q4_k.gguf" 
    
    LOCAL_MODEL_HEAVY_REPO: str = "abetlen/Phi-3.5-vision-instruct-gguf"
    LOCAL_MODEL_HEAVY_FILENAME: str = "Phi-3.5-vision-instruct-Q4_K_M.gguf"
    
    LOCAL_MODELS_DIR: str = "local_models"

    # AI Model Configuration
    # Prioritized list based on research (Nov 2025 availability)
    GEMINI_MODELS: List[str] = [
        "gemini-3-pro-preview", # Latest reasoning model
        "gemini-2.5-pro",       # Stable high-capability
        "gemini-2.5-flash",     # Fast and efficient
        "gemini-2.0-flash-exp", # Experimental features
        "gemini-1.5-pro",       # Validated fallback
        "gemini-1.5-flash"
    ]


settings = Settings()