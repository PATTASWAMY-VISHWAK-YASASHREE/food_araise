import os
from typing import List
from pydantic_settings import BaseSettings
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Settings(BaseSettings):
    # API Keys
    GOOGLE_API_KEY: str = os.getenv("GOOGLE_API_KEY", "")
    SERPAPI_API_KEY: str = os.getenv("SERPAPI_API_KEY", "")

    # Application Settings
    APP_NAME: str = "Food Agent Server"
    DEBUG: bool = True


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

    class Config:
        env_file = ".env"

settings = Settings()