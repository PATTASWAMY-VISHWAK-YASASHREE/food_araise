import asyncio
import time
import httpx
from loguru import logger
from app.config import settings
from async_lru import alru_cache

class SerpAPIWrapper:
    def __init__(self):
        self.api_key = settings.SERPAPI_API_KEY
        self.base_url = "https://serpapi.com/search"
        self._lock = asyncio.Lock()
        self._last_request_ts = 0.0
        self.min_interval = max(0.0, settings.SERPAPI_MIN_INTERVAL)
        self.max_retries = max(0, settings.SERPAPI_MAX_RETRIES)
        self.backoff_factor = max(1.0, settings.SERPAPI_BACKOFF_FACTOR)

    async def _throttle(self):
        """Ensure a minimum delay between outbound SerpAPI calls."""
        async with self._lock:
            now = time.monotonic()
            wait = self.min_interval - (now - self._last_request_ts)
            if wait > 0:
                await asyncio.sleep(wait)
            self._last_request_ts = time.monotonic()

    @alru_cache(maxsize=128) # lightweight RAM-efficient caching
    async def search_food_info(self, query: str) -> dict:
        """
        Searches for food information using SerpAPI.
        Async + Cached.
        """
        if not self.api_key:
            logger.warning("SerpAPI Key is missing. Skipping web search.")
            return {"error": "API Key missing"}

        logger.info(f"Searching SerpAPI for: {query}")
        
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "hl": "en",
            "gl": "us"
        }

        try:
            # Throttled, retried request to reduce 429s
            backoff = self.min_interval
            for attempt in range(self.max_retries + 1):
                await self._throttle()
                try:
                    async with httpx.AsyncClient(timeout=10.0) as client:
                        response = await client.get(self.base_url, params=params)
                    if response.status_code == 429:
                        if attempt < self.max_retries:
                            backoff *= self.backoff_factor
                            logger.warning(f"SerpAPI 429 received; backing off {backoff:.2f}s (attempt {attempt+1}/{self.max_retries})")
                            await asyncio.sleep(backoff)
                            continue
                        response.raise_for_status()
                    response.raise_for_status()
                    data = response.json()
                    break
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429 and attempt < self.max_retries:
                        backoff *= self.backoff_factor
                        logger.warning(f"SerpAPI 429 received; backing off {backoff:.2f}s (attempt {attempt+1}/{self.max_retries})")
                        await asyncio.sleep(backoff)
                        continue
                    raise
            
            # RAM Efficiency: Extract only what's needed, discard the rest
            results = {
                "snippets": [],
                "knowledge_graph": data.get("knowledge_graph", {})
            }

            if "organic_results" in data:
                for res in data["organic_results"][:3]: # Limit to top 3
                    results["snippets"].append({
                        "title": res.get("title"),
                        "snippet": res.get("snippet"),
                        "link": res.get("link")
                    })
            
            return results

        except httpx.HTTPError as e:
            logger.error(f"SerpAPI request failed: {e}")
            return {"error": str(e)}
        except Exception as e:
            logger.error(f"Unexpected search error: {e}")
            return {"error": str(e)}

search_client = SerpAPIWrapper()
