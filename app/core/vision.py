from app.core.intelligence import gemini_client
from fastapi import HTTPException
from app.core.search import search_client
from loguru import logger
import json
import re
import asyncio
import io
from PIL import Image

from app.core.local_intelligence import local_client

async def analyze_food_image_with_search(image_bytes: bytes, deep_search: bool = False):
    """
    Orchestrates the Identify -> Search -> Synthesize workflow.
    Optimized for RAM (resizing) and Speed (Parallel Search).
    """
    try:
        # RAM Efficiency: Resize image if too large
        processed_image = _resize_image(image_bytes)
        
        # Step 1: Identify Items & Estimate Portions
        logger.info("Step 1: Identifying food items and estimating portions...")
        identify_prompt = (
            "Analyze this food image. Identify each distinct food item and ESTIMATE its portion size "
            "(e.g., '1 cup', '200g', '1 large slice'). "
            "Return ONLY a list in this format: Item Name (Portion Size). "
            "Example: 'Cheeseburger (1 item), French Fries (medium portion), Coke (330ml)'."
        )
        identification_result = gemini_client.generate_content(identify_prompt, processed_image)
        
        cleaned_result = identification_result.replace('\n', ',')
        items_list = [item.strip() for item in cleaned_result.split(',') if item.strip()]
        logger.info(f"Identified items: {items_list}")

        # Step 2: Search for Information (Parallel / Async)
        logger.info("Step 2: Searching for items in parallel...")
        
        async def fetch_info(item):
            query = f"{item} calories nutrition facts"
            info = await search_client.search_food_info(query)
            if info and "error" not in info:
                return f"Item Search: {item}\nData: {json.dumps(info)}"
            return None

        # Threading/Concurrency: Run all searches at once
        search_results = await asyncio.gather(*[fetch_info(item) for item in items_list])
        context_str = "\n".join([res for res in search_results if res])

        # Step 3: Synthesize Final Response using Chain of Thought
        logger.info("Step 3: Synthesizing final response with advanced scripting...")
        final_prompt = f"""
        You are an expert AI Dietitian and Nutritionist with advanced spatial reasoning capabilities.
        Your task is to analyze the provided image and context to produce a highly accurate nutritional assessment.

        **CONTEXT FROM WEB SEARCH (Use to ground your specific calculations):**
        {context_str}

        **INSTRUCTIONS - THINK STEP-BY-STEP:**
        1. **Scene & Scale Analysis**: 
           - Identify any reference objects (utensils, plates, table grain) to establish scale. 
           - Estimate the plate diameter if visible (standard is ~10-12 inches).
           - Account for 3D depth and stacking of food items.
        
        2. **Item Identification & Segmentation**:
           - List every distinct edible item. 
           - Differentiate between sauces, garnishes, and main components.

        3. **Volumetric Estimation**:
           - For each item, estimate its volume (e.g., "cup", "cubic inches", "tablespoons").
           - Convert this volume to Weight (grams) using standard density factors (e.g., 1 cup rice ~= 158g).
           - *Explicitly state your logic for these conversions in the description.*

        4. **Nutritional Calculation**:
           - Use the provided Web Search Context for "per 100g" values.
           - Multiply by your estimated weight.
           - Sum up totals.

        5. **Final Output Generation**:
           - Return the result in specific JSON format.
        
        **REQUIRED JSON OUTPUT FORMAT**:
        {{
            "overall_description": "A detailed 2-3 sentence summary of the meal and portion sizes.",
            "items": [
                {{
                    "name": "Food Name (e.g. Grilled Chicken)",
                    "estimated_portion": "e.g. 6 oz (approx 170g)",
                    "confidence": 0.95,
                    "description": "Visual reasoning: Covers 1/3 of 10-inch plate, thickness approx 1 inch.",
                    "nutrition": {{
                        "calories": "Calculated Value",
                        "protein": "...",
                        "carbs": "...",
                        "fats": "...",
                        "vitamins": ["Vit A", "B12"]
                    }},
                    "search_insights": {{ "source": "USDA data for [Search Term]" }}
                }}
            ],
            "total_calories_estimate": "Sum numeric string (e.g. '850 kcal')",
            "health_score": 8,
            "dietary_warnings": ["Allergen 1", "High Sodium", etc.]
        }}
        
        RETURN ONLY RAW JSON. NO MARKDOWN.
        """
        
        raw_response = gemini_client.generate_content(final_prompt, processed_image)
        cleaned_json = _clean_json_string(raw_response)
        
        return json.loads(cleaned_json)

    except Exception as e:
        logger.warning(f"Cloud API/Search failed: {e}. Switching to Local Failsafe. Deep Search: {deep_search}")
        try:
            # Fallback to Local Intelligence
            return local_client.analyze_image(image_bytes, deep_search=deep_search)
        except Exception as local_e:
            logger.critical(f"Local Failsafe also failed: {local_e}")
            raise HTTPException(status_code=500, detail="All AI systems failed.")

def _resize_image(image_bytes: bytes, max_size=(1024, 1024)) -> bytes:
    """
    RAM Efficient: Resizes large images to max_size.
    Returns bytes of the resized image.
    """
    try:
        with Image.open(io.BytesIO(image_bytes)) as img:
            img.thumbnail(max_size)
            # Convert back to bytes
            buf = io.BytesIO()
            # Save as JPEG for compression/speed
            img.convert("RGB").save(buf, format="JPEG", quality=85)
            return buf.getvalue()
    except Exception as e:
        logger.warning(f"Image resize failed, using original: {e}")
        return image_bytes

def _clean_json_string(s: str) -> str:
    """Removes markdown code blocks and whitespace."""
    s = s.strip()
    pattern = r"^```json\s*(.*?)\s*```$"
    match = re.search(pattern, s, re.DOTALL)
    if match:
        s = match.group(1)
    return s