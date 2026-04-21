"""
GPT-5.2 Vision API client for food item detection.

Uses OpenAI's GPT-5.2 model to analyze images and detect food items.
"""
import base64
import json
from dataclasses import dataclass
from typing import List, Optional

from openai import OpenAI

from app.core.config import get_openai_api_key


@dataclass
class DetectedFoodItem:
    """Single food item detected in an image."""
    name: str
    category: Optional[str] = None
    quantity: Optional[float] = None
    unit: Optional[str] = None
    quantity_confidence: Optional[float] = None  # 0.0-1.0


# Prompt for GPT-5.2 to detect food items
DETECTION_PROMPT = """Analyze this image and identify all visible food items.

For each food item you can clearly identify, provide:
- name: specific food name (e.g., "whole milk", "chicken breast", "romaine lettuce")
- category: one of these categories ONLY (use exact capitalization):
  - Fruits (apples, bananas, oranges, berries, etc.)
  - Vegetables (lettuce, tomatoes, carrots, onions, etc.)
  - Dairy (milk, cheese, yogurt, butter, eggs, etc.)
  - Meat (beef, pork, chicken, turkey, lamb, etc.)
  - Fish (salmon, tuna, cod, shrimp, seafood, etc.)
  - Grains (pasta, rice, bread, cereals, oats, gnocchi, noodles, flour, etc.)
  - Snacks (chips, cookies, crackers, candy, etc.)
  - Beverages (juice, soda, water, coffee, tea, etc.)
  - Frozen (ice cream, frozen meals, frozen vegetables, etc.)
  - Condiments (ketchup, mustard, mayo, sauces, spices, etc.)
  - Other (anything that doesn't fit above)
- quantity: estimated amount (number only, e.g., 1, 500, 2.5)
- unit: one of these units ONLY (use exact capitalization):
  - Pieces (for countable items: apples, eggs, bottles, cans, packages)
  - Grams (for small weight items: deli meat, cheese slices, small produce)
  - Kilograms (for larger weight items: whole chicken, large bags)
  - Milliliters (for small liquid volumes: cream, small sauce bottles)
  - Liters (for larger liquid volumes: milk cartons, juice bottles)
- quantity_confidence: your confidence in the quantity estimate (0.0 to 1.0)

Quantity detection (in order of priority):
1. VISIBLE ON PACKAGE: If quantity is printed on packaging (e.g., "500g", "1L", "6 eggs"), use that exact value. Set quantity_confidence to 0.95+
2. COUNTABLE: If you can count individual items (apples, bottles, cans), count them as Pieces. Set confidence based on clarity.
3. ESTIMATE: Only if no visible quantity and not countable, estimate based on typical sizes:
   - Milk carton: 1 Liters | Yogurt cup: 125 Grams | Soda can: 330 Milliliters
   - Cheese block: 200-300 Grams | Deli meat: 150 Grams
   Set quantity_confidence to 0.5-0.7 for estimates.
4. If you cannot determine quantity at all, set quantity, unit, and quantity_confidence to null

Rules:
- Only include food items you can clearly identify
- Be specific with names (e.g., "cheddar cheese" not just "cheese")
- Use the EXACT category and unit names shown above (capitalized)
- If you cannot determine the category, use "Other"
- Do not include non-food items
- Quantity estimates are suggestions - users will verify

Return a JSON object with this exact structure:
{"items": [{"name": "item name", "category": "Category", "quantity": 2, "unit": "Pieces", "quantity_confidence": 0.9}]}

Example output:
{"items": [
  {"name": "whole milk", "category": "Dairy", "quantity": 1, "unit": "Liters", "quantity_confidence": 0.85},
  {"name": "red apples", "category": "Fruits", "quantity": 4, "unit": "Pieces", "quantity_confidence": 0.95},
  {"name": "cheddar cheese", "category": "Dairy", "quantity": 200, "unit": "Grams", "quantity_confidence": 0.6}
]}

If no food items are visible, return: {"items": []}"""


class GPT4oVisionClient:
    """
    Client for GPT-5.2 Vision API.

    Handles image encoding, API calls, and response parsing
    for food item detection.
    """

    def __init__(self):
        """Initialize the OpenAI client."""
        self._client: Optional[OpenAI] = None

    @property
    def client(self) -> OpenAI:
        """Lazy initialization of OpenAI client."""
        if self._client is None:
            self._client = OpenAI(api_key=get_openai_api_key())
        return self._client

    def detect_food_items(self, image_bytes: bytes) -> List[DetectedFoodItem]:
        """
        Detect food items in an image using GPT-5.2.

        Args:
            image_bytes: Raw image bytes (JPEG, PNG, etc.)

        Returns:
            List of detected food items with names and categories

        Raises:
            ValueError: If image cannot be processed
            RuntimeError: If API call fails
        """
        # Encode image to base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")

        # Determine image type (default to jpeg)
        image_type = self._detect_image_type(image_bytes)

        try:
            response = self.client.chat.completions.create(
                model="gpt-5.2",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": DETECTION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/{image_type};base64,{base64_image}",
                                    "detail": "low"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=800,
            )
        except Exception as e:
            raise RuntimeError(f"GPT-5.2 API error: {str(e)}")

        # Parse response
        content = response.choices[0].message.content
        if not content:
            return []

        try:
            result = json.loads(content)
            items = result.get("items", [])

            return [
                DetectedFoodItem(
                    name=item.get("name", "unknown"),
                    category=item.get("category"),
                    quantity=item.get("quantity"),
                    unit=item.get("unit"),
                    quantity_confidence=item.get("quantity_confidence")
                )
                for item in items
                if item.get("name")
            ]
        except json.JSONDecodeError as e:
            raise RuntimeError(f"Failed to parse GPT-5.2 response: {str(e)}")

    def _detect_image_type(self, image_bytes: bytes) -> str:
        """
        Detect image type from magic bytes.

        Args:
            image_bytes: Raw image bytes

        Returns:
            Image type string (jpeg, png, gif, webp)
        """
        if image_bytes[:3] == b"\xff\xd8\xff":
            return "jpeg"
        elif image_bytes[:8] == b"\x89PNG\r\n\x1a\n":
            return "png"
        elif image_bytes[:6] in (b"GIF87a", b"GIF89a"):
            return "gif"
        elif image_bytes[:4] == b"RIFF" and image_bytes[8:12] == b"WEBP":
            return "webp"
        else:
            # Default to jpeg
            return "jpeg"


# Singleton instance
gpt4o_vision_client = GPT4oVisionClient()
