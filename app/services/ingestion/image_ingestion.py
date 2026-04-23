"""
Image-based food item detection service.

Orchestrates GPT-5.2 vision detection with category normalization
and expiry prediction to produce draft-ready item data.
"""
from dataclasses import dataclass, field
from typing import List, Optional
from datetime import date

from app.services.ingestion.gpt52_vision import gpt52_vision_client, DetectedFoodItem
from app.services.expiry_prediction import expiry_prediction_service


# Default confidence score for GPT-5.2 detections
# User confirmation is the real trust gate, so we use a moderate fixed value
GPT52_DEFAULT_CONFIDENCE = 0.75


@dataclass
class DetectedItemWithPrediction:
    """Food item with expiry prediction, ready for draft creation."""
    name: str
    category: Optional[str]
    predicted_expiry: Optional[str]  # ISO date string
    confidence_score: float
    reasoning: Optional[str]
    quantity: Optional[float] = None
    unit: Optional[str] = None
    quantity_confidence: Optional[float] = None


# Valid units that match the mobile app
VALID_UNITS = {"Pieces", "Grams", "Kilograms", "Milliliters", "Liters"}

# Normalization map for common variations
UNIT_NORMALIZATION_MAP = {
    "pieces": "Pieces", "piece": "Pieces", "pcs": "Pieces", "pc": "Pieces",
    "grams": "Grams", "gram": "Grams", "g": "Grams",
    "kilograms": "Kilograms", "kilogram": "Kilograms", "kg": "Kilograms",
    "milliliters": "Milliliters", "milliliter": "Milliliters", "ml": "Milliliters",
    "liters": "Liters", "liter": "Liters", "l": "Liters",
}


@dataclass
class ImageIngestionResult:
    """Result of image-based food detection."""
    success: bool
    detected_items: List[DetectedItemWithPrediction] = field(default_factory=list)
    error_message: Optional[str] = None


class ImageIngestionService:
    """
    Orchestrates image-based food detection.

    Pipeline:
    1. Send image to GPT-5.2 Vision API
    2. Normalize categories for each detected item
    3. Predict expiry dates using existing prediction service
    4. Return draft-ready data for router to persist
    """

    def ingest_from_image(
        self,
        image_bytes: bytes,
        storage_location: str = "fridge"
    ) -> ImageIngestionResult:
        """
        Detect food items from image and return draft-ready data.

        Args:
            image_bytes: Raw image file bytes
            storage_location: Where items will be stored (fridge, freezer, pantry)

        Returns:
            ImageIngestionResult with detected items and predictions
        """
        # Step 1: Call GPT-5.2 Vision API
        try:
            raw_items = gpt52_vision_client.detect_food_items(image_bytes)
        except RuntimeError as e:
            return ImageIngestionResult(
                success=False,
                error_message=str(e)
            )
        except Exception as e:
            return ImageIngestionResult(
                success=False,
                error_message=f"Unexpected error during image analysis: {str(e)}"
            )

        # Step 2: Check if any items were detected
        if not raw_items:
            return ImageIngestionResult(
                success=False,
                error_message="No food items detected in the image. "
                              "Try taking a clearer photo with better lighting."
            )

        # Step 3: Process each item - normalize category and predict expiry
        processed_items = []
        for item in raw_items:
            normalized_category = self._normalize_category(item.category)

            # Predict expiry using existing service
            prediction = expiry_prediction_service.predict_expiry(
                name=item.name,
                category=normalized_category,
                storage_location=storage_location
            )

                # Validate and normalize unit
            normalized_unit = self._normalize_unit(item.unit)

            processed_items.append(
                DetectedItemWithPrediction(
                    name=item.name,
                    category=normalized_category,
                    predicted_expiry=prediction.expiry_date.isoformat(),
                    confidence_score=GPT52_DEFAULT_CONFIDENCE,
                    reasoning=prediction.reasoning,
                    quantity=item.quantity,
                    unit=normalized_unit,
                    quantity_confidence=item.quantity_confidence
                )
            )

        return ImageIngestionResult(
            success=True,
            detected_items=processed_items
        )

    def _normalize_category(self, category: Optional[str]) -> Optional[str]:
        """
        Normalize GPT-5.2 category to SnapShelf category.

        Maps to categories used by expiry prediction rules.
        """
        if not category:
            return None

        category_lower = category.lower().strip()

        # Direct mappings for GPT-5.2 categories
        # These should match the categories in our prediction rules
        category_map = {
            "dairy": "dairy",
            "meat": "meat",
            "poultry": "poultry",
            "fish": "fish",
            "seafood": "seafood",
            "vegetables": "vegetables",
            "fruits": "fruits",
            "bread": "bakery",
            "bakery": "bakery",
            "eggs": "eggs",
            "condiments": "condiments",
            "beverages": "beverages",
            "snacks": "snacks",
            "frozen": "frozen",
            "canned": "canned",
            "other": None,  # No specific category, use fallback prediction
        }

        return category_map.get(category_lower, category_lower)

    def _normalize_unit(self, unit: Optional[str]) -> Optional[str]:
        """
        Normalize and validate unit from GPT-5.2.

        Returns the normalized unit if valid, None otherwise.
        """
        if not unit:
            return None

        # Check if already valid
        if unit in VALID_UNITS:
            return unit

        # Try normalization map
        unit_lower = unit.lower().strip()
        return UNIT_NORMALIZATION_MAP.get(unit_lower, None)


# Singleton instance
image_ingestion_service = ImageIngestionService()
