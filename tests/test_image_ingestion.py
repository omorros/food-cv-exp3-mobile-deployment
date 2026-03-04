"""
Unit tests for image ingestion service.

Tests GPT-5.2 vision integration, category normalization,
and expiry prediction pipeline.
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import date, timedelta

from app.services.ingestion.gpt4o_vision import (
    GPT4oVisionClient,
    DetectedFoodItem,
    DETECTION_PROMPT
)
from app.services.ingestion.image_ingestion import (
    ImageIngestionService,
    ImageIngestionResult,
    DetectedItemWithPrediction,
    GPT4O_DEFAULT_CONFIDENCE,
    VALID_UNITS,
    UNIT_NORMALIZATION_MAP
)


class TestGPT4oVisionClient:
    """Tests for the GPT-5.2 Vision API client."""

    def setup_method(self):
        self.client = GPT4oVisionClient()

    def test_detect_image_type_jpeg(self):
        """JPEG images should be detected correctly."""
        jpeg_bytes = b"\xff\xd8\xff\xe0\x00\x10JFIF"
        assert self.client._detect_image_type(jpeg_bytes) == "jpeg"

    def test_detect_image_type_png(self):
        """PNG images should be detected correctly."""
        png_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 10
        assert self.client._detect_image_type(png_bytes) == "png"

    def test_detect_image_type_gif(self):
        """GIF images should be detected correctly."""
        gif_bytes = b"GIF89a" + b"\x00" * 10
        assert self.client._detect_image_type(gif_bytes) == "gif"

    def test_detect_image_type_webp(self):
        """WebP images should be detected correctly."""
        webp_bytes = b"RIFF\x00\x00\x00\x00WEBP"
        assert self.client._detect_image_type(webp_bytes) == "webp"

    def test_detect_image_type_unknown_defaults_to_jpeg(self):
        """Unknown image types should default to jpeg."""
        unknown_bytes = b"UNKNOWN" + b"\x00" * 10
        assert self.client._detect_image_type(unknown_bytes) == "jpeg"

    @patch("app.services.ingestion.gpt4o_vision.OpenAI")
    def test_detect_food_items_success(self, mock_openai_class):
        """Successful detection should return list of DetectedFoodItem."""
        # Mock the OpenAI response
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"items": [{"name": "milk", "category": "Dairy", "quantity": 1, "unit": "Liters", "quantity_confidence": 0.9}, {"name": "chicken breast", "category": "Meat", "quantity": 500, "unit": "Grams", "quantity_confidence": 0.7}]}'
        mock_client.chat.completions.create.return_value = mock_response

        client = GPT4oVisionClient()
        client._client = mock_client  # Inject mock

        result = client.detect_food_items(b"\xff\xd8\xff")

        assert len(result) == 2
        assert result[0].name == "milk"
        assert result[0].category == "Dairy"
        assert result[0].quantity == 1
        assert result[0].unit == "Liters"
        assert result[0].quantity_confidence == 0.9
        assert result[1].name == "chicken breast"
        assert result[1].category == "Meat"
        assert result[1].quantity == 500
        assert result[1].unit == "Grams"

    @patch("app.services.ingestion.gpt4o_vision.OpenAI")
    def test_detect_food_items_with_null_quantity(self, mock_openai_class):
        """Items with null quantity should be handled correctly."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"items": [{"name": "unknown item", "category": "Other", "quantity": null, "unit": null, "quantity_confidence": null}]}'
        mock_client.chat.completions.create.return_value = mock_response

        client = GPT4oVisionClient()
        client._client = mock_client

        result = client.detect_food_items(b"\xff\xd8\xff")

        assert len(result) == 1
        assert result[0].name == "unknown item"
        assert result[0].quantity is None
        assert result[0].unit is None
        assert result[0].quantity_confidence is None

    @patch("app.services.ingestion.gpt4o_vision.OpenAI")
    def test_detect_food_items_empty_response(self, mock_openai_class):
        """Empty detection should return empty list."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"items": []}'
        mock_client.chat.completions.create.return_value = mock_response

        client = GPT4oVisionClient()
        client._client = mock_client

        result = client.detect_food_items(b"\xff\xd8\xff")

        assert len(result) == 0

    @patch("app.services.ingestion.gpt4o_vision.OpenAI")
    def test_detect_food_items_api_error(self, mock_openai_class):
        """API errors should raise RuntimeError."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        client = GPT4oVisionClient()
        client._client = mock_client

        with pytest.raises(RuntimeError) as exc_info:
            client.detect_food_items(b"\xff\xd8\xff")

        assert "GPT-5.2 API error" in str(exc_info.value)


class TestImageIngestionService:
    """Tests for the image ingestion orchestration service."""

    def setup_method(self):
        self.service = ImageIngestionService()

    def test_normalize_category_dairy(self):
        """Dairy category should be normalized correctly."""
        assert self.service._normalize_category("dairy") == "dairy"
        assert self.service._normalize_category("DAIRY") == "dairy"
        assert self.service._normalize_category("  Dairy  ") == "dairy"

    def test_normalize_category_meat(self):
        """Meat category should be normalized correctly."""
        assert self.service._normalize_category("meat") == "meat"
        assert self.service._normalize_category("poultry") == "poultry"

    def test_normalize_category_bread_to_bakery(self):
        """Bread category should map to bakery."""
        assert self.service._normalize_category("bread") == "bakery"
        assert self.service._normalize_category("bakery") == "bakery"

    def test_normalize_category_other_returns_none(self):
        """Other category should return None for fallback prediction."""
        assert self.service._normalize_category("other") is None

    def test_normalize_category_none_input(self):
        """None input should return None."""
        assert self.service._normalize_category(None) is None

    def test_normalize_category_unknown_passthrough(self):
        """Unknown categories should pass through."""
        assert self.service._normalize_category("exotic_food") == "exotic_food"

    def test_normalize_unit_valid_units(self):
        """Valid units should pass through unchanged."""
        for unit in VALID_UNITS:
            assert self.service._normalize_unit(unit) == unit

    def test_normalize_unit_lowercase_variations(self):
        """Lowercase variations should be normalized."""
        assert self.service._normalize_unit("pieces") == "Pieces"
        assert self.service._normalize_unit("grams") == "Grams"
        assert self.service._normalize_unit("kilograms") == "Kilograms"
        assert self.service._normalize_unit("milliliters") == "Milliliters"
        assert self.service._normalize_unit("liters") == "Liters"

    def test_normalize_unit_abbreviations(self):
        """Common abbreviations should be normalized."""
        assert self.service._normalize_unit("g") == "Grams"
        assert self.service._normalize_unit("kg") == "Kilograms"
        assert self.service._normalize_unit("ml") == "Milliliters"
        assert self.service._normalize_unit("l") == "Liters"
        assert self.service._normalize_unit("pcs") == "Pieces"

    def test_normalize_unit_none_input(self):
        """None input should return None."""
        assert self.service._normalize_unit(None) is None

    def test_normalize_unit_invalid_returns_none(self):
        """Invalid units should return None."""
        assert self.service._normalize_unit("cups") is None
        assert self.service._normalize_unit("ounces") is None
        assert self.service._normalize_unit("unknown") is None

    @patch("app.services.ingestion.image_ingestion.gpt4o_vision_client")
    @patch("app.services.ingestion.image_ingestion.expiry_prediction_service")
    def test_ingest_from_image_success(self, mock_expiry_service, mock_vision_client):
        """Successful ingestion should return processed items with quantity."""
        # Mock GPT-5.2 detection with quantity
        mock_vision_client.detect_food_items.return_value = [
            DetectedFoodItem(name="whole milk", category="dairy", quantity=1, unit="Liters", quantity_confidence=0.9),
            DetectedFoodItem(name="chicken breast", category="meat", quantity=500, unit="Grams", quantity_confidence=0.7),
        ]

        # Mock expiry prediction
        mock_prediction = MagicMock()
        mock_prediction.expiry_date = date.today() + timedelta(days=7)
        mock_prediction.reasoning = "Based on category 'dairy' stored in 'fridge'"
        mock_expiry_service.predict_expiry.return_value = mock_prediction

        result = self.service.ingest_from_image(
            image_bytes=b"\xff\xd8\xff",
            storage_location="fridge"
        )

        assert result.success is True
        assert len(result.detected_items) == 2
        assert result.detected_items[0].name == "whole milk"
        assert result.detected_items[0].category == "dairy"
        assert result.detected_items[0].confidence_score == GPT4O_DEFAULT_CONFIDENCE
        assert result.detected_items[0].quantity == 1
        assert result.detected_items[0].unit == "Liters"
        assert result.detected_items[0].quantity_confidence == 0.9
        assert result.detected_items[1].name == "chicken breast"
        assert result.detected_items[1].quantity == 500
        assert result.detected_items[1].unit == "Grams"

    @patch("app.services.ingestion.image_ingestion.gpt4o_vision_client")
    def test_ingest_from_image_no_items_detected(self, mock_vision_client):
        """Empty detection should return error."""
        mock_vision_client.detect_food_items.return_value = []

        result = self.service.ingest_from_image(
            image_bytes=b"\xff\xd8\xff",
            storage_location="fridge"
        )

        assert result.success is False
        assert "No food items detected" in result.error_message

    @patch("app.services.ingestion.image_ingestion.gpt4o_vision_client")
    def test_ingest_from_image_api_error(self, mock_vision_client):
        """API errors should be handled gracefully."""
        mock_vision_client.detect_food_items.side_effect = RuntimeError("API failed")

        result = self.service.ingest_from_image(
            image_bytes=b"\xff\xd8\xff",
            storage_location="fridge"
        )

        assert result.success is False
        assert "API failed" in result.error_message

    @patch("app.services.ingestion.image_ingestion.gpt4o_vision_client")
    @patch("app.services.ingestion.image_ingestion.expiry_prediction_service")
    def test_ingest_from_image_with_freezer_storage(self, mock_expiry_service, mock_vision_client):
        """Storage location should be passed to expiry prediction."""
        mock_vision_client.detect_food_items.return_value = [
            DetectedFoodItem(name="chicken", category="meat"),
        ]

        mock_prediction = MagicMock()
        mock_prediction.expiry_date = date.today() + timedelta(days=90)
        mock_prediction.reasoning = "Based on category 'meat' stored in 'freezer'"
        mock_expiry_service.predict_expiry.return_value = mock_prediction

        result = self.service.ingest_from_image(
            image_bytes=b"\xff\xd8\xff",
            storage_location="freezer"
        )

        assert result.success is True
        # Verify storage location was passed
        mock_expiry_service.predict_expiry.assert_called_with(
            name="chicken",
            category="meat",
            storage_location="freezer"
        )


class TestDetectionPrompt:
    """Tests for the GPT-5.2 detection prompt."""

    def test_prompt_includes_required_categories(self):
        """Prompt should include all GPT-5.2 food categories."""
        # These are the categories GPT-5.2 is asked to use
        required_categories = [
            "dairy", "meat", "fish", "vegetables", "fruits",
            "grains", "snacks", "beverages", "frozen", "condiments", "other"
        ]

        for category in required_categories:
            assert category in DETECTION_PROMPT.lower()

    def test_prompt_requests_json_format(self):
        """Prompt should request JSON output format."""
        assert "json" in DETECTION_PROMPT.lower()
        assert "items" in DETECTION_PROMPT.lower()

    def test_prompt_emphasizes_clarity(self):
        """Prompt should emphasize clear identification."""
        assert "clearly identify" in DETECTION_PROMPT.lower()

    def test_prompt_includes_quantity_fields(self):
        """Prompt should request quantity estimation."""
        assert "quantity" in DETECTION_PROMPT.lower()
        assert "unit" in DETECTION_PROMPT.lower()
        assert "quantity_confidence" in DETECTION_PROMPT.lower()

    def test_prompt_includes_valid_units(self):
        """Prompt should list all valid unit options."""
        for unit in VALID_UNITS:
            assert unit in DETECTION_PROMPT
