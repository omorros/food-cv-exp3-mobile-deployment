"""
Unit tests for image ingestion service.

Tests GPT-5.2 vision integration, category/unit normalization,
and the ingestion pipeline.
"""
import pytest
from unittest.mock import patch, MagicMock
from datetime import date, timedelta

from app.services.ingestion.gpt4o_vision import (
    GPT4oVisionClient,
    DetectedFoodItem,
)
from app.services.ingestion.image_ingestion import (
    ImageIngestionService,
    GPT4O_DEFAULT_CONFIDENCE,
    VALID_UNITS,
)


class TestGPT4oVisionClient:
    """Tests for the GPT-5.2 Vision API client."""

    def setup_method(self):
        self.client = GPT4oVisionClient()

    def test_detect_image_type(self):
        """Known formats should be detected; unknown defaults to jpeg."""
        assert self.client._detect_image_type(b"\xff\xd8\xff\xe0\x00\x10JFIF") == "jpeg"
        assert self.client._detect_image_type(b"\x89PNG\r\n\x1a\n" + b"\x00" * 10) == "png"
        assert self.client._detect_image_type(b"UNKNOWN" + b"\x00" * 10) == "jpeg"

    @patch("app.services.ingestion.gpt4o_vision.OpenAI")
    def test_detect_food_items_success(self, mock_openai_class):
        """Successful detection should return list of DetectedFoodItem."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"items": [{"name": "milk", "category": "Dairy", "quantity": 1, "unit": "Liters", "quantity_confidence": 0.9}, {"name": "chicken breast", "category": "Meat", "quantity": 500, "unit": "Grams", "quantity_confidence": 0.7}]}'
        mock_client.chat.completions.create.return_value = mock_response

        client = GPT4oVisionClient()
        client._client = mock_client

        result = client.detect_food_items(b"\xff\xd8\xff")

        assert len(result) == 2
        assert result[0].name == "milk"
        assert result[0].category == "Dairy"
        assert result[0].quantity == 1
        assert result[0].unit == "Liters"
        assert result[1].name == "chicken breast"
        assert result[1].quantity == 500

    @patch("app.services.ingestion.gpt4o_vision.OpenAI")
    def test_detect_food_items_null_quantity(self, mock_openai_class):
        """Items with null quantity should be handled gracefully."""
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
        assert result[0].quantity is None
        assert result[0].unit is None

    @patch("app.services.ingestion.gpt4o_vision.OpenAI")
    def test_detect_food_items_api_error(self, mock_openai_class):
        """API errors should raise RuntimeError."""
        mock_client = MagicMock()
        mock_openai_class.return_value = mock_client
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        client = GPT4oVisionClient()
        client._client = mock_client

        with pytest.raises(RuntimeError, match="GPT-5.2 API error"):
            client.detect_food_items(b"\xff\xd8\xff")


class TestImageIngestionService:
    """Tests for the image ingestion orchestration service."""

    def setup_method(self):
        self.service = ImageIngestionService()

    def test_normalize_category(self):
        """Categories should be lowercased, mapped, or returned as None."""
        assert self.service._normalize_category("dairy") == "dairy"
        assert self.service._normalize_category("  DAIRY  ") == "dairy"
        assert self.service._normalize_category("bread") == "bakery"
        assert self.service._normalize_category("other") is None
        assert self.service._normalize_category(None) is None
        assert self.service._normalize_category("exotic_food") == "exotic_food"

    def test_normalize_unit(self):
        """Units should be title-cased, abbreviations expanded, invalid rejected."""
        # Valid units pass through
        for unit in VALID_UNITS:
            assert self.service._normalize_unit(unit) == unit
        # Lowercase normalized
        assert self.service._normalize_unit("grams") == "Grams"
        assert self.service._normalize_unit("liters") == "Liters"
        # Abbreviations expanded
        assert self.service._normalize_unit("g") == "Grams"
        assert self.service._normalize_unit("kg") == "Kilograms"
        assert self.service._normalize_unit("ml") == "Milliliters"
        # None / invalid
        assert self.service._normalize_unit(None) is None
        assert self.service._normalize_unit("cups") is None

    @patch("app.services.ingestion.image_ingestion.gpt4o_vision_client")
    @patch("app.services.ingestion.image_ingestion.expiry_prediction_service")
    def test_ingest_from_image_success(self, mock_expiry_service, mock_vision_client):
        """Successful ingestion should return processed items with predictions."""
        mock_vision_client.detect_food_items.return_value = [
            DetectedFoodItem(name="whole milk", category="dairy", quantity=1, unit="Liters", quantity_confidence=0.9),
            DetectedFoodItem(name="chicken breast", category="meat", quantity=500, unit="Grams", quantity_confidence=0.7),
        ]

        mock_prediction = MagicMock()
        mock_prediction.expiry_date = date.today() + timedelta(days=7)
        mock_prediction.reasoning = "Based on category 'dairy' stored in 'fridge'"
        mock_expiry_service.predict_expiry.return_value = mock_prediction

        result = self.service.ingest_from_image(image_bytes=b"\xff\xd8\xff", storage_location="fridge")

        assert result.success is True
        assert len(result.detected_items) == 2
        assert result.detected_items[0].name == "whole milk"
        assert result.detected_items[0].confidence_score == GPT4O_DEFAULT_CONFIDENCE
        assert result.detected_items[0].quantity == 1
        assert result.detected_items[0].unit == "Liters"

    @patch("app.services.ingestion.image_ingestion.gpt4o_vision_client")
    def test_ingest_from_image_no_items(self, mock_vision_client):
        """Empty detection should return error."""
        mock_vision_client.detect_food_items.return_value = []

        result = self.service.ingest_from_image(image_bytes=b"\xff\xd8\xff", storage_location="fridge")

        assert result.success is False
        assert "No food items detected" in result.error_message

    @patch("app.services.ingestion.image_ingestion.gpt4o_vision_client")
    def test_ingest_from_image_api_error(self, mock_vision_client):
        """API errors should be handled gracefully."""
        mock_vision_client.detect_food_items.side_effect = RuntimeError("API failed")

        result = self.service.ingest_from_image(image_bytes=b"\xff\xd8\xff", storage_location="fridge")

        assert result.success is False
        assert "API failed" in result.error_message
