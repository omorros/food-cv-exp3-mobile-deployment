"""
Integration tests for API endpoints.

Tests the core user flows: registration, login,
draft-to-inventory promotion, and image ingestion.
"""
import pytest
from datetime import date, timedelta
from unittest.mock import patch, MagicMock
from uuid import uuid4

from app.services.ingestion.gpt52_vision import DetectedFoodItem


class TestAuthFlow:
    """Tests for registration and login."""

    def test_register_and_login(self, client):
        """Full auth flow: register, then login with same credentials."""
        # Register
        reg = client.post("/auth/register", json={
            "email": "new@example.com",
            "password": "securepass123",
        })
        assert reg.status_code == 201
        assert "access_token" in reg.json()

        # Login
        login = client.post("/auth/login", json={
            "email": "new@example.com",
            "password": "securepass123",
        })
        assert login.status_code == 200
        assert "access_token" in login.json()

    def test_login_wrong_password(self, client, test_user):
        """Wrong password should be rejected."""
        response = client.post("/auth/login", json={
            "email": "test@example.com",
            "password": "wrongpassword",
        })
        assert response.status_code == 401

    def test_protected_route_without_token(self, client):
        """Protected endpoints should reject unauthenticated requests."""
        assert client.get("/auth/me").status_code == 401
        assert client.get("/api/draft-items").status_code == 401
        assert client.get("/api/inventory").status_code == 401


class TestDraftToInventoryFlow:
    """Tests the core app flow: create draft -> confirm -> inventory."""

    def test_full_draft_to_inventory_flow(self, client, test_user, auth_headers):
        """Draft creation, confirmation to inventory, and draft cleanup."""
        # 1. Create draft
        draft = client.post("/api/draft-items", json={
            "name": "Whole Milk",
            "category": "dairy",
            "location": "fridge",
        }, headers=auth_headers)
        assert draft.status_code == 201
        draft_id = draft.json()["id"]
        assert draft.json()["expiration_date"] is not None  # Auto-predicted

        # 2. Confirm draft -> inventory
        confirm = client.post(f"/api/draft-items/{draft_id}/confirm", json={
            "name": "Whole Milk",
            "category": "dairy",
            "quantity": 1.0,
            "unit": "Liters",
            "storage_location": "fridge",
            "expiry_date": (date.today() + timedelta(days=7)).isoformat(),
        }, headers=auth_headers)
        assert confirm.status_code == 201
        item_id = confirm.json()["id"]

        # 3. Draft should be deleted after confirmation
        assert client.get(f"/api/draft-items/{draft_id}", headers=auth_headers).status_code == 404

        # 4. Inventory item should exist
        inv = client.get(f"/api/inventory/{item_id}", headers=auth_headers)
        assert inv.status_code == 200
        assert inv.json()["name"] == "Whole Milk"

    def test_delete_inventory_item(self, client, test_user, auth_headers):
        """Should be able to delete consumed/discarded items."""
        # Create and confirm
        draft = client.post("/api/draft-items", json={
            "name": "Chicken", "category": "meat", "location": "fridge",
        }, headers=auth_headers)
        draft_id = draft.json()["id"]

        confirm = client.post(f"/api/draft-items/{draft_id}/confirm", json={
            "name": "Chicken Breast", "category": "meat", "quantity": 500,
            "unit": "Grams", "storage_location": "fridge",
            "expiry_date": (date.today() + timedelta(days=3)).isoformat(),
        }, headers=auth_headers)
        item_id = confirm.json()["id"]

        # Delete
        assert client.delete(f"/api/inventory/{item_id}", headers=auth_headers).status_code == 204
        assert client.get(f"/api/inventory/{item_id}", headers=auth_headers).status_code == 404


class TestImageIngestion:
    """Tests for the image recognition endpoint."""

    @patch("app.services.ingestion.image_ingestion.gpt52_vision_client")
    @patch("app.services.ingestion.image_ingestion.expiry_prediction_service")
    def test_ingest_image_creates_drafts(
        self, mock_expiry, mock_vision, client, test_user, auth_headers
    ):
        """Image upload should detect items and create draft items."""
        mock_vision.detect_food_items.return_value = [
            DetectedFoodItem(
                name="whole milk", category="dairy",
                quantity=1, unit="Liters", quantity_confidence=0.9,
            ),
        ]
        mock_prediction = MagicMock()
        mock_prediction.expiry_date = date.today() + timedelta(days=7)
        mock_prediction.reasoning = "dairy in fridge: 7 days"
        mock_expiry.predict_expiry.return_value = mock_prediction

        fake_image = b"\xff\xd8\xff\xe0" + b"\x00" * 100
        response = client.post(
            "/api/ingest/image",
            files={"image": ("test.jpg", fake_image, "image/jpeg")},
            data={"storage_location": "fridge"},
            headers=auth_headers,
        )
        assert response.status_code == 201
        data = response.json()
        assert len(data) == 1
        assert data[0]["name"] == "whole milk"
        assert data[0]["source"] == "image"

    def test_ingest_rejects_non_image(self, client, test_user, auth_headers):
        """Non-image files should be rejected."""
        response = client.post(
            "/api/ingest/image",
            files={"image": ("test.txt", b"not an image", "text/plain")},
            data={"storage_location": "fridge"},
            headers=auth_headers,
        )
        assert response.status_code == 400


class TestHealthCheck:
    """Smoke test for the health endpoint."""

    def test_health(self, client):
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json() == {"status": "ok"}
