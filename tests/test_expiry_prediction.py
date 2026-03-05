"""
Unit tests for expiry prediction service.

Tests the core academic requirement: deterministic, transparent predictions.
"""
import pytest
from datetime import date, timedelta

from app.services.expiry_prediction.strategies.rule_based import RuleBasedStrategy
from app.services.expiry_prediction import ExpiryPredictionService


class TestRuleBasedStrategy:
    """Test rule-based prediction strategy."""

    def setup_method(self):
        self.strategy = RuleBasedStrategy()

    def test_dairy_in_fridge(self):
        """Dairy in fridge should predict 7-day shelf life."""
        prediction = self.strategy.predict(
            name="Milk", category="dairy", storage_location="fridge"
        )
        assert prediction.expiry_date == date.today() + timedelta(days=7)
        assert prediction.confidence == 0.85
        assert prediction.strategy_name == "rule_based"
        assert "dairy" in prediction.reasoning.lower()

    def test_meat_in_freezer(self):
        """Meat in freezer should predict 90-day shelf life."""
        prediction = self.strategy.predict(
            name="Chicken breast", category="meat", storage_location="freezer"
        )
        assert prediction.expiry_date == date.today() + timedelta(days=90)
        assert prediction.confidence == 0.90

    def test_fallback_predictions(self):
        """Missing category or storage should produce low-confidence fallbacks."""
        no_category = self.strategy.predict(
            name="Unknown item", category=None, storage_location="fridge"
        )
        assert no_category.confidence == 0.50
        assert "category unknown" in no_category.reasoning.lower()

        no_storage = self.strategy.predict(
            name="Milk", category="dairy", storage_location=None
        )
        assert no_storage.confidence == 0.30
        assert "no category or storage" in no_storage.reasoning.lower()

    def test_determinism(self):
        """Same inputs must always produce same outputs (academic requirement)."""
        p1 = self.strategy.predict(name="Eggs", category="eggs", storage_location="fridge")
        p2 = self.strategy.predict(name="Eggs", category="eggs", storage_location="fridge")

        assert p1.expiry_date == p2.expiry_date
        assert p1.confidence == p2.confidence
        assert p1.reasoning == p2.reasoning

    def test_custom_purchase_date(self):
        """Prediction should be relative to the provided purchase date."""
        purchase = date(2024, 1, 1)
        prediction = self.strategy.predict(
            name="Milk", category="dairy",
            storage_location="fridge", purchase_date=purchase
        )
        assert prediction.expiry_date == purchase + timedelta(days=7)


class TestExpiryPredictionService:
    """Test the service orchestrator."""

    def setup_method(self):
        self.service = ExpiryPredictionService()

    def test_predict_expiry(self):
        """Service should return a valid prediction with date, confidence, and reasoning."""
        prediction = self.service.predict_expiry(
            name="Yogurt", category="dairy", storage_location="fridge"
        )
        assert isinstance(prediction.expiry_date, date)
        assert 0.0 <= prediction.confidence <= 1.0
        assert len(prediction.reasoning) > 0
