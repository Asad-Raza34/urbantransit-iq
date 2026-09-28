"""Tests for recommendation engine."""

import pytest
import pandas as pd
import numpy as np

from src.recommend.engine import (
    RecommendationEngine,
    _determine_priority,
    _select_action,
    _build_explanation,
    PRIORITY_RULES,
    CATEGORY_ACTIONS,
)


class TestPriorityRules:
    """Test the priority determination logic."""

    def test_critical_priority_crowding(self):
        evidence = {"crowding_share": 0.45, "critical_crowding_share": 0.15}
        assert _determine_priority(evidence) == "CRITICAL"

    def test_critical_priority_on_time(self):
        evidence = {"on_time_share": 0.50}
        assert _determine_priority(evidence) == "CRITICAL"

    def test_high_priority_crowding(self):
        evidence = {"crowding_share": 0.30}
        assert _determine_priority(evidence) == "HIGH"

    def test_medium_priority_bunching(self):
        evidence = {"bunching_share": 0.20}
        assert _determine_priority(evidence) == "MEDIUM"

    def test_low_priority_default(self):
        evidence = {"crowding_share": 0.05, "on_time_share": 0.90}
        assert _determine_priority(evidence) == "LOW"

    def test_priority_order(self):
        """Verify priority order is respected."""
        critical_ev = {"crowding_share": 0.5}
        high_ev = {"crowding_share": 0.3}
        medium_ev = {"bunching_share": 0.2}
        low_ev = {"crowding_share": 0.05}

        priorities = [
            _determine_priority(critical_ev),
            _determine_priority(high_ev),
            _determine_priority(medium_ev),
            _determine_priority(low_ev),
        ]

        # Should be in order: CRITICAL > HIGH > MEDIUM > LOW
        assert priorities == ["CRITICAL", "HIGH", "MEDIUM", "LOW"]


class TestActionSelection:
    """Test action and effect selection."""

    def test_known_categories(self):
        for category in CATEGORY_ACTIONS:
            action, effect = _select_action(category, {})
            assert action
            assert effect
            assert isinstance(action, str)
            assert isinstance(effect, str)

    def test_unknown_category(self):
        action, effect = _select_action("unknown_category", {})
        assert action == "Review and take appropriate operational action"
        assert effect == "Improve operational performance"


class TestExplanationBuilder:
    """Test explanation generation."""

    def test_build_explanation(self):
        from src.recommend.engine import Recommendation
        rec = Recommendation(
            recommendation_id="test-001",
            category="capacity_increase",
            priority="HIGH",
            affected_route="R-001",
            affected_stop=None,
            title="Test",
            description="Test description",
            evidence={"crowding_share": 0.3},
            metric_value=0.3,
            threshold_used=0.85,
            suggested_action="Add capacity",
            expected_effect="Reduce crowding",
            confidence=0.85,
            generated_at="2024-01-01T00:00:00+00:00",
        )

        explanation = _build_explanation(rec)

        assert "What happened?" in explanation
        assert "Why is it important?" in explanation
        assert "Evidence" in explanation
        assert "What is suggested?" in explanation
        assert "Why this action?" in explanation
        assert "Expected effect" in explanation
        assert "Confidence: 85%" in explanation
        assert "capacity_increase" in explanation.lower() or "capacity" in explanation.lower()


class TestRecommendationEngine:
    """Test the recommendation engine."""

    def test_engine_initialization(self):
        engine = RecommendationEngine()
        assert engine.recommendations == []

    def test_priority_rules_structure(self):
        """Verify priority rules have expected structure."""
        for priority in ["CRITICAL", "HIGH", "MEDIUM", "LOW"]:
            assert priority in PRIORITY_RULES
            assert isinstance(PRIORITY_RULES[priority], dict)
            assert len(PRIORITY_RULES[priority]) > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])