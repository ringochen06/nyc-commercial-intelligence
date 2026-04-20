"""Tests for POI ``simplify_category`` in ``src.feature_engineering`` (keyword-based on license text)."""

from src.feature_engineering import simplify_category


def test_unknown_is_other():
    assert simplify_category("unknown") == "other"


def test_grocery_store_is_retail():
    assert simplify_category("grocery store industry text") == "retail"


def test_electronics_store_is_retail():
    assert simplify_category("electronics store") == "retail"


def test_coffee_shop_is_food():
    assert simplify_category("coffee shop") == "food"


def test_restaurant_keyword_is_food():
    assert simplify_category("italian restaurant") == "food"
