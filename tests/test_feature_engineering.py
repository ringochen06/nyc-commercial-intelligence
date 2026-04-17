"""Tests for POI ``simplify_category`` helpers in ``src.feature_engineering``."""

from src.feature_engineering import simplify_category


def test_unknown_restaurant_defaults_to_food():
    assert simplify_category("unknown", "restaurant") == "food"


def test_retail_license_always_retail():
    assert simplify_category("anything", "retail") == "retail"


def test_shopping_keywords_retail_before_restaurant_type():
    assert simplify_category("grocery store industry text", None) == "retail"
    assert simplify_category("electronics store", "restaurant") == "retail"


def test_coffee_shop_is_food_not_retail():
    assert simplify_category("coffee shop", "restaurant") == "food"


def test_plain_unknown_without_poi_type_is_other():
    assert simplify_category("unknown", None) == "other"
