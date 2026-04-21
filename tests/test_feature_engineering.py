"""Tests for POI ``simplify_category`` in ``src.feature_engineering`` (keyword-based on license text)."""

from src.feature_engineering import simplify_category


def test_unknown_is_other():
    print("\n[TEST] test_unknown_is_other: Testing that 'unknown' category simplifies to 'other'")
    assert simplify_category("unknown") == "other"


def test_grocery_store_is_retail():
    print("\n[TEST] test_grocery_store_is_retail: Testing that 'grocery store' category simplifies to 'retail'")
    assert simplify_category("grocery store industry text") == "retail"


def test_electronics_store_is_retail():
    print("\n[TEST] test_electronics_store_is_retail: Testing that 'electronics store' category simplifies to 'retail'")
    assert simplify_category("electronics store") == "retail"


def test_coffee_shop_is_food():
    print("\n[TEST] test_coffee_shop_is_food: Testing that 'coffee shop' category simplifies to 'food'")
    assert simplify_category("coffee shop") == "food"


def test_restaurant_keyword_is_food():
    print("\n[TEST] test_restaurant_keyword_is_food: Testing that 'italian restaurant' category simplifies to 'food'")
    assert simplify_category("italian restaurant") == "food"
