"""Tests for storefront activity column normalization helpers."""

from src.feature_engineering import storefront_activity_column_name


def test_other_upper_maps_to_lower_other():
    assert storefront_activity_column_name("OTHER") == "act_other_storefront"


def test_whitespace_and_case_normalization():
    assert (
        storefront_activity_column_name("  food services  ")
        == "act_FOOD_SERVICES_storefront"
    )


def test_slashes_and_dashes_are_normalized():
    assert (
        storefront_activity_column_name("HEALTH CARE / SOCIAL-ASSISTANCE")
        == "act_HEALTH_CARE_SOCIAL_ASSISTANCE_storefront"
    )


def test_empty_activity_falls_back_to_unknown():
    assert storefront_activity_column_name("") == "act_UNKNOWN_storefront"
