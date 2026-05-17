import pytest
from app.utils.label_parsing import parse_class_label


def test_compound_label_parses_species_and_viewpoint():
    assert parse_class_label("salamander_fire_adult:up", compound_labels=True) == \
        ("salamander_fire_adult", "up")


def test_compound_label_default_sentinel_suppresses_species():
    assert parse_class_label("species:left", compound_labels=True) == (None, "left")


def test_compound_label_explicit_empty_sentinel_keeps_species():
    assert parse_class_label("species:left", compound_labels=True,
                             sentinel_prefixes=[]) == ("species", "left")


def test_compound_label_custom_sentinel_list():
    assert parse_class_label("species:left", compound_labels=True,
                             sentinel_prefixes=["species", "viewpoint"]) == (None, "left")


def test_compound_true_no_colon_returns_viewpoint_only():
    assert parse_class_label("up", compound_labels=True) == (None, "up")


def test_compound_false_no_colon_returns_viewpoint_only():
    assert parse_class_label("up", compound_labels=False) == (None, "up")


def test_compound_false_with_colon_raises_value_error():
    with pytest.raises(ValueError) as exc_info:
        parse_class_label("salamander_fire_adult:up", compound_labels=False)
    assert "compound_labels" in str(exc_info.value).lower()


def test_compound_label_only_one_colon_splits_correctly():
    # Multi-colon labels split on first colon only.
    assert parse_class_label("a:b:c", compound_labels=True) == ("a", "b:c")
