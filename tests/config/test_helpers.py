"""Shared config-helper tests."""

import pytest


def test_parse_edge_counts():
    from kinovsr.config.helpers import parse_edge_counts

    assert parse_edge_counts("0,1,0,0") == (0, 1, 0, 0)
    assert parse_edge_counts(" 2, 3 ,4,5 ") == (2, 3, 4, 5)
    with pytest.raises(ValueError, match="four integers"):
        parse_edge_counts("1,2,3")
    with pytest.raises(ValueError, match=">= 0"):
        parse_edge_counts("1,-2,3,4")
