import pytest
import tiled
from tiled._utils import cdiv, next_power_of_2, get_powers_of_2
from tiled._configs import get_attn_tile_configs


def test_version():
  assert tiled.__version__ == "0.0.1"


def test_public_api():
  assert hasattr(tiled, "cdiv")
  assert hasattr(tiled, "next_power_of_2")


@pytest.mark.parametrize(
  "a,b,expected",
  [(10, 3, 4), (9, 3, 3), (1, 1, 1), (0, 8, 0), (4224, 64, 66)],
)
def test_cdiv(a, b, expected):
  assert cdiv(a, b) == expected


@pytest.mark.parametrize(
  "n,expected",
  [(1, 1), (2, 2), (3, 4), (64, 64), (65, 128), (128, 128), (129, 256)],
)
def test_next_power_of_2(n, expected):
  assert next_power_of_2(n) == expected


def test_get_powers_of_2():
  assert get_powers_of_2(1, 8) == [1, 2, 4, 8]
  assert get_powers_of_2(4, 4) == [4]
  assert get_powers_of_2(256, 1024) == [256, 512, 1024]


def test_attn_tile_configs_d128():
  for tile_m, _ in get_attn_tile_configs(D=128, sm="sm_120"):
    assert tile_m <= 64  # TILE_M=128 spills at occupancy=2


def test_attn_tile_configs_d64():
  assert len(get_attn_tile_configs(D=64, sm="sm_120")) > 0


def test_attn_tile_configs_fallback():
  assert get_attn_tile_configs(D=128, sm="sm_999") == [(64, 64)]

