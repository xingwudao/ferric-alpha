import importlib.metadata as metadata
from datetime import datetime

import ferric_alpha
import polars as pl
import pytest
from ferric_alpha import utils


def test_package_exposes_version_metadata() -> None:
    assert ferric_alpha.__version__ == metadata.version("ferric-alpha")


def test_validate_factor_frame_round_trip() -> None:
    frame = pl.DataFrame(
        {
            "date": [
                datetime(2024, 1, 2),
                datetime(2024, 1, 1),
            ],
            "asset": ["B", "A"],
            "factor": [2.0, 1.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    result = utils.validate_factor_frame(frame)

    assert isinstance(result, pl.DataFrame)
    assert result.get_column("asset").to_list() == ["A", "B"]


def test_validate_factor_frame_rejects_duplicate_key() -> None:
    frame = pl.DataFrame(
        {
            "date": [datetime(2024, 1, 1)] * 2,
            "asset": ["A", "A"],
            "factor": [1.0, 2.0],
        }
    ).with_columns(pl.col("date").cast(pl.Datetime("ms")))

    with pytest.raises(ValueError, match="duplicate date/asset key"):
        utils.validate_factor_frame(frame)
