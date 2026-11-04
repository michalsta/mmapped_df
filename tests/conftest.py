import numpy as np
import pandas as pd
import pytest

import mmapped_df


@pytest.fixture
def small_df():
    return pd.DataFrame(
        {
            "id": np.arange(5, dtype=np.uint32),
            "value": np.linspace(0.0, 1.0, 5),
            "with space": np.arange(5, dtype=np.int16) - 2,
        }
    )


@pytest.fixture
def written(tmp_path, small_df):
    """Path of a dataset holding `small_df`."""
    path = tmp_path / "ds"
    with mmapped_df.DatasetWriter(path) as w:
        w.append_df(small_df)
    return path
