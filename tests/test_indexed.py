import numpy as np
import pandas as pd
import pytest

import mmapped_df
from mmapped_df import IndexedReader, open_dataset


@pytest.fixture
def grouped(tmp_path):
    """Groups 0, 2, 3 and 5; ids 1 and 4 have no rows."""
    path = tmp_path / "grouped"
    with mmapped_df.DatasetWriter(path) as w:
        w.append_df(
            pd.DataFrame(
                {
                    "g": np.array([0, 0, 2, 2, 2, 3, 5], dtype=np.uint32),
                    "v": np.arange(7, dtype=np.float64),
                }
            )
        )
    return path


def test_mkindex():
    index, counts, last = mmapped_df.mkindex(np.array([0, 0, 2, 2, 2, 3], dtype=np.uint32))
    assert index.tolist() == [0, 2, 2, 5, 6]
    assert counts.tolist() == [2, 0, 3, 1]
    assert last == 4


@pytest.mark.parametrize(
    "values, exc",
    [
        (np.array([-3, 1]), ValueError),  # used to write out of bounds
        (np.array([0.0, 1.0]), TypeError),
    ],
    ids=["negative", "float"],
)
def test_mkindex_rejects_bad_input(values, exc):
    with pytest.raises(exc):
        mmapped_df.mkindex(values)


def test_getitem_and_len(grouped):
    reader = IndexedReader(grouped, "g")
    assert len(reader) == 6
    assert reader[2]["v"].tolist() == [2.0, 3.0, 4.0]
    assert reader[1]["v"].tolist() == []
    assert reader.get_column(5, "v").tolist() == [6.0]
    with pytest.raises(IndexError):
        reader[6]
    with pytest.raises(IndexError):
        reader[-1]


def test_iteration(grouped):
    reader = IndexedReader(grouped, "g")
    assert [len(group["v"]) for group in reader] == [2, 0, 3, 1, 0, 1]
    assert [group["v"].tolist() for group in reader.iter_nonempty()] == [
        [0.0, 1.0],
        [2.0, 3.0, 4.0],
        [5.0],
        [6.0],
    ]


def test_filter_with_gaps(grouped, tmp_path):
    # Regression: groups that were empty from the start made filter() fail.
    reader = IndexedReader(grouped, "g")
    keep = np.array([True, False, True, False, True, True, True])
    removed = reader.filter(tmp_path / "out", keep)
    assert removed.tolist() == [1, 0, 1, 0, 0, 0]
    out = open_dataset(tmp_path / "out")
    assert out["v"].tolist() == [0.0, 2.0, 4.0, 5.0, 6.0]
    assert out["g"].tolist() == [0, 2, 2, 3, 5]


def test_filter_vanishing_group(grouped, tmp_path):
    reader = IndexedReader(grouped, "g")
    keep = np.array([True, True, True, True, True, False, True])  # empties group 3
    with pytest.raises(ValueError, match=r"\[3\]"):
        reader.filter(tmp_path / "out", keep)
    assert not (tmp_path / "out").exists()
    removed = reader.filter(tmp_path / "out", keep, allow_vanishing_groups=True)
    assert removed.tolist() == [0, 0, 0, 1, 0, 0]
    assert open_dataset(tmp_path / "out")["g"].tolist() == [0, 0, 2, 2, 2, 5]


def test_single_col_reader(grouped):
    reader = mmapped_df.SingleColReader(grouped, "g", "v")
    assert len(reader) == 6
    assert reader[2].tolist() == [2.0, 3.0, 4.0]


def test_grouped_index():
    df = pd.DataFrame({"x": [10, 20, 30]})
    gi = mmapped_df.GroupedIndex(pd.Series([0, 1, 1]), df)
    assert len(gi) == 2
    assert gi[1]["x"].tolist() == [20, 30]
