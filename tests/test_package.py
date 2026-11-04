import subprocess
import sys

import mmapped_df


def test_public_api():
    assert set(mmapped_df.__all__) == {
        "schema_to_str",
        "str_to_schema",
        "write_schema",
        "get_scheme",
        "DatasetWriter",
        "open_dataset_dct",
        "open_new_dataset_dct",
        "open_dataset_simple_namespace",
        "open_dataset",
        "np_to_pa",
        "open_dataset_pa",
        "open_dataset_pl",
        "IndexedReader",
        "SingleColReader",
        "GroupedIndex",
        "mkindex",
        "count_falses_in_groups",
    }
    for name in mmapped_df.__all__:
        assert getattr(mmapped_df, name) is not None
    for leaked in ("np", "pd", "pa", "os", "mmap", "Path", "ProgressBar"):
        assert not hasattr(mmapped_df, leaked)


def test_numba_is_imported_lazily(tmp_path):
    # Fresh interpreter: the test session itself may already have numba loaded.
    script = f"""
import sys
import numpy as np, pandas as pd
import mmapped_df
with mmapped_df.DatasetWriter({str(tmp_path / "ds")!r}) as w:
    w.append_df(pd.DataFrame({{"g": np.array([0, 1, 1], dtype=np.uint32)}}))
mmapped_df.open_dataset({str(tmp_path / "ds")!r})
assert "numba" not in sys.modules, "numba loaded by plain read/write"
mmapped_df.IndexedReader({str(tmp_path / "ds")!r}, "g")
assert "numba" in sys.modules
"""
    result = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


def test_schema_text_round_trip():
    scheme = mmapped_df.get_scheme(**{"a": "uint8", "with space": ">i4"})
    text = mmapped_df.schema_to_str(scheme)
    assert text.splitlines() == ["uint8 a", ">i4 with space"]
    assert mmapped_df.str_to_schema(text).dtypes.equals(scheme.dtypes)
