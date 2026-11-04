# mmapped_df

Memory-mapped, on-disk data frames for NumPy, pandas, PyArrow and Polars.

`mmapped_df` stores a table as a directory of raw binary column files. Opening
it maps those files into memory, so a table far larger than RAM opens
instantly and only the pages you actually touch are read from disk. The
columns come back as ordinary NumPy arrays (or a pandas / PyArrow / Polars
frame over them) without any parsing or copying.

It is meant for large numeric tables that are written once (or appended to)
and read many times, e.g. by several processes at once, which then all share
the same page cache.

## Installation

The package is not on PyPI; install it from GitHub:

```sh
pip install git+https://github.com/michalsta/mmapped_df
```

Requires Python 3.10+, NumPy, pandas, PyArrow, Numba and Click. Polars is
optional and only needed for `open_dataset_pl`:

```sh
pip install "mmapped_df[polars] @ git+https://github.com/michalsta/mmapped_df"
```

## Quick start

```python
import numpy as np
import pandas as pd
import mmapped_df

# Write: append data frames chunk by chunk.
with mmapped_df.DatasetWriter("events.mmdf") as writer:
    for chunk in range(3):
        writer.append_df(pd.DataFrame({
            "event_id": np.arange(chunk * 1000, (chunk + 1) * 1000, dtype=np.uint64),
            "intensity": np.random.rand(1000).astype(np.float32),
        }))

# Read: a pandas DataFrame backed by the mapped files.
df = mmapped_df.open_dataset("events.mmdf")
print(len(df), df.dtypes.to_dict())
print(df.intensity.mean())
```

## On-disk format

A dataset is a directory:

```
events.mmdf/
├── schema.txt    # one line per column: "<numpy dtype> <column name>"
├── 0.bin         # raw values of column 0, back to back
└── 1.bin         # raw values of column 1
```

```
uint64 event_id
float32 intensity
```

Column `i` lives in `i.bin`, in the order of `schema.txt`. Each `.bin` file is
just the column's values in native NumPy layout, with no header, so
`np.fromfile("events.mmdf/1.bin", dtype=np.float32)` reads it too, and any
other language can read it as easily. Column names may contain spaces.

## Reading

All readers map the files read-only by default; nothing is loaded until it is
accessed.

| Function | Returns |
|---|---|
| `open_dataset(path)` | `pandas.DataFrame` |
| `open_dataset_dct(path)` | `dict[str, numpy.ndarray]` |
| `open_dataset_simple_namespace(path)` | `types.SimpleNamespace` with one attribute per column |
| `open_dataset_pa(path)` | `dict[str, pyarrow.Array]` sharing the mapped memory |
| `open_dataset_pl(path)` | `polars.DataFrame` over the PyArrow arrays |

```python
cols = mmapped_df.open_dataset_dct("events.mmdf")
ids = cols["event_id"]                # a numpy array over the mapped file
print(ids[:5], ids.flags.writeable)   # read-only: assignment raises ValueError

ns = mmapped_df.open_dataset_simple_namespace("events.mmdf")
print(ns.intensity.max())

arrow = mmapped_df.open_dataset_pa("events.mmdf")
print(arrow["event_id"].type)
```

`open_dataset_pa` and `open_dataset_pl` never copy. If a column cannot be
shared with Arrow as-is, they raise `TypeError` instead of copying or
misreading it. This applies to `bool` columns (Arrow stores booleans as bits,
NumPy as bytes) and `datetime64[D]` (Arrow reads days as 32-bit `date32`). Use
`uint8` instead of `bool` for datasets you want to open with Arrow or Polars.

### Modifying data in place

Pass `read_write=True` to get writable arrays; writes go straight to the files.

```python
cols = mmapped_df.open_dataset_dct("events.mmdf", read_write=True)
cols["intensity"][:10] = 0.0
del cols   # changes reach disk; the OS flushes the pages
```

The number of rows cannot change this way; use `DatasetWriter` to append.

## Writing

### `DatasetWriter`

```python
with mmapped_df.DatasetWriter("out.mmdf") as writer:
    writer.append_df(df)            # whole data frames
    writer.append_row(event_id=np.uint64(7), intensity=np.float32(0.5))  # one row
    print(len(writer))              # rows written so far
```

- The schema (column names, order and dtypes) is fixed by the first data
  written, or up front with `DatasetWriter.new` (below).
- `append_df` requires exactly the dataset's columns, in the same order and
  with the same dtypes. Anything else raises `ValueError` / `TypeError`
  before any data is written, so a failed append never leaves columns of
  different lengths behind.
- `append_row` converts each value to the column's dtype.
- Data is buffered: use the writer as a context manager, or call `close()` /
  `flush()`, before reading the dataset.

Constructor options:

| Argument | Default | Meaning |
|---|---|---|
| `append_ok` | `False` | If the dataset exists, open it and append to it. |
| `overwrite_dir` | `False` | Delete an existing directory at `path` and start fresh. |

With neither, an existing directory raises `FileExistsError`.

```python
with mmapped_df.DatasetWriter("events.mmdf", append_ok=True) as writer:
    writer.append_df(more_events)
```

When reopening, the writer checks the files against the schema and raises
`RuntimeError("Corrupted dataset: ...")` if their sizes do not match.

### Declaring the schema up front

```python
writer = mmapped_df.DatasetWriter.new("new.mmdf", event_id=np.uint64, intensity=np.float32)
writer.append_row(event_id=1, intensity=2.5)
writer.close()
```

`get_scheme(name=dtype, ...)` builds the same schema as an empty data frame,
for the functions below that take one.

### Adding columns

New columns can be added to a dataset that already has rows, as long as their
length matches:

```python
with mmapped_df.DatasetWriter("events.mmdf", append_ok=True) as writer:
    cols = mmapped_df.open_dataset_dct("events.mmdf")
    writer.append_column("double_intensity", cols["intensity"] * 2)
    writer.append_columns(flag=np.zeros(len(writer), dtype=np.uint8))
```

Adding a column whose name already exists raises `ValueError`.

### Preallocating and filling in place

When the number of rows is known in advance, allocate the files once and
fill the arrays directly, e.g. from Numba code or several processes:

```python
cols = mmapped_df.open_new_dataset_dct(
    "grid.mmdf",
    mmapped_df.get_scheme(x=np.float64, y=np.float64),
    nrows=1_000_000,
)
cols["x"][:] = np.linspace(0, 1, 1_000_000)
cols["y"][:] = cols["x"] ** 2
del cols
```

`DatasetWriter.preallocate_dataset(path, schema, nrows)` does the allocation
alone. The files are created sparse where the file system supports it.

## Supported dtypes

Any fixed-width NumPy dtype:

- signed and unsigned integers
- `float16` / `float32` / `float64`
- `complex64` / `complex128`
- `bool`
- `datetime64` and `timedelta64` with any unit
- non-native byte orders such as `>i4`

The following raise `TypeError` when written:

- object columns, including Python strings and NumPy byte strings, which
  pandas stores as objects
- pandas extension dtypes: `str`, `category`, nullable `Int64` etc.,
  tz-aware datetimes

Encode strings as integer codes (e.g. with `pd.factorize`) and store the
mapping separately.

## Command-line tools

Installing the package provides two commands that edit datasets in place:

```sh
copy_column   PATH OLD_NAME NEW_NAME   # copy a column (reflinked on CoW file systems)
rename_column PATH OLD_NAME NEW_NAME   # rename a column (rewrites schema.txt only)
```

Both fail with exit code 2 if `OLD_NAME` does not exist or `NEW_NAME` already
does; with `--ignore_errors` they exit successfully and do nothing instead.
`copy_column` uses `cp --reflink=auto` and therefore needs GNU coreutils.

## Grouped access (legacy)

`IndexedReader(path, index_col)` gives access to a dataset grouped by an
integer column, CSR-style: `reader[i]` returns a dict of column slices for all
rows whose `index_col` equals `i`. The dataset must be sorted by `index_col`,
whose values must be non-negative integers.

```python
reader = mmapped_df.IndexedReader("clusters.mmdf", "cluster_id")
rows_of_cluster_5 = reader[5]
for group in reader.iter_nonempty():   # skips ids with no rows
    ...
removed = reader.filter("filtered.mmdf", keep_mask)   # write a filtered copy
```

`filter` takes one boolean per row (`True` = keep) and returns the number of
rows removed from each group. It raises `ValueError` if a group that had rows loses all of them,
unless `allow_vanishing_groups=True` is passed.

`IndexedReader`, `SingleColReader`, `GroupedIndex` and the helpers `mkindex`
and `count_falses_in_groups` are kept for backward compatibility and may be
removed. They are the only part of the package that uses Numba, which is
imported only when they are first used.

## Caveats

- **No concurrency control.** Several readers are fine; a writer running
  while others read the same dataset is not coordinated, and two writers
  corrupt it.
- **Arrays keep the files mapped.** The mapping lives as long as any array
  (or data frame) derived from it. On Windows, mapped files cannot be
  deleted or truncated until then.
- **Byte order is the writer's.** Values are stored in the writing machine's
  native byte order unless a column's dtype says otherwise (e.g. `>i4`).
  Datasets written on little-endian machines (x86-64, ARM64: nearly all of
  them) read back wrongly on big-endian ones.
- **Platforms.** Developed on Linux; macOS should work. Windows is untested.

## Development

```sh
git clone https://github.com/michalsta/mmapped_df
cd mmapped_df
pip install -e ".[test]"
pytest
```

`examples/` holds standalone scripts that generate and read sample datasets
in the working directory.

## Authors

Michał Piotr Startek and Mateusz Krzysztof Łącki.

## License

MIT; see [LICENSE](LICENSE).
