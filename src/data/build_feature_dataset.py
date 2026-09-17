"""Build compact per-shower feature datasets from a huge CORSIKA .DAT file
(millions of showers), processing it in batches via
corsario.iter_shower_batches() so the full particle table is never held in
memory at once -- see corsario's docs/LARGE_FILES.md and this project's
docs/PIPELINE.md.

    python data/build_feature_dataset.py \\
        --dat-file /data/DAT000001 \\
        --out-dir /data/features/run1 \\
        --input-types stats grid circle \\
        --batch-size 50000

Writes one HDF5 file per input_type: <out-dir>/<input_type>.h5, with
resizable datasets "X" (n_showers, n_features) and "y" (n_showers,).

Resumable: the HDF5 file's current row count IS the checkpoint. If
interrupted, re-running the exact same command skips that many showers
(cheaply -- see docs/LARGE_FILES.md on why this still means re-parsing,
just not re-computing features, for the already-done portion) and appends
from there.

Once built, point search/run_search.py's `search.feature_path` at the
directory instead of `search.dat_file`/`search.data_path` -- see
docs/SEARCH.md#large-files.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # -> src/

import corsario  # noqa: E402
from data.features.circle import circle_generator  # noqa: E402
from data.features.grid import grid_generator  # noqa: E402
from data.features.stats import stats_generator  # noqa: E402

INPUT_TYPES = ("stats", "grid", "circle")


def compute_global_xy_range(dat_file: str, batch_size: int = 50_000,
                             percentile: float | None = None,
                             max_showers: int | None = 500_000):
    """A cheap(er) pass over the file's particle x/y to pick a FIXED grid
    range for grid_generator -- REQUIRED for correctness when calling it
    batch by batch (see the warning in data.features.grid.grid_generator).

    `max_showers` bounds this to a leading sample of the file instead of a
    full second pass (set to None to scan every shower for the exact
    global min/max). A sample is representative as long as the file isn't
    sorted in a way that clusters shower size with file position (e.g. by
    ascending primary energy) -- if it might be, pass max_showers=None or
    supply --x-range/--y-range explicitly from known detector-array bounds
    instead of estimating from the data at all.

    `percentile` (e.g. 99.5) uses a percentile range instead of the true
    min/max, more robust to rare outlier particles; None matches
    grid_generator's own single-call default (true min/max) exactly.
    """
    xs, ys = [], []
    n_seen = 0
    for df_particles, df_shower in corsario.iter_shower_batches(dat_file, batch_size=batch_size):
        xs.append(df_particles["x"].to_numpy())
        ys.append(df_particles["y"].to_numpy())
        n_seen += len(df_shower)
        if max_showers and n_seen >= max_showers:
            break
    x_all = np.concatenate(xs)
    y_all = np.concatenate(ys)
    if percentile is None:
        x_range = (float(x_all.min()), float(x_all.max()))
        y_range = (float(y_all.min()), float(y_all.max()))
    else:
        lo, hi = (100 - percentile) / 2, 100 - (100 - percentile) / 2
        x_range = tuple(float(v) for v in np.percentile(x_all, [lo, hi]))
        y_range = tuple(float(v) for v in np.percentile(y_all, [lo, hi]))
    print(f"[build_dataset] estimated from {n_seen} shower(s): "
          f"x_range={x_range} y_range={y_range}")
    return x_range, y_range


def _get_or_create_h5(h5file, X_width):
    import h5py

    if "X" in h5file:
        X_ds, y_ds = h5file["X"], h5file["y"]
        if X_ds.shape[1] != X_width:
            raise ValueError(f"existing dataset has width {X_ds.shape[1]}, expected {X_width} "
                              f"-- input_type/grid_n mismatch with an earlier run?")
    else:
        X_ds = h5file.create_dataset("X", shape=(0, X_width), maxshape=(None, X_width),
                                      dtype="f4", chunks=(min(4096, max(1, 4096)), X_width),
                                      compression="gzip")
        y_ds = h5file.create_dataset("y", shape=(0,), maxshape=(None,), dtype="f4",
                                      chunks=True, compression="gzip")
    return X_ds, y_ds


def _append(ds, array):
    old = ds.shape[0]
    ds.resize(old + array.shape[0], axis=0)
    ds[old:] = array


def build_one(input_type: str, dat_file: str, out_path: Path, batch_size: int,
              grid_n: int = 8, x_range=None, y_range=None, strict: bool = True,
              progress_every: int = 1, device: str = "cpu"):
    import h5py

    generator = {
        "stats": lambda dfp, dfs: stats_generator(df_particles=dfp, df_shower=dfs, device=device),
        "grid": lambda dfp, dfs: grid_generator(df_particles=dfp, df_shower=dfs, n=grid_n,
                                                  flatten=True, x_range=x_range, y_range=y_range,
                                                  device=device),
        "circle": lambda dfp, dfs: circle_generator(df_particles=dfp, df_shower=dfs, device=device),
    }[input_type]

    with h5py.File(out_path, "a") as h5file:
        n_done = h5file["X"].shape[0] if "X" in h5file else 0
        t0 = time.time()
        n_skipped = 0
        n_new = 0
        for i, (df_particles, df_shower) in enumerate(
            corsario.iter_shower_batches(dat_file, batch_size=batch_size, strict=strict)
        ):
            if n_skipped + len(df_shower) <= n_done:
                # Already checkpointed in an earlier (interrupted) run --
                # skip the (cheap, no feature computation) re-parsed batch.
                n_skipped += len(df_shower)
                continue

            X_batch, y_batch = generator(df_particles, df_shower)
            X_batch = X_batch.detach().cpu().numpy().astype("f4")
            y_batch = y_batch.detach().cpu().numpy().astype("f4")

            X_ds, y_ds = _get_or_create_h5(h5file, X_batch.shape[1])
            _append(X_ds, X_batch)
            _append(y_ds, y_batch)
            n_new += X_batch.shape[0]

            if i % progress_every == 0:
                elapsed = time.time() - t0
                rate = n_new / max(elapsed, 1e-9)
                print(f"[{input_type}] batch {i}: {X_ds.shape[0]} showers total "
                      f"(+{n_new} this run), {rate:.0f} showers/s")

        total = h5file["X"].shape[0] if "X" in h5file else 0
        print(f"[{input_type}] done: {total} showers -> {out_path} "
              f"({time.time()-t0:.1f}s this run)")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dat-file", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--input-types", nargs="+", default=list(INPUT_TYPES), choices=INPUT_TYPES)
    parser.add_argument("--batch-size", type=int, default=50_000)
    parser.add_argument("--grid-n", type=int, default=8, help="Grid side length (only for input_type=grid)")
    parser.add_argument("--x-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"),
                         help="Fixed grid x range (only for input_type=grid). If omitted, "
                              "estimated from a sample of the file -- see --range-sample-showers.")
    parser.add_argument("--y-range", type=float, nargs=2, default=None, metavar=("MIN", "MAX"))
    parser.add_argument("--range-percentile", type=float, default=None,
                         help="Use this percentile range instead of true min/max when "
                              "auto-estimating x/y range (e.g. 99.5). Default: true min/max.")
    parser.add_argument("--range-sample-showers", type=int, default=500_000,
                         help="How many leading showers to sample when auto-estimating the grid "
                              "range (default: 500000). Use 0 to scan the whole file instead.")
    parser.add_argument("--tolerant", action="store_true")
    parser.add_argument("--progress-every", type=int, default=1)
    parser.add_argument("--device", default="cpu", help="'cpu' or 'cuda' -- the feature "
                         "generators are torch scatter_add ops, so a GPU can meaningfully "
                         "speed up feature construction on very large files.")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    x_range, y_range = args.x_range, args.y_range
    if "grid" in args.input_types and (x_range is None or y_range is None):
        max_showers = None if args.range_sample_showers == 0 else args.range_sample_showers
        est_x, est_y = compute_global_xy_range(
            args.dat_file, batch_size=args.batch_size,
            percentile=args.range_percentile, max_showers=max_showers,
        )
        x_range = x_range or est_x
        y_range = y_range or est_y
        with open(out_dir / "grid_range.json", "w") as f:
            json.dump({"x_range": x_range, "y_range": y_range}, f)

    for input_type in args.input_types:
        out_path = out_dir / f"{input_type}.h5"
        print(f"=== {input_type} -> {out_path} ===")
        build_one(input_type, args.dat_file, out_path, args.batch_size,
                  grid_n=args.grid_n, x_range=x_range, y_range=y_range,
                  strict=not args.tolerant, progress_every=args.progress_every,
                  device=args.device)


if __name__ == "__main__":
    main()
