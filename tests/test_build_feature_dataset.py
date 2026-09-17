"""Tests for the large-file streaming feature pipeline
(data/build_feature_dataset.py + the grid_generator fixed-range fix).

Requires corsario (pip install -e ".[corsario]") and a small synthetic
CORSIKA .DAT file, generated inline below so these tests don't depend on
having real CORSIKA output available.

Run with: cd src && pytest ../tests -v
"""
from __future__ import annotations

import struct
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np
import pytest

corsario = pytest.importorskip("corsario")

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
sys.path.insert(0, str(SRC_DIR))

from data.load_db import DAT_file  # noqa: E402
from data.preprocessing import data_processed  # noqa: E402


def _tag(s):
    return s.encode("ascii")


def _f(x):
    return struct.pack("<f", float(x))


def _write_synthetic_dat(path: Path, n_showers: int = 200, seed: int = 0) -> None:
    """A minimal but non-degenerate synthetic file: varying energy and
    particle count per shower, random-ish spatial spread -- enough for
    grid_generator's binning and stats_generator's variance-based features
    to be meaningful (a constant-everything file would trivially "pass").
    """
    import math
    import random

    class Writer:
        def __init__(self):
            self.words = []
            self.block_count = 0

        def write_block(self, block_words):
            if self.block_count > 0 and self.block_count % 21 == 0:
                self.words.append(_f(0.0))
                self.words.append(_f(0.0))
            block = list(block_words)
            while len(block) < 273:
                block.append(_f(0.0))
            self.words.extend(block)
            self.block_count += 1

        def to_bytes(self):
            return _f(0.0) + b"".join(self.words) + _f(0.0)

    rng = random.Random(seed)
    w = Writer()
    w.write_block([_tag("RUNH")] + [_f(v) for v in [1] + [0] * 10])

    for n in range(1, n_showers + 1):
        log_e = rng.uniform(4, 6)
        energy = 10 ** log_e
        n_particles = max(5, int(20 + (log_e - 4) * 100 + rng.gauss(0, 10)))
        w.write_block([_tag("EVTH")] + [_f(v) for v in [n, 14, energy, 112.8, 0, 0, 0, 0, -1, 0.1, 0.2]])

        idx = 0
        remaining = n_particles
        while remaining > 0:
            n_this = min(39, remaining)
            block = []
            for _ in range(n_this):
                pid = 6003 + (idx % 5)
                r = abs(rng.gauss(0, 30 + log_e * 10))
                theta = rng.uniform(0, 2 * math.pi)
                x = r * math.cos(theta)
                y = r * math.sin(theta)
                t = 100.0 + idx + rng.gauss(0, 5)
                for v in [pid, 0.1 * idx, 0.2 * idx, 1.0 + 0.01 * idx, x, y, t]:
                    block.append(_f(v))
                idx += 1
            w.write_block(block)
            remaining -= n_this

        w.write_block([_tag("EVTE")] + [_f(v) for v in [n, 0, 0, 0, n_particles]])

    w.write_block([_tag("RUNE")] + [_f(v) for v in [1, n_showers, 0]])
    path.write_bytes(w.to_bytes())


@pytest.fixture()
def synthetic_dat(tmp_path):
    path = tmp_path / "DAT_TEST"
    _write_synthetic_dat(path, n_showers=150)
    return str(path)


def test_grid_generator_is_batch_invariant_with_fixed_range(synthetic_dat):
    """Regression test for the batch-local x/y range bug: computing grid
    features in two separate halves (with a shared, fixed range) must give
    the same result as computing it in one call over the merged data."""
    df_particles, df_shower = DAT_file(synthetic_dat)

    x_range = (float(df_particles["x"].min()), float(df_particles["x"].max()))
    y_range = (float(df_particles["y"].min()), float(df_particles["y"].max()))

    X_all, y_all = data_processed(input_type="grid", df_particles=df_particles,
                                   df_shower=df_shower, x_range=x_range, y_range=y_range)

    mid = df_particles["shower"].unique()[len(df_particles["shower"].unique()) // 2]
    dfp_a = df_particles[df_particles["shower"] < mid]
    dfs_a = df_shower[df_shower["event_no"] < mid]
    dfp_b = df_particles[df_particles["shower"] >= mid]
    dfs_b = df_shower[df_shower["event_no"] >= mid]

    X_a, y_a = data_processed(input_type="grid", df_particles=dfp_a, df_shower=dfs_a,
                               x_range=x_range, y_range=y_range)
    X_b, y_b = data_processed(input_type="grid", df_particles=dfp_b, df_shower=dfs_b,
                               x_range=x_range, y_range=y_range)

    X_concat = np.concatenate([X_a.numpy(), X_b.numpy()], axis=0)
    y_concat = np.concatenate([y_a.numpy(), y_b.numpy()], axis=0)

    assert np.allclose(X_concat, X_all.numpy(), atol=1e-4)
    assert np.allclose(y_concat, y_all.numpy(), atol=1e-4)


def test_grid_generator_batch_local_range_would_disagree(synthetic_dat):
    """Sanity check that the test above is actually exercising the fix:
    WITHOUT a shared fixed range (the old default behavior), splitting into
    two batches gives each one a different x/y range and therefore
    different-meaning grid cells -- so naively concatenating them should
    generally NOT match the single-call result."""
    df_particles, df_shower = DAT_file(synthetic_dat)
    mid = df_particles["shower"].unique()[len(df_particles["shower"].unique()) // 2]
    dfp_a = df_particles[df_particles["shower"] < mid]
    dfp_b = df_particles[df_particles["shower"] >= mid]

    range_a = (dfp_a["x"].min(), dfp_a["x"].max())
    range_b = (dfp_b["x"].min(), dfp_b["x"].max())
    assert range_a != range_b, "test fixture isn't varied enough to demonstrate the bug"


def test_build_feature_dataset_matches_monolithic(tmp_path, synthetic_dat):
    df_particles, df_shower = DAT_file(synthetic_dat)
    x_range = (float(df_particles["x"].min()), float(df_particles["x"].max()))
    y_range = (float(df_particles["y"].min()), float(df_particles["y"].max()))

    out_dir = tmp_path / "features"
    subprocess.run(
        [sys.executable, str(SRC_DIR / "data" / "build_feature_dataset.py"),
         "--dat-file", synthetic_dat, "--out-dir", str(out_dir),
         "--input-types", "stats", "grid", "circle",
         "--batch-size", "37",  # deliberately not a divisor of n_showers
         "--x-range", str(x_range[0]), str(x_range[1]),
         "--y-range", str(y_range[0]), str(y_range[1])],
        check=True, cwd=SRC_DIR,
    )

    for kind, x_r, y_r in [("stats", None, None), ("grid", x_range, y_range), ("circle", None, None)]:
        X_ref, y_ref = data_processed(input_type=kind, df_particles=df_particles,
                                       df_shower=df_shower, x_range=x_r, y_range=y_r)
        with h5py.File(out_dir / f"{kind}.h5") as f:
            X_built, y_built = f["X"][:], f["y"][:]
        assert np.allclose(X_ref.numpy(), X_built, atol=1e-3)
        assert np.allclose(y_ref.numpy(), y_built, atol=1e-3)


@pytest.mark.parametrize("input_type", ["stats", "circle"])
def test_build_feature_dataset_resume_is_lossless(tmp_path, synthetic_dat, input_type):
    """Interrupting after a few batches and resuming must produce exactly
    the same final dataset as an uninterrupted run -- this is the
    correctness guarantee that makes it safe to resubmit a killed/timed-out
    cluster job instead of restarting from scratch."""
    script = str(SRC_DIR / "data" / "build_feature_dataset.py")

    full_dir = tmp_path / "full"
    subprocess.run(
        [sys.executable, script, "--dat-file", synthetic_dat, "--out-dir", str(full_dir),
         "--input-types", input_type, "--batch-size", "40"],
        check=True, cwd=SRC_DIR,
    )

    # Simulate an interruption: seed a partial output with only the first
    # batch's worth of rows, then let build_feature_dataset.py resume it.
    partial_dir = tmp_path / "partial"
    partial_dir.mkdir()
    with h5py.File(full_dir / f"{input_type}.h5") as src, \
         h5py.File(partial_dir / f"{input_type}.h5", "w") as dst:
        n_seed = 40
        dst.create_dataset("X", data=src["X"][:n_seed], maxshape=(None, src["X"].shape[1]))
        dst.create_dataset("y", data=src["y"][:n_seed], maxshape=(None,))

    subprocess.run(
        [sys.executable, script, "--dat-file", synthetic_dat, "--out-dir", str(partial_dir),
         "--input-types", input_type, "--batch-size", "40"],
        check=True, cwd=SRC_DIR,
    )

    with h5py.File(full_dir / f"{input_type}.h5") as f:
        X_full, y_full = f["X"][:], f["y"][:]
    with h5py.File(partial_dir / f"{input_type}.h5") as f:
        X_resumed, y_resumed = f["X"][:], f["y"][:]

    assert X_full.shape == X_resumed.shape
    assert np.allclose(X_full, X_resumed)
    assert np.allclose(y_full, y_resumed)
