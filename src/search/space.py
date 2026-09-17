"""Expand a search-space YAML into a flat list of concrete run specs.

See docs/SEARCH.md for the config schema and pipeline/config_search_example.yaml
for a full example. Kept deliberately separate from run_search.py so the
combination logic (grid vs. random sampling, sharding) can be unit-tested/
inspected (e.g. `--dry-run`) without touching any training code.
"""
from __future__ import annotations

import itertools
import random
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

# Which input_type(s) make sense for each architecture: CNN needs the
# spatial "grid" representation (Conv2d over a 2D layout); "stats"/"circle"
# are already-flattened per-shower feature vectors with no spatial
# structure to convolve over.
ARCH_COMPATIBLE_INPUT_TYPES = {
    "mlp": {"stats", "grid", "circle"},
    "cnn": {"grid"},
}

# Architectures that experiments.experiment_runner.ExperimentRunner can
# actually build a model for today. Anything else in the search YAML is
# reported and skipped rather than left to fail later with a confusing
# AttributeError (see ExperimentRunner.build_model()).
IMPLEMENTED_ARCHITECTURES = set(ARCH_COMPATIBLE_INPUT_TYPES.keys())


@dataclass
class RunSpec:
    """Everything needed to train+evaluate exactly one model."""
    run_id: int
    architecture: str
    input_type: str
    layers: int
    neurons: int
    activation: str
    dropout: float
    batchnorm: bool
    loss: str
    optimizer: str
    lr: float
    scheduler_enabled: bool
    scheduler_type: Optional[str]
    batch_size: int
    seed: int

    def as_dict(self) -> Dict[str, Any]:
        return dict(
            run_id=self.run_id, architecture=self.architecture, input_type=self.input_type,
            layers=self.layers, neurons=self.neurons, activation=self.activation,
            dropout=self.dropout, batchnorm=self.batchnorm, loss=self.loss,
            optimizer=self.optimizer, lr=self.lr, scheduler_enabled=self.scheduler_enabled,
            scheduler_type=self.scheduler_type, batch_size=self.batch_size, seed=self.seed,
        )


def _as_list(value) -> list:
    return value if isinstance(value, list) else [value]


def _arch_combinations(arch_name: str, arch_cfg: Dict[str, Any], global_batch_sizes: List[int]) -> List[Dict[str, Any]]:
    scheduler_cfg = arch_cfg.get("scheduler", {"enabled": False, "type": None})
    grid = dict(
        input_type=_as_list(arch_cfg["input_types"]),
        layers=[int(v) for v in _as_list(arch_cfg.get("layers", [2]))],
        neurons=[int(v) for v in _as_list(arch_cfg.get("neurons", [32]))],
        activation=_as_list(arch_cfg.get("activation", ["relu"])),
        dropout=[float(v) for v in _as_list(arch_cfg.get("dropout", [0.0]))],
        batchnorm=[bool(v) for v in _as_list(arch_cfg.get("batchnorm", [False]))],
        loss=_as_list(arch_cfg.get("loss", ["mse"])),
        optimizer=_as_list(arch_cfg.get("optimizer", ["adam"])),
        # PyYAML parses bare exponent notation ("1e-3", no decimal point) as
        # a *string*, not a float -- cast explicitly so a config typo like
        # `lr: [1e-3]` doesn't silently reach torch.optim as a str.
        lr=[float(v) for v in _as_list(arch_cfg.get("lr", [1e-3]))],
        batch_size=[int(v) for v in _as_list(arch_cfg.get("batch_size", global_batch_sizes))],
        seed=[int(v) for v in _as_list(arch_cfg.get("seeds", [1]))],
    )
    keys = list(grid.keys())
    combos = []
    for values in itertools.product(*[grid[k] for k in keys]):
        combo = dict(zip(keys, values))
        combo["architecture"] = arch_name
        combo["scheduler_enabled"] = bool(scheduler_cfg.get("enabled", False))
        combo["scheduler_type"] = scheduler_cfg.get("type")
        combos.append(combo)
    return combos


def build_run_specs(search_config: Dict[str, Any]) -> List[RunSpec]:
    """Expand `architectures:` into the full (or randomly sampled) grid."""
    global_cfg = search_config["search"]
    global_batch_sizes = _as_list(global_cfg.get("batch_size", [64]))
    strategy = global_cfg.get("strategy", "grid")

    all_combos: List[Dict[str, Any]] = []
    skipped = []
    for arch_name, arch_cfg in search_config["architectures"].items():
        if arch_name not in IMPLEMENTED_ARCHITECTURES:
            skipped.append(arch_name)
            continue
        combos = _arch_combinations(arch_name, arch_cfg, global_batch_sizes)
        compatible = ARCH_COMPATIBLE_INPUT_TYPES[arch_name]
        incompatible = {c["input_type"] for c in combos} - compatible
        if incompatible:
            print(f"[search] {arch_name}: dropping input_type(s) {sorted(incompatible)} "
                  f"-- not compatible with this architecture (expected one of {sorted(compatible)})")
            combos = [c for c in combos if c["input_type"] in compatible]
        all_combos.extend(combos)

    if skipped:
        print(f"[search] skipping architecture(s) not yet implemented: {skipped} "
              f"(see ExperimentRunner.build_model / README known gaps)")

    if strategy == "random":
        n_samples = int(global_cfg.get("n_samples", min(50, len(all_combos))))
        rng = random.Random(global_cfg.get("sampling_seed", 0))
        if n_samples < len(all_combos):
            all_combos = rng.sample(all_combos, n_samples)

    specs = [RunSpec(run_id=i, **combo) for i, combo in enumerate(all_combos)]
    return specs


def shard(specs: List[RunSpec], shard_index: int, shard_count: int) -> List[RunSpec]:
    """Return the subset of `specs` assigned to this shard (round-robin).

    Used to split one search across a SLURM job array: each array task
    calls run_search.py with --shard-index $SLURM_ARRAY_TASK_ID
    --shard-count $SLURM_ARRAY_TASK_COUNT, and search/report.py later
    merges the per-shard results*.csv files.
    """
    if shard_count <= 1:
        return specs
    return [s for s in specs if s.run_id % shard_count == shard_index]
