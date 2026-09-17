"""Shared "which direction is better" metadata for the metrics produced by
evaluation.evaluate.evaluate_model(), used by both report.py (ranking) and
run_search.py (Optuna's minimize/maximize direction)."""

METRIC_DIRECTION = {
    "mse": "min", "mae": "min", "rmse": "min", "relative_error": "min",
    "bias": "abs_min", "resolution": "min", "p68": "min", "p95": "min",
    "r2": "max", "correlation": "max", "energy_scale": "closest_to_1",
}


def optuna_direction(metric: str) -> str:
    """Map a metric's direction to Optuna's {"minimize", "maximize"}.
    abs_min/closest_to_1 metrics are minimized on a transformed value (see
    run_search.py's objective(), which applies the same transform before
    returning) -- from Optuna's point of view they are always "minimize".
    """
    return "maximize" if METRIC_DIRECTION.get(metric, "min") == "max" else "minimize"


def transform_for_direction(metric: str, value: float) -> float:
    """Transform a raw metric value into "lower is better", matching
    report.py's `rank()` scoring so Optuna and the report agree on what
    counts as the best trial."""
    direction = METRIC_DIRECTION.get(metric, "min")
    if direction == "abs_min":
        return abs(value)
    if direction == "closest_to_1":
        return abs(value - 1.0)
    if direction == "max":
        return value  # optuna_direction() already returns "maximize" for this case
    return value
