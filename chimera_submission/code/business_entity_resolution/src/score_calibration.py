"""Phase F — score reliability diagnostics (stdlib only).

Calibration selection (raw vs isotonic) must be measured on OOF data, never
assumed. These helpers bin OOF scores without any model fitting so the
/+delta stability discussion in threshold_search has a reliability input.
Isotonic fitting itself lives in oof_predictions.py; this module only
diagnoses whether it is worth evaluating.
"""

from __future__ import annotations


def reliability_bins(
    scores: list[float], labels: list[int], num_bins: int = 10,
) -> list[dict]:
    """Mean predicted vs empirical positive rate per bin (stdlib only)."""
    if len(scores) != len(labels):
        raise ValueError("scores and labels must align")
    if num_bins < 1:
        raise ValueError("num_bins must be positive")
    for s in scores:
        if isinstance(s, bool) or not 0 <= s <= 1:
            raise ValueError("scores must be probabilities")
    for y in labels:
        if y not in (0, 1):
            raise ValueError("labels must be binary")
    bins: list[dict] = []
    for b in range(num_bins):
        lo, hi = b / num_bins, (b + 1) / num_bins
        idx = [i for i, s in enumerate(scores)
               if (s >= lo and (s < hi or (b == num_bins - 1 and s <= hi)))]
        if not idx:
            bins.append({"bin": (lo, hi), "count": 0,
                         "mean_score": None, "empirical_rate": None})
            continue
        mean_score = sum(scores[i] for i in idx) / len(idx)
        emp = sum(labels[i] for i in idx) / len(idx)
        bins.append({"bin": (lo, hi), "count": len(idx),
                     "mean_score": mean_score, "empirical_rate": emp})
    return bins


def max_calibration_gap(bins: list[dict]) -> float | None:
    """Max |mean_score - empirical_rate| over non-empty bins."""
    gaps = [abs(b["mean_score"] - b["empirical_rate"])
            for b in bins if b["count"]]
    return max(gaps) if gaps else None
