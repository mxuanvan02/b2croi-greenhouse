#!/usr/bin/env python3
"""Deterministic baseline decision tests for the shared B2CRoI-H(Q) simulator.

These tests verify exact-by-definition classical behavior and document adapted
comparator behavior. They do not claim official third-party reference reproduction.
"""
from __future__ import annotations
import importlib.util
import inspect
from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("b2", ROOT / "scripts" / "run_primary_benchmark.py")
b2 = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(b2)
CHOOSE_PARAMS = list(inspect.signature(b2.choose).parameters)


def choose(policy, xh=None, xt=None, a=None, krel=0, bw=1, pi_bad=0.2, kind="burst"):
    """Call benchmark choose() across known historical signatures.

    Supported signatures include:
    - choose(policy, xh, xt, a, bw, krel, pi_bad, kind, ar)
    - choose(policy, krel, xh, xt, a, bw, pi_bad, kind, ar)
    """
    xh = np.array([25.0, 25.0, 25.0]) if xh is None else np.array(xh, dtype=float)
    xt = np.array([25.0, 25.0, 25.0]) if xt is None else np.array(xt, dtype=float)
    a = np.array([1.0, 2.0, 3.0]) if a is None else np.array(a, dtype=float)
    ar = (np.ones_like(xh) * 0.8, np.zeros_like(xh), np.ones_like(xh) * 0.1)
    n = len(xh)
    kwargs = {
        "policy": policy,
        "xh": xh,
        "xt": xt,
        "a": a,
        "bw": bw,
        "krel": krel,
        "pi_bad": pi_bad,
        "kind": kind,
        "ar": ar,
        "dual_f": 0.0,
        "dual_s": np.zeros(n, dtype=float),
        "counts": np.zeros(n, dtype=float),
        "total_choices": 0,
        "q_s": 0.0,
        "q_f": np.zeros(n, dtype=float),
        "q_e": 0.0,
    }
    filtered = {k: v for k, v in kwargs.items() if k in CHOOSE_PARAMS}
    return list(b2.choose(**filtered))


def test_round_robin_cycle():
    assert choose("round_robin", krel=0) == [0]
    assert choose("round_robin", krel=1) == [1]
    assert choose("round_robin", krel=2) == [2]
    assert choose("round_robin", krel=3) == [0]
    assert choose("round_robin", krel=2, bw=2) == [2, 0]


def test_max_aoi_selects_largest_age():
    assert choose("max_aoi", a=[1, 9, 3]) == [1]


def test_error_trigger_selects_largest_mismatch():
    assert choose("error_trigger", xt=[25, 25, 25], xh=[25, 23, 29], a=[1, 1, 1]) == [2]


def test_generic_voi_age_multiplier_changes_tie():
    assert choose("generic_voi", xt=[26, 26], xh=[25, 25], a=[0, 20]) == [1]


def test_cvoi_sf_prefers_higher_safety_risk_when_mismatch_similar():
    # Both loops have equal absolute mismatch, but 29.8 is near the upper safety bound.
    assert choose("cvoi_sf", xt=[29.8, 25.0], xh=[29.0, 24.2], a=[1, 1]) == [0]


def test_oracle_uses_privileged_loss_reduction():
    assert choose("oracle", xt=[31, 25], xh=[25, 25], a=[1, 1]) == [0]


def test_channel_aware_rr_is_not_per_loop_channel_specific_in_current_model():
    # With scalar channel belief and equal ages, the cyclic freshness term determines ranking.
    assert choose("channel_aware_rr", xh=[25, 25, 25], xt=[25, 25, 25], a=[5, 5, 5], krel=0) == [0]
    # With unequal ages, AoI dominates because predicted success is scalar across loops.
    assert choose("channel_aware_rr", xh=[25, 25, 25], xt=[25, 25, 25], a=[0, 10, 5], krel=0) == [1]


if __name__ == "__main__":
    tests=[v for k,v in globals().items() if k.startswith('test_')]
    for t in tests:
        t()
    print(f"[OK] {len(tests)} baseline decision tests passed")
