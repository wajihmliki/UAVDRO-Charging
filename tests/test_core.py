"""Fast checks of the analytic building blocks (run: python -m pytest -q)."""
import math
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from uavdro.dro import build_model, l1_distances, solve  # noqa: E402
from uavdro.queueing import erlang_c, system_p95  # noqa: E402


def test_erlang_c_mm1_matches_closed_form():
    # M/M/1: P(wait) = rho, decay rate = mu - lam
    pc, rate = erlang_c(6.0, 8.0, 1)
    assert pc == pytest.approx(0.75)
    assert rate == pytest.approx(2.0)


def test_erlang_c_known_value():
    # a = 2 Erlangs, c = 3 servers -> Erlang-C = 4/9
    pc, _ = erlang_c(2.0, 1.0, 3)
    assert pc == pytest.approx(4 / 9, rel=1e-9)


def test_unstable_sites_are_counted_and_excluded():
    lam = {"A": 20.0, "B": 4.0, "C": 3.0}
    p95, unstable = system_p95(lam, {"A": 2, "B": 1, "C": 0}, mu=8.0)
    assert unstable == 2                               # A has rho >= 1, C has no chargers
    # only B remains: M/M/1 with rho = 0.5 -> P(W > t) = 0.5 exp(-4 t) = 0.05 -> t = ln(10)/4 h
    assert p95 == pytest.approx(math.log(10) / 4 * 60, rel=1e-6)


def test_dro_respects_budget_and_stability():
    sites = ["A", "B"]
    S = pd.DataFrame({"sid": ["s0"], "lam_A": [10.0], "lam_B": [2.0]})
    R = pd.DataFrame({"sid": ["s0", "r1"], "lam_A": [10.0, 14.0], "lam_B": [2.0, 1.0]})
    D = l1_distances(S, R, 12.0)
    m = build_model(S, R, D, sites, 0.1, mu=8.0, beta=0.95, c_max=16,
                    bkpts=[0.1, 0.3, 0.5, 0.7, 0.85, 0.95], coeffs=[0, 0.5, 2, 6, 20, 80], total_chargers=6)
    c = solve(m, "highs", 1, 1e-6)["c"]
    assert sum(c.values()) == 6
    assert 14.0 <= 0.95 * 8.0 * c["A"] and 2.0 <= 0.95 * 8.0 * c["B"]
