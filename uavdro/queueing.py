"""M/M/c queue metrics (Erlang-C), system p95 of the waiting-time mixture, and a SimPy check."""
from __future__ import annotations

import math
import random

import numpy as np


def erlang_c(lam: float, mu: float, c: int) -> tuple[float, float]:
    """Return (P(wait>0), decay rate c*mu - lam). Unstable (rho>=1) -> (1, 0)."""
    if lam <= 0 or c <= 0:
        return 0.0, 0.0
    a = lam / mu
    rho = a / c
    if rho >= 1.0:
        return 1.0, 0.0
    s, term = 0.0, 1.0
    for n in range(c):
        if n > 0:
            term *= a / n
        s += term
    term *= a / c                      # a^c / c!
    pc = (term / (1.0 - rho)) / (s + term / (1.0 - rho))
    return pc, c * mu - lam


def site_table(lam: dict, c: dict, mu: float) -> list[dict]:
    rows = []
    for j, l in lam.items():
        cj = int(c.get(j, 0))
        rho = (l / (mu * cj)) if cj > 0 else (math.inf if l > 0 else 0.0)
        pc, rate = erlang_c(l, mu, cj)
        wq = (pc / rate * 60.0) if rate > 0 else (math.inf if l > 0 else 0.0)
        rows.append({"site_id": j, "lam_hr": l, "c": cj, "rho": rho, "Wq_mean_min": wq,
                     "unstable": bool(l > 0 and (cj == 0 or rho >= 1.0))})
    return rows


def system_p95(lam: dict, c: dict, mu: float, q: float = 0.95) -> tuple[float, int]:
    """Quantile (min) of the arrival-weighted mixture of M/M/c waits over *stable* sites,
    plus the number of unstable sites (lam>0 with c=0 or rho>=1), which are excluded."""
    parts, unstable = [], 0
    for j, l in lam.items():
        if l <= 0:
            continue
        pc, rate = erlang_c(l, mu, int(c.get(j, 0)))
        if rate <= 0:
            unstable += 1
            continue
        parts.append((l, pc, rate))
    tot = sum(p[0] for p in parts)
    if tot == 0:
        return 0.0, unstable

    def tail(t):
        return sum(l / tot * pc * math.exp(-rate * t) for l, pc, rate in parts)

    if tail(0.0) <= 1 - q:
        return 0.0, unstable
    lo, hi = 0.0, 1.0
    while tail(hi) > 1 - q:
        hi *= 2.0
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if tail(mid) > 1 - q else (lo, mid)
    return 0.5 * (lo + hi) * 60.0, unstable


def simulate_p95(lam: dict, c: dict, mu: float, hours: float, seeds, q: float = 0.95) -> dict:
    """Discrete-event M/M/c simulation of every stable site; pooled waiting-time quantile per seed."""
    import simpy

    def run(seed):
        rng = random.Random(seed)
        waits = []
        for j, l in lam.items():
            cj = int(c.get(j, 0))
            if l <= 0 or cj <= 0 or l / (mu * cj) >= 1.0:
                continue
            env = simpy.Environment()
            server = simpy.Resource(env, capacity=cj)

            def job(env, server=server):
                t0 = env.now
                with server.request() as req:
                    yield req
                    waits.append((env.now - t0) * 60.0)
                    yield env.timeout(rng.expovariate(mu))

            def arrivals(env, l=l):
                while True:
                    yield env.timeout(rng.expovariate(l))
                    env.process(job(env))

            env.process(arrivals(env))
            env.run(until=hours)
        w = np.asarray(waits)
        return float(np.quantile(w, q)) if w.size else 0.0, float(w.mean()) if w.size else 0.0

    res = [run(s) for s in seeds]
    p95s = [r[0] for r in res]
    return {"p95_sim_mean": float(np.mean(p95s)), "p95_sim_std": float(np.std(p95s)),
            "wait_mean_sim": float(np.mean([r[1] for r in res])), "seeds": list(seeds)}
