"""Design-time simulator for the improved synthetic delay generator (v3).

Read-only: writes nothing.  It answers one question *before* any data is
regenerated - **what accuracy and macro F1 can any classifier possibly reach**
if the synthetic delay process looks like the candidate ``params``?

Per-trip model (delay in minutes):

    mu = latent(category)
       + congestion_base * r_route * band * w_cong(weather) * day_mult
       + weather_add(weather) * r_weather
       + disruption(route, day)          # persistent within the route-day
       + vehicle_shift + systemic(day) + dow_effect + event_extra
    y  = clip(mu + eps, -2, 95)
    eps = N(0, sigma)
        + Bernoulli(p_minor) * U(2, 10)      # routine incidents
        + Bernoulli(p_major) * U(20, 45)     # rare serious incidents

Every term except ``eps`` is *persistent*: chronic per-route congestion, per
route weather sensitivity, per-vehicle reliability and the day-level systemic
factor are learned from a year of earlier departures, and a route-day
disruption regime is observable from the trips that already departed on that
route that day.  The ``ceiling`` observer therefore knows every persistent term
but not ``eps`` - the correct asymptote for a model with a full history feature
battery.  ``no_disruption_knowledge`` is the pessimistic bound (the disruption
regime is unknown, as for the first trips of a disrupted route-day).

Reported per candidate: class mix under the mandated 5/9/30 thresholds, and the
Bayes accuracy / macro F1 both for the accuracy-optimal rule and after tuning
per-class probability multipliers (the same decision adjustment the pipeline
fits on validation).  That output is the reachable (accuracy, macro F1)
frontier - the honest ceiling no honest model can exceed.
"""

from __future__ import annotations

from itertools import product

import numpy as np
from scipy.stats import norm

THRESHOLDS = (5.0, 9.0, 30.0)

# ---- structure of the synthetic process (shares measured from the data) ----
CATEGORY_LATENT = {"core": 2.2, "feeder": 1.2, "express": 1.4}
CATEGORY_SHARES = {"core": 1 / 3, "feeder": 1 / 3, "express": 1 / 3}
BAND_SHARES = {"morning": 0.357, "evening": 0.365, "midday": 0.148, "night": 0.130}
BAND_MULT = {"morning": 1.0, "evening": 0.9, "midday": 0.55, "night": 0.30}
W_CONG = {"clear": 1.0, "rain": 1.25, "snow": 2.2, "storm": 2.8}
WEATHER_ADD = {"clear": 0.0, "rain": 1.5, "snow": 16.0, "storm": 24.0}
WEATHER_SHARES = {"clear": 0.7678, "rain": 0.2131, "snow": 0.0109, "storm": 0.0082}
DAY_MULT = {"weekday": 1.0, "weekend": 0.78}
DAY_SHARES = {"weekday": 0.77, "weekend": 0.23}
DOW_EFFECT = {"mon": 0.8, "tue": 0.0, "wed": 0.0, "thu": 0.0, "fri": 0.5,
              "sat": -0.3, "sun": -0.5}
DOW_NAMES = list(DOW_EFFECT)
EVENT_PROB = 0.030
EVENT_DELAY = 6.0
EVENT_NOISE = 0.30

# disruption regime magnitudes (minutes added to every trip of the route-day)
DISRUPTION_VALUES = np.array([0.0, 4.0, 9.0, 16.0, 26.0])
DISRUPTION_PROPORTIONS = np.array([0.40, 0.28, 0.18, 0.14])

N_MC = 150_000
MINOR_GRID = np.linspace(2.0, 10.0, 33)
MAJOR_GRID = np.linspace(20.0, 45.0, 26)

DEFAULT = {
    "latent": CATEGORY_LATENT,
    "congestion_base": 2.6,
    "chronic_sd": 1.0,
    "sigma": 1.2,
    "route_congestion_cv": 0.25,
    "vehicle_sd": 0.8,
    "systemic_sd": 1.0,
    "p_disrupt": 0.12,
    "unseen_share": 0.10,
    "p_minor": 0.018,
    "p_major": 0.002,
    "w_cong": W_CONG,
    "weather_add": WEATHER_ADD,
    "disruption_values": DISRUPTION_VALUES,
    "disruption_proportions": DISRUPTION_PROPORTIONS,
}

_CDF_CACHE: dict[tuple, tuple[np.ndarray, np.ndarray]] = {}


def residual_cdf(params: dict) -> tuple[np.ndarray, np.ndarray]:
    """Grid CDF of the per-trip residual ``eps``."""
    key = (round(params["sigma"], 4), params["p_minor"], params["p_major"])
    cached = _CDF_CACHE.get(key)
    if cached is not None:
        return cached
    sigma = params["sigma"]
    grid = np.linspace(-90.0, 130.0, 13001)
    step = grid[1] - grid[0]
    pdf = (1.0 - params["p_minor"] - params["p_major"]) * norm.pdf(grid, 0.0, sigma)
    for prob, nodes in ((params["p_minor"], MINOR_GRID), (params["p_major"], MAJOR_GRID)):
        conv = np.zeros_like(grid)
        for u in nodes:
            conv += norm.pdf(grid - u, 0.0, sigma)
        pdf = pdf + prob * conv * (step / len(nodes))
    result = (grid, np.cumsum(pdf) * step)
    _CDF_CACHE[key] = result
    return result


def draw_mu(params: dict, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(mu_ceiling, mu_no_disruption)`` for ``N_MC`` synthetic trips."""
    n = N_MC
    cats = rng.choice(list(CATEGORY_SHARES), size=n, p=list(CATEGORY_SHARES.values()))
    latent = np.array([params["latent"][c] for c in cats])

    band = rng.choice(list(BAND_SHARES), size=n, p=list(BAND_SHARES.values()))
    band_m = np.array([BAND_MULT[b] for b in band])
    day = rng.choice(list(DAY_SHARES), size=n, p=list(DAY_SHARES.values()))
    day_m = np.array([DAY_MULT[d] for d in day])
    weather = rng.choice(list(WEATHER_SHARES), size=n, p=list(WEATHER_SHARES.values()))
    w_cong = np.array([params["w_cong"][w] for w in weather])
    w_add = np.array([params["weather_add"][w] for w in weather])
    dow_e = np.array([DOW_EFFECT[d] for d in rng.choice(DOW_NAMES, size=n)])

    ev_delay = np.where(rng.random(n) < EVENT_PROB, EVENT_DELAY, 0.0)
    ev_delay *= 1.0 + rng.normal(0, EVENT_NOISE, n)

    r_route = np.exp(rng.normal(0, params["route_congestion_cv"], n))
    chronic = rng.normal(0, params["chronic_sd"], n)
    r_weather = 1.0 + rng.normal(0, 0.20, n)
    vehicle = rng.normal(0, params["vehicle_sd"], n)
    systemic = rng.normal(0, params["systemic_sd"], n)

    values, proportions = params["disruption_values"], params["disruption_proportions"]
    disrupted = rng.random(n) < params["p_disrupt"]
    magnitude = rng.choice(values[1:], size=n, p=proportions / proportions.sum())
    disruption = np.where(disrupted, magnitude, 0.0)
    seen = rng.random(n) > params["unseen_share"]

    congestion = params["congestion_base"] * r_route * band_m * w_cong * day_m
    base = (latent + congestion + w_add * r_weather + chronic + vehicle
            + systemic + dow_e + ev_delay)
    return base + disruption, base + np.where(seen, disruption, 0.0)


def class_probabilities(mu: np.ndarray, params: dict) -> np.ndarray:
    grid, cdf = residual_cdf(params)
    cum = [np.zeros(len(mu))]
    for t in THRESHOLDS:
        cum.append(np.interp(t - mu, grid, cdf, left=0.0, right=1.0))
    cum.append(np.ones(len(mu)))
    probs = np.clip(np.diff(np.stack(cum, axis=1), axis=1), 0.0, None)
    return probs / probs.sum(axis=1, keepdims=True)


def expected_confusion(probs: np.ndarray, multipliers: np.ndarray) -> np.ndarray:
    """Expected confusion matrix under ``argmax(multipliers * P(class|mu))``."""
    pred = (probs * multipliers[None, :]).argmax(axis=1)
    cm = np.zeros((4, 4))
    for k in range(4):
        cm[k, :] = np.bincount(pred, weights=probs[:, k], minlength=4)
    return cm


def metrics(cm: np.ndarray) -> dict:
    acc = float(np.trace(cm) / cm.sum())
    tp = np.diag(cm)
    prec = np.divide(tp, cm.sum(axis=0), out=np.zeros(4), where=cm.sum(axis=0) > 0)
    rec = np.divide(tp, cm.sum(axis=1), out=np.zeros(4), where=cm.sum(axis=1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0)
    return {"accuracy": round(acc, 4), "macro_f1": round(float(f1.mean()), 4),
            "per_class_f1": [round(float(x), 3) for x in f1]}


_GRID = np.concatenate([np.linspace(0.4, 3.0, 14), np.linspace(3.4, 12.0, 10)])


def tune_multipliers(probs: np.ndarray) -> tuple[dict, np.ndarray]:
    """Coordinate ascent on per-class probability multipliers (validation rule)."""
    mult = np.ones(4)
    best = metrics(expected_confusion(probs, mult))
    improved = True
    while improved:
        improved = False
        for k in range(4):
            for g in _GRID:
                trial = mult.copy()
                trial[k] = g
                m = metrics(expected_confusion(probs, trial))
                if m["macro_f1"] > best["macro_f1"] + 1e-6:
                    best, mult, improved = m, trial, True
    return best, mult


def evaluate(name: str, params: dict, seed: int = 7) -> dict:
    rng = np.random.default_rng(seed)
    mu, mu_nd = draw_mu(params, rng)
    probs = class_probabilities(mu, params)
    tuned, mult = tune_multipliers(probs)
    p_delay = probs[:, 1:].sum(axis=1)
    return {
        "name": name,
        "class_mix": {c: round(float(x), 4) for c, x in
                      zip(["on time", "moderate", "severe", "critical"], probs.mean(axis=0))},
        "accuracy_rule": metrics(expected_confusion(probs, np.ones(4))),
        "tuned_rule": tuned,
        "multipliers": [round(float(x), 2) for x in mult],
        "no_disruption_knowledge": {
            "class_mix": [round(float(x), 4) for x in
                          class_probabilities(mu_nd, params).mean(axis=0)],
            "tuned_rule": tune_multipliers(class_probabilities(mu_nd, params))[0],
        },
        "binary": {"accuracy_rule": round(float(np.maximum(p_delay, 1 - p_delay).mean()), 4),
                   "delayed_share": round(float(p_delay.mean()), 4)},
    }


DIS_R = np.array([0.0, 5.0, 11.0, 20.0, 32.0])
W_ADD_WIDE = {"clear": 0.0, "rain": 2.5, "snow": 16.0, "storm": 28.0}
W_CONG_WIDE = {"clear": 1.0, "rain": 1.6, "snow": 2.6, "storm": 3.4}
LATENT_LOW = {"core": 1.2, "feeder": 0.6, "express": 0.8}


# variants: the wide-regime family keeps the always-on terms small and lets
# severe weather / route-day disruption create the severe and critical classes
WIDE = dict(DEFAULT, p_disrupt=0.12, disruption_values=DIS_R,
            weather_add=W_ADD_WIDE, w_cong=W_CONG_WIDE, p_major=0.0015)


def variants() -> list[tuple[str, dict]]:
    out = [("K1: wide + chronic route sd",
            dict(WIDE, latent=LATENT_LOW, congestion_base=1.8, chronic_sd=1.0))]
    out.append(("K2: K1 tighter residual 1.0",
                dict(WIDE, latent=LATENT_LOW, congestion_base=1.8, chronic_sd=1.0,
                     sigma=1.0)))
    out.append(("K3: K2 + stronger chronic route sd 1.4",
                dict(WIDE, latent=LATENT_LOW, congestion_base=1.8, chronic_sd=1.4,
                     sigma=1.0)))
    out.append(("K4: K2 with congestion base 2.6",
                dict(WIDE, latent=LATENT_LOW, congestion_base=2.6, chronic_sd=1.0,
                     sigma=1.0)))
    return out


if __name__ == "__main__":
    for name, params in variants():
        r = evaluate(name, params)
        print(f"\n=== {name} ===")
        print("  mix       ", r["class_mix"])
        print("  acc-rule  ", r["accuracy_rule"])
        print("  tuned     ", r["tuned_rule"], "mult", r["multipliers"])
        print("  no-disrupt", r["no_disruption_knowledge"]["tuned_rule"])
        print("  binary    ", r["binary"])
