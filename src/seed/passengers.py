"""Passenger master data - the travelling population.

Passenger records carry a home zone (tied to the stop zones) and a travel
profile so that boarding/alighting behaviour and later segmentation behave
consistently across the generated data.
"""

import numpy as np
import pandas as pd

from src.log import get_logger

logger = get_logger(__name__)

PASS_TYPES = ["monthly", "weekly", "day", "single"]


def build_passengers(n_passengers: int, stops: pd.DataFrame, seed: int) -> pd.DataFrame:
    """Create passenger master records."""
    rng = np.random.default_rng(seed)
    zones = stops["zone"].unique()

    home_zone = rng.choice(zones, size=n_passengers, p=[0.3, 0.32, 0.1, 0.18, 0.1])
    birth_year = rng.integers(1955, 2008, size=n_passengers)
    age_group = np.select(
        [birth_year >= 2003, birth_year >= 1985, birth_year >= 1972],
        ["18-24", "25-39", "40-52"],
        default="53+",
    )
    pass_type = rng.choice(PASS_TYPES, size=n_passengers, p=[0.4, 0.2, 0.25, 0.15])
    # peakiness: how strongly the passenger travels at rush hour
    peakness = rng.beta(2.0, 2.2, n_passengers).round(3)
    freq_mean = np.select(
        [pass_type == "monthly", pass_type == "weekly", pass_type == "day"],
        [38.0, 18.0, 7.0],
        default=3.0,
    ) * peakness
    annual_trips = rng.poisson(np.maximum(freq_mean, 1.0) * rng.uniform(0.85, 1.15, n_passengers))
    annual_trips = np.minimum(annual_trips.astype(int), 400)
    annual_trips = np.maximum(annual_trips, 1)  # an active passenger rides at least once

    frame = pd.DataFrame(
        {
            "passenger_id": [f"P-{i + 1:05d}" for i in range(n_passengers)],
            "birth_year": birth_year,
            "age_group": age_group,
            "home_zone": home_zone,
            "pass_type": pass_type,
            "peakness": peakness,
            "annual_trips": annual_trips,
        }
    )
    logger.info("passengers: %d", len(frame))
    return frame