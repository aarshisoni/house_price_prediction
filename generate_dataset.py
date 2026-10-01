"""
generate_dataset.py
-------------------
Generates a synthetic housing dataset for educational/demo purposes.
This is NOT real housing-market data.

Run:
    python generate_dataset.py
"""

import numpy as np
import pandas as pd

# ── Reproducibility ──────────────────────────────────────────────────────────
RANDOM_SEED = 42
rng = np.random.default_rng(RANDOM_SEED)

N = 150  # number of records

# ── Feature generation ────────────────────────────────────────────────────────
locations = ["Downtown", "City_Center", "Suburban", "Rural"]
location_premium = {
    "City_Center": 1.35,
    "Downtown":    1.20,
    "Suburban":    1.00,
    "Rural":       0.75,
}

area_sqft  = rng.integers(600,  3500, size=N)
bedrooms   = rng.integers(1,    6,    size=N)
bathrooms  = rng.integers(1,    4,    size=N)
stories    = rng.integers(1,    4,    size=N)
parking    = rng.integers(0,    3,    size=N)
age_years  = rng.integers(0,    40,   size=N)
location   = rng.choice(locations, size=N)

# ── Price generation (realistic formula + noise) ──────────────────────────────
# Base price driven by area
base_price = area_sqft * 1200  # ₹1,200 per sq ft base

# Adjustments
bedroom_adj   = bedrooms   * 80_000
bathroom_adj  = bathrooms  * 60_000
stories_adj   = stories    * 40_000
parking_adj   = parking    * 30_000
age_penalty   = age_years  * 15_000   # older → cheaper

# Location multiplier
loc_multiplier = np.array([location_premium[l] for l in location])

# Combine
raw_price = (base_price + bedroom_adj + bathroom_adj
             + stories_adj + parking_adj - age_penalty)
price = raw_price * loc_multiplier

# Add ±10 % random noise
noise = rng.uniform(0.90, 1.10, size=N)
price = (price * noise).astype(int)

# Floor at a minimum realistic price
price = np.clip(price, 500_000, None)

# ── Assemble DataFrame ────────────────────────────────────────────────────────
df = pd.DataFrame({
    "area_sqft": area_sqft,
    "bedrooms":  bedrooms,
    "bathrooms": bathrooms,
    "stories":   stories,
    "parking":   parking,
    "age_years": age_years,
    "location":  location,
    "price":     price,
})

# ── Save ──────────────────────────────────────────────────────────────────────
output_path = "housing_price.csv"
df.to_csv(output_path, index=False)

print(f"[OK] Dataset saved to '{output_path}'")
print(f"   Records  : {len(df)}")
print(f"   Features : {list(df.columns[:-1])}")
print(f"   Target   : price")
print(f"   Price range: Rs {df['price'].min():,} to Rs {df['price'].max():,}")
