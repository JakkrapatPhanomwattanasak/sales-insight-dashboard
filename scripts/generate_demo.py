"""Generate reproducible synthetic sales records; no real customer data."""

import csv
import random
from pathlib import Path


def main():
    rng = random.Random(42)
    path = Path(__file__).resolve().parents[1] / "src/data_app/data/demo_sales.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["marketing_spend", "visits", "region", "channel", "revenue"])
        for index in range(120):
            spend, visits = rng.randint(200, 5000), rng.randint(100, 2500)
            region, channel = (
                ["North", "Central", "South"][index % 3],
                ["Online", "Retail"][index % 2],
            )
            revenue = 1000 + 2.4 * spend + 8 * visits + (700 if region == "Central" else 0)
            revenue += (500 if channel == "Online" else 0) + rng.gauss(0, 600)
            writer.writerow([spend, visits, region, channel, round(max(0, revenue), 2)])


if __name__ == "__main__":
    main()
