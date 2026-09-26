"""C3 crossover [mask−ERM]_TrapB − [mask−ERM]_TrapA from saved spec predictions (seed clusters)."""
import json, sys
import pandas as pd
from wtss.stats import difference_of_deltas
d = sys.argv[1]
p = pd.read_csv(f"{d}/predictions.csv.gz")
r = difference_of_deltas(p[p.trap == "trapB"], p[p.trap == "trapA"], "mask", "erm", "test_rev", 10000, 11)
open(f"{d}/C3_crossover.json", "w").write(json.dumps(r, indent=2, default=float))
print({k: r[k] for k in ("seed_delta_mean", "ci95_lo", "ci95_hi", "seed_deltas_json")})
