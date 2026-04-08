import pandas as pd
from pathlib import Path

selected = [
    ("mirage",   "2389251_de_mirage",   "2389253_de_mirage"),
    ("inferno",  "2389253_de_inferno",  "2389254_de_inferno"),
    ("dust2",    "2389254_de_dust2",    "2389259_de_dust2"),
    ("nuke",     "2389251_de_nuke",     "2389255_de_nuke"),
    ("ancient",  "2389252_de_ancient",  "2389256_de_ancient"),
    ("overpass", "2389252_de_overpass", "2389255_de_overpass"),
    ("anubis",   "2389261_de_anubis",   "2389262_de_anubis"),
]

p = Path(__file__).parent / "data" / "processed"
for map_name, s1, s2 in selected:
    for stem in (s1, s2):
        df = pd.read_parquet(p / f"{stem}.parquet")
        rounds = df.groupby("round_num", sort=True)["bomb_site"].first()
        parts = [f"R{rn:02d}:{site}" for rn, site in zip(rounds.index, rounds)]
        print(f"[{map_name}] {stem}")
        print("  " + "  ".join(parts))
        print()
