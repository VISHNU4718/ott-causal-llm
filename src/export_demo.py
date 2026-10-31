"""
export_demo.py
---------------
Writes web/data.json, the precomputed results the static demo page reads.
Everything in it comes from the pipeline outputs in data/ (run the three pipeline
steps first). Nothing is computed in the browser except sums over this file.

Run:  python -m src.export_demo
"""

import json
import os

import pandas as pd

from .causal_uplift import baseline_features, load_data, run
from .visualize_embeddings import compute_tsne

ROOT = os.path.join(os.path.dirname(__file__), "..")
DATA_DIR = os.path.join(ROOT, "data")
WEB_DIR = os.path.join(ROOT, "web")
HISTORY_SHOWN = 25


def build_payload(df, df_eval, results, sequences, shows, coords, training):
    genre_cols = [c for c in shows.columns if c.startswith("genre_")]
    top_genre = shows[genre_cols].values.argmax(axis=1)
    by_user = {e["user_id"]: e["sequence"] for e in sequences}

    users = []
    for _, r in df_eval.sort_values("predicted_uplift_xlearner", ascending=False).iterrows():
        seq = by_user[int(r["user_id"])]
        users.append(
            {
                "id": int(r["user_id"]),
                "segment": r["engagement_type"],
                "treated": int(r["treatment"]),
                "retained": int(r["retained_30d"]),
                "true_effect": round(float(r["true_treatment_effect"]), 4),
                "uplift_x": round(float(r["predicted_uplift_xlearner"]), 4),
                "uplift_t": round(float(r["predicted_uplift_tlearner"]), 4),
                "n_watched": len(seq),
                "recent": seq[-HISTORY_SHOWN:],
            }
        )

    seg_effect = df.groupby("engagement_type")["true_treatment_effect"].first().to_dict()
    seg_base = df[df["treatment"] == 0].groupby("engagement_type")["retained_30d"].mean().to_dict()
    held = set(df_eval["user_id"])
    tsne = [
        [round(float(x), 2), round(float(y), 2), seg, int(uid in held), int(uid)]
        for (x, y), seg, uid in zip(coords, df["engagement_type"], df["user_id"])
    ]
    return {
        "meta": {
            "n_users": int(len(df)),
            "n_shows": int(len(shows)),
            "n_heldout": int(len(df_eval)),
            "embedding_dim": int(sum(c.startswith("emb_") for c in df.columns)),
            "treated_share": round(float(df["treatment"].mean()), 3),
            "retention_rate": round(float(df["retained_30d"].mean()), 3),
            "segment_true_effect": {k: float(v) for k, v in seg_effect.items()},
            "segment_control_retention": {k: round(float(v), 3) for k, v in seg_base.items()},
            "segment_share": {
                k: round(float(v), 3) for k, v in df["engagement_type"].value_counts(normalize=True).items()
            },
            "training": training,
            "metrics": {
                k: {"correlation": round(v[0], 3), "deciles": [round(d, 3) for d in v[1]]} for k, v in results.items()
            },
        },
        "shows": [{"genre": int(g), "popularity": round(float(p), 5)} for g, p in zip(top_genre, shows["popularity"])],
        "users": users,
        "tsne": tsne,
    }


def main():
    df = load_data()
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    with open(os.path.join(DATA_DIR, "watch_sequences.json")) as f:
        sequences = json.load(f)
    shows = pd.read_csv(os.path.join(DATA_DIR, "shows.csv"))
    with open(os.path.join(DATA_DIR, "training_summary.json")) as f:
        training = json.load(f)

    df_eval, results = run(df, df[emb_cols].values, baseline_features(df, sequences, shows))
    coords = compute_tsne(df[emb_cols].values)
    payload = build_payload(df, df_eval, results, sequences, shows, coords, training)

    os.makedirs(WEB_DIR, exist_ok=True)
    path = os.path.join(WEB_DIR, "data.json")
    with open(path, "w") as f:
        json.dump(payload, f, separators=(",", ":"))
    print(f"Wrote {path} ({os.path.getsize(path) / 1024:.0f} KB, {len(payload['users'])} held-out users)")


if __name__ == "__main__":
    main()
