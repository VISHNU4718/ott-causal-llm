"""
generate_data.py
-----------------
Generates a synthetic OTT platform dataset:
  - Users with latent taste profiles (genre affinities) and a latent "engagement type"
  - Shows with genre embeddings
  - Realistic watch sequences per user (genre affinity + recency/trending bias + noise)
  - A simulated intervention (personalized recommendation push) with a KNOWN
    heterogeneous treatment effect (HTE) on 30-day retention, so we can later
    validate a causal uplift model against ground truth.

This mirrors the Sony JD's ask: "value-based models that capture how users derive
value from the platform" + "causal inference to identify effective interventions."
"""

import numpy as np
import pandas as pd
import json
import os

RNG = np.random.default_rng(42)

N_USERS = 4000
N_SHOWS = 400
N_GENRES = 12
MIN_SEQ_LEN = 8
MAX_SEQ_LEN = 60
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "data")


def generate_shows():
    """Each show has a genre-affinity vector (soft genre mixture) + a base popularity."""
    genre_mix = RNG.dirichlet(alpha=np.ones(N_GENRES) * 0.6, size=N_SHOWS)  # sparse-ish mixtures
    popularity = RNG.pareto(a=2.0, size=N_SHOWS) + 0.1
    popularity = popularity / popularity.sum()
    shows = pd.DataFrame({
        "show_id": np.arange(N_SHOWS),
        "popularity": popularity,
    })
    genre_cols = pd.DataFrame(genre_mix, columns=[f"genre_{i}" for i in range(N_GENRES)])
    return pd.concat([shows, genre_cols], axis=1)


def generate_users():
    """
    Each user has:
      - a genre-affinity vector (what they like)
      - a latent engagement_type in {low, medium, high} which drives base retention
        AND how responsive they are to the intervention (this is the HTE driver)
    """
    affinity = RNG.dirichlet(alpha=np.ones(N_GENRES) * 0.8, size=N_USERS)

    # Latent engagement type -> drives baseline retention AND treatment responsiveness.
    # This is deliberately NOT observed directly by the model - only inferred from behavior.
    engagement_type = RNG.choice(
        ["dormant", "casual", "core"], size=N_USERS, p=[0.35, 0.45, 0.20]
    )

    users = pd.DataFrame({
        "user_id": np.arange(N_USERS),
        "engagement_type": engagement_type,
    })
    affinity_cols = pd.DataFrame(affinity, columns=[f"aff_genre_{i}" for i in range(N_GENRES)])
    return pd.concat([users, affinity_cols], axis=1)


def sample_watch_sequence(user_affinity, shows_df, engagement_type):
    """
    Sample a watch sequence biased by genre affinity + popularity.

    Crucially, engagement_type shapes the BEHAVIOR itself (not just a hidden
    label), so a model observing only watch history has a real signal to
    learn from:
      - core users: long sequences, strongly affinity-driven (consistent taste)
      - casual users: medium sequences, a blend of affinity and popularity
      - dormant users: short, sparse, more random/exploratory sequences
    """
    genre_cols = [c for c in shows_df.columns if c.startswith("genre_")]
    show_genre_matrix = shows_df[genre_cols].values  # (n_shows, n_genres)
    affinity_score = show_genre_matrix @ user_affinity

    if engagement_type == "core":
        seq_len = RNG.integers(45, MAX_SEQ_LEN)
        affinity_weight = 0.9
    elif engagement_type == "casual":
        seq_len = RNG.integers(18, 40)
        affinity_weight = 0.6
    else:  # dormant
        seq_len = RNG.integers(MIN_SEQ_LEN, 16)
        affinity_weight = 0.25  # more exploratory / less consistent taste signal

    score = affinity_weight * affinity_score + (1 - affinity_weight) * shows_df["popularity"].values
    score = np.clip(score, 1e-6, None)
    score = score / score.sum()

    watched = RNG.choice(shows_df["show_id"].values, size=seq_len, replace=True, p=score)
    return watched.tolist()


def simulate_intervention_and_outcome(users_df):
    """
    Simulate a binary intervention (e.g. personalized nudge / recommendation email)
    and a 30-day retention outcome, with a KNOWN heterogeneous treatment effect:

      - dormant users: intervention has a STRONG positive effect (they needed the nudge)
      - casual users: intervention has a MILD positive effect
      - core users: intervention has ~ZERO effect (already maximally engaged / possible mild annoyance)

    Baseline retention probability also varies by engagement_type.
    Random assignment (RCT-style) so we can validate causal estimates cleanly.
    """
    n = len(users_df)
    treatment = RNG.binomial(1, 0.5, size=n)  # random assignment (RCT)

    base_retention = users_df["engagement_type"].map({
        "dormant": 0.15,
        "casual": 0.45,
        "core": 0.80,
    }).values

    true_treatment_effect = users_df["engagement_type"].map({
        "dormant": 0.35,   # big uplift
        "casual": 0.12,    # moderate uplift
        "core": -0.02,     # negligible / slightly negative
    }).values

    noise = RNG.normal(0, 0.05, size=n)
    retention_prob = np.clip(
        base_retention + treatment * true_treatment_effect + noise, 0.01, 0.99
    )
    retained = RNG.binomial(1, retention_prob)

    users_df = users_df.copy()
    users_df["treatment"] = treatment
    users_df["true_treatment_effect"] = true_treatment_effect  # ground truth, for evaluation ONLY
    users_df["retained_30d"] = retained
    return users_df


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    shows_df = generate_shows()
    users_df = generate_users()

    affinity_cols = [c for c in users_df.columns if c.startswith("aff_genre_")]

    sequences = []
    for _, row in users_df.iterrows():
        seq = sample_watch_sequence(
            row[affinity_cols].values.astype(float), shows_df, row["engagement_type"]
        )
        sequences.append({"user_id": int(row["user_id"]), "sequence": seq})

    users_df = simulate_intervention_and_outcome(users_df)

    shows_df.to_csv(os.path.join(OUT_DIR, "shows.csv"), index=False)
    users_df.to_csv(os.path.join(OUT_DIR, "users.csv"), index=False)
    with open(os.path.join(OUT_DIR, "watch_sequences.json"), "w") as f:
        json.dump(sequences, f)

    print(f"Generated {N_USERS} users, {N_SHOWS} shows.")
    print(f"Avg sequence length: {np.mean([len(s['sequence']) for s in sequences]):.1f}")
    print(users_df["engagement_type"].value_counts())
    print(f"Overall retention rate: {users_df['retained_30d'].mean():.3f}")
    print(f"Files written to {OUT_DIR}")


if __name__ == "__main__":
    main()
