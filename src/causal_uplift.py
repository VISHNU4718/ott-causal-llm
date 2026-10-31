"""
causal_uplift.py
------------------
Heterogeneous treatment-effect (uplift) estimation on the learned user embeddings.

Assignment to the intervention (a personalised recommendation push) was randomised
in generate_data.py, and the simulation records the true per-user effect, so the
estimates can be scored against ground truth on held-out users. That is the usual
way uplift methods are validated on semi-synthetic data.

Estimators:
  - T-Learner: separate outcome models for treated and control; uplift is the
    difference in predicted outcomes.
  - X-Learner (Kunzel et al. 2019): imputes individual effects from the other
    arm's model, fits effect models on them, and combines the two. With a 50/50
    randomised assignment the combination weights are 0.5 / 0.5.

The covariates are the embeddings only. engagement_type is never a model input.
As a reference point, the same X-Learner is also run on a few hand-built
features of the watch history (length, genre mix, genre concentration).

Run:  python -m src.causal_uplift
"""

import json
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def baseline_features(df, sequences, shows):
    """Hand-built history features: log length, mean genre mix of watched shows, genre entropy."""
    genre = shows[[c for c in shows.columns if c.startswith("genre_")]].values
    by_user = {e["user_id"]: e["sequence"] for e in sequences}
    rows = []
    for uid in df["user_id"]:
        seq = by_user[int(uid)]
        mix = genre[seq].mean(axis=0)
        entropy = -(mix * np.log(mix + 1e-12)).sum()
        rows.append(np.concatenate([[np.log(len(seq))], mix, [entropy]]))
    return np.array(rows)


def load_data():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    emb_ids = np.load(os.path.join(DATA_DIR, "user_embedding_ids.npy"))
    embs = np.load(os.path.join(DATA_DIR, "user_embeddings.npy"))

    emb_df = pd.DataFrame(embs, columns=[f"emb_{i}" for i in range(embs.shape[1])])
    emb_df["user_id"] = emb_ids
    merged = users.merge(emb_df, on="user_id", how="inner")
    return merged


def _gbr():
    return GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05, random_state=0)


def t_learner(X_train, t_train, y_train, X_eval):
    model_treated = _gbr()
    model_control = _gbr()

    model_treated.fit(X_train[t_train == 1], y_train[t_train == 1])
    model_control.fit(X_train[t_train == 0], y_train[t_train == 0])

    uplift = model_treated.predict(X_eval) - model_control.predict(X_eval)
    return uplift, model_treated, model_control


def x_learner(X_train, t_train, y_train, X_eval):
    # Stage 1: same as T-learner
    uplift_t, model_treated, model_control = t_learner(X_train, t_train, y_train, X_train)

    # Stage 2: impute individual treatment effects
    treated_mask = t_train == 1
    control_mask = t_train == 0

    d_treated = y_train[treated_mask] - model_control.predict(X_train[treated_mask])
    d_control = model_treated.predict(X_train[control_mask]) - y_train[control_mask]

    model_tau_treated = _gbr()
    model_tau_control = _gbr()
    model_tau_treated.fit(X_train[treated_mask], d_treated)
    model_tau_control.fit(X_train[control_mask], d_control)

    # Propensity is 0.5 (RCT random assignment) so we can just average the two tau estimates
    tau_treated_pred = model_tau_treated.predict(X_eval)
    tau_control_pred = model_tau_control.predict(X_eval)
    uplift = 0.5 * tau_treated_pred + 0.5 * tau_control_pred
    return uplift


def evaluate_uplift(df, predicted_uplift, model_name, verbose=True):
    """Correlation with the true effect, plus mean TRUE effect per predicted-uplift decile (top -> bottom)."""
    df = df.copy()
    df["predicted_uplift"] = np.asarray(predicted_uplift)
    corr = float(np.corrcoef(df["predicted_uplift"], df["true_treatment_effect"])[0, 1])

    df_sorted = df.sort_values("predicted_uplift", ascending=False).reset_index(drop=True)
    decile_size = len(df_sorted) // 10
    deciles = [
        float(df_sorted.iloc[i * decile_size : (i + 1) * decile_size]["true_treatment_effect"].mean())
        for i in range(10)
    ]
    if verbose:
        print(f"\n=== {model_name} ===")
        print(f"Correlation(predicted uplift, true treatment effect): {corr:.3f}")
        print("Mean TRUE treatment effect by predicted-uplift decile (top -> bottom):")
        for i, val in enumerate(deciles):
            print(f"  Decile {i + 1}: {val:.3f}")
    return corr, deciles


def plot_results(results, true_effect_by_segment, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for name, (corr, deciles) in results.items():
        axes[0].plot(range(1, 11), deciles, marker="o", label=f"{name} (r={corr:.2f})")
    axes[0].set_xlabel("Predicted-uplift decile (1=highest predicted uplift)")
    axes[0].set_ylabel("Mean TRUE treatment effect")
    axes[0].set_title("Uplift model validation vs ground truth (held-out users)")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].legend()

    segs = ["dormant", "casual", "core"]
    axes[1].bar(segs, [true_effect_by_segment[k] for k in segs], color=["#d97757", "#e8c07d", "#7fa8a3"])
    axes[1].set_title("Ground-truth intervention effect by latent user segment")
    axes[1].set_ylabel("True treatment effect on 30-day retention")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"\nSaved plot to {out_path}")


def run(df, X, X_baseline, seed=42):
    """Fit both learners on a 70% split; return held-out frame with predictions and the metrics."""
    t = df["treatment"].values
    y = df["retained_30d"].values
    idx_train, idx_eval = train_test_split(np.arange(len(df)), test_size=0.3, random_state=seed)
    df_eval = df.iloc[idx_eval].reset_index(drop=True)

    uplift_t, _, _ = t_learner(X[idx_train], t[idx_train], y[idx_train], X[idx_eval])
    uplift_x = x_learner(X[idx_train], t[idx_train], y[idx_train], X[idx_eval])
    uplift_b = x_learner(X_baseline[idx_train], t[idx_train], y[idx_train], X_baseline[idx_eval])
    df_eval["predicted_uplift_tlearner"] = uplift_t
    df_eval["predicted_uplift_xlearner"] = uplift_x
    df_eval["predicted_uplift_baseline"] = uplift_b

    results = {
        "T-Learner": evaluate_uplift(df_eval, uplift_t, "T-Learner"),
        "X-Learner": evaluate_uplift(df_eval, uplift_x, "X-Learner"),
        "X-Learner on hand-built features": evaluate_uplift(df_eval, uplift_b, "X-Learner on hand-built features"),
    }
    return df_eval, results


def main():
    df = load_data()
    emb_cols = [c for c in df.columns if c.startswith("emb_")]
    with open(os.path.join(DATA_DIR, "watch_sequences.json")) as f:
        sequences = json.load(f)
    shows = pd.read_csv(os.path.join(DATA_DIR, "shows.csv"))

    df_eval, results = run(df, df[emb_cols].values, baseline_features(df, sequences, shows))

    seg_effect = df.groupby("engagement_type")["true_treatment_effect"].first().to_dict()
    plot_results(
        {k: v for k, v in results.items() if k != "X-Learner on hand-built features"},
        seg_effect,
        os.path.join(OUT_DIR, "uplift_validation.png"),
    )

    cols = ["user_id", "engagement_type", "predicted_uplift_xlearner", "true_treatment_effect"]
    df_eval.sort_values("predicted_uplift_xlearner", ascending=False)[cols].to_csv(
        os.path.join(OUT_DIR, "recommended_targeting.csv"), index=False
    )
    with open(os.path.join(OUT_DIR, "metrics.json"), "w") as f:
        json.dump({k: {"correlation": v[0], "deciles": v[1]} for k, v in results.items()}, f, indent=2)
    print(f"\nSaved targeting recommendations and metrics to {OUT_DIR}")


if __name__ == "__main__":
    main()
