"""
causal_uplift.py
------------------
This is the "causal inference to identify effective interventions" piece of
the JD. Because assignment to the intervention (personalized recommendation
push) was randomized (RCT-style) in generate_data.py, we can validate our
uplift model against a KNOWN ground-truth treatment effect - this is the
standard way causal ML papers validate their methods on semi-synthetic data
(e.g. as done in the Uplift Modeling / CausalML literature).

We implement two standard estimators:
  - T-Learner: train two separate outcome models (treated vs control), take
    the difference in predicted outcomes as the estimated uplift.
  - X-Learner (Kunzel et al. 2019): improves on T-learner in settings with
    imbalanced treatment groups or heterogeneous effect sizes, by
    cross-imputing individual treatment effects and using a propensity-
    weighted combination.

Covariates: the transformer's learned user embeddings (behavioral signal)
concatenated with engagement_type as a sanity-check feature is intentionally
EXCLUDED from the model inputs - the whole point is to see whether the model
can recover the latent segments purely from behavior embeddings, the way a
real system would have to (you don't get to observe "engagement_type" in
production, only behavior).
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def load_data():
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))
    emb_ids = np.load(os.path.join(DATA_DIR, "user_embedding_ids.npy"))
    embs = np.load(os.path.join(DATA_DIR, "user_embeddings.npy"))

    emb_df = pd.DataFrame(embs, columns=[f"emb_{i}" for i in range(embs.shape[1])])
    emb_df["user_id"] = emb_ids
    merged = users.merge(emb_df, on="user_id", how="inner")
    return merged


def t_learner(X_train, t_train, y_train, X_eval):
    model_treated = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05)
    model_control = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05)

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

    model_tau_treated = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05)
    model_tau_control = GradientBoostingRegressor(n_estimators=150, max_depth=3, learning_rate=0.05)
    model_tau_treated.fit(X_train[treated_mask], d_treated)
    model_tau_control.fit(X_train[control_mask], d_control)

    # Propensity is 0.5 (RCT random assignment) so we can just average the two tau estimates
    tau_treated_pred = model_tau_treated.predict(X_eval)
    tau_control_pred = model_tau_control.predict(X_eval)
    uplift = 0.5 * tau_treated_pred + 0.5 * tau_control_pred
    return uplift


def evaluate_uplift(df, predicted_uplift, model_name):
    df = df.copy()
    df["predicted_uplift"] = predicted_uplift

    # Correlation with ground truth (available because we simulated it)
    corr = np.corrcoef(df["predicted_uplift"], df["true_treatment_effect"])[0, 1]

    # Qini-style decile analysis: sort by predicted uplift, check if top decile
    # really does have higher TRUE treatment effect than bottom decile
    df_sorted = df.sort_values("predicted_uplift", ascending=False).reset_index(drop=True)
    n = len(df_sorted)
    decile_size = n // 10
    decile_true_effects = [
        df_sorted.iloc[i * decile_size:(i + 1) * decile_size]["true_treatment_effect"].mean()
        for i in range(10)
    ]

    print(f"\n=== {model_name} ===")
    print(f"Correlation(predicted uplift, true treatment effect): {corr:.3f}")
    print("Mean TRUE treatment effect by predicted-uplift decile (top -> bottom):")
    for i, val in enumerate(decile_true_effects):
        print(f"  Decile {i+1}: {val:.3f}")

    return corr, decile_true_effects


def plot_results(results, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for name, (corr, deciles) in results.items():
        axes[0].plot(range(1, 11), deciles, marker="o", label=f"{name} (r={corr:.2f})")
    axes[0].set_xlabel("Predicted-uplift decile (1=highest predicted uplift)")
    axes[0].set_ylabel("Mean TRUE treatment effect")
    axes[0].set_title("Uplift model validation vs ground truth")
    axes[0].axhline(0, color="gray", linestyle="--", linewidth=0.8)
    axes[0].legend()

    # Bar chart: baseline retention & true effect by engagement type (context chart)
    seg_summary = pd.DataFrame({
        "engagement_type": ["dormant", "casual", "core"],
        "true_effect": [0.35, 0.12, -0.02],
    })
    axes[1].bar(seg_summary["engagement_type"], seg_summary["true_effect"], color=["#d97757", "#e8c07d", "#7fa8a3"])
    axes[1].set_title("Ground-truth intervention effect by latent user segment")
    axes[1].set_ylabel("True treatment effect on 30-day retention")
    axes[1].axhline(0, color="gray", linestyle="--", linewidth=0.8)

    plt.tight_layout()
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=150)
    print(f"\nSaved plot to {out_path}")


def main():
    df = load_data()
    emb_cols = [c for c in df.columns if c.startswith("emb_")]

    X = df[emb_cols].values
    t = df["treatment"].values
    y = df["retained_30d"].values

    idx_train, idx_eval = train_test_split(np.arange(len(df)), test_size=0.3, random_state=42)
    X_train, t_train, y_train = X[idx_train], t[idx_train], y[idx_train]
    X_eval = X[idx_eval]
    df_eval = df.iloc[idx_eval].reset_index(drop=True)

    results = {}

    uplift_t, _, _ = t_learner(X_train, t_train, y_train, X_eval)
    results["T-Learner"] = evaluate_uplift(df_eval, uplift_t, "T-Learner")

    uplift_x = x_learner(X_train, t_train, y_train, X_eval)
    results["X-Learner"] = evaluate_uplift(df_eval, uplift_x, "X-Learner")

    plot_results(results, os.path.join(OUT_DIR, "uplift_validation.png"))

    # Save a summary CSV of recommended targeting (who to send the intervention to)
    df_eval["predicted_uplift_xlearner"] = uplift_x
    top_targets = df_eval.sort_values("predicted_uplift_xlearner", ascending=False)
    top_targets[["user_id", "engagement_type", "predicted_uplift_xlearner", "true_treatment_effect"]].to_csv(
        os.path.join(OUT_DIR, "recommended_targeting.csv"), index=False
    )
    print(f"\nSaved targeting recommendations to {OUT_DIR}/recommended_targeting.csv")


if __name__ == "__main__":
    main()
