"""
visualize_embeddings.py
------------------------
Projects the transformer's learned user embeddings into 2D with t-SNE and
colors by the LATENT engagement_type (dormant/casual/core) - a label the
model never saw during training. If the embeddings cluster by this label,
it's evidence the transformer learned a meaningful behavioral
representation purely from watch sequences, not just memorized noise.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main():
    embs = np.load(os.path.join(DATA_DIR, "user_embeddings.npy"))
    emb_ids = np.load(os.path.join(DATA_DIR, "user_embedding_ids.npy"))
    users = pd.read_csv(os.path.join(DATA_DIR, "users.csv"))

    emb_df = pd.DataFrame({"user_id": emb_ids})
    emb_df = emb_df.merge(users[["user_id", "engagement_type"]], on="user_id", how="left")

    print("Running t-SNE (this takes a minute for 4000 points)...")
    coords = TSNE(n_components=2, perplexity=40, random_state=42, init="pca").fit_transform(embs)

    fig, ax = plt.subplots(figsize=(7, 6))
    colors = {"dormant": "#d97757", "casual": "#e8c07d", "core": "#7fa8a3"}
    for etype, color in colors.items():
        mask = emb_df["engagement_type"] == etype
        ax.scatter(coords[mask, 0], coords[mask, 1], s=8, alpha=0.6, c=color, label=etype)
    ax.set_title("Learned user embeddings (t-SNE)\ncolored by latent engagement type (unseen during training)")
    ax.legend()
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()

    out_path = os.path.join(OUT_DIR, "embedding_tsne.png")
    plt.savefig(out_path, dpi=150)
    print(f"Saved plot to {out_path}")


if __name__ == "__main__":
    main()
