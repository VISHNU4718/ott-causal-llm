"""
transformer_model.py
---------------------
A SASRec-style self-attentive sequential recommender (Kang & McAuley, 2018).
A user's watch history is treated as a sequence (like tokens in a sentence)
and a causally-masked transformer encoder predicts the next show.

The model is trained on next-item prediction. The hidden state at the user's
most recent watched item is then reused as a dense behavioural embedding (last hidden state plus a mean over the history);
those embeddings are the covariates for the uplift models in causal_uplift.py.

Run:  python -m src.transformer_model
"""

import json
import os

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MAX_LEN = 50
PAD_TOKEN = 0  # show ids are 0..n_shows-1; real ids are shifted by +1 and 0 is padding
SEED = 42
EPOCHS = 20


def pad_left(tokens, max_len=MAX_LEN):
    """Keep the most recent `max_len` tokens and left-pad with PAD_TOKEN."""
    tokens = tokens[-max_len:]
    return [PAD_TOKEN] * (max_len - len(tokens)) + tokens


class SequenceDataset(Dataset):
    """Next-item prediction samples built from the watch history *minus its last item*.

    The last item of every history is held out for evaluation (see `evaluate_next_item`),
    so the model never trains on it.
    """

    def __init__(self, sequences, max_len=MAX_LEN):
        self.samples = []
        for entry in sequences:
            seq = [s + 1 for s in entry["sequence"]][:-1]  # shift ids, drop held-out last item
            seq = seq[-(max_len + 1) :]
            if len(seq) < 2:
                continue
            self.samples.append((entry["user_id"], seq))
        self.max_len = max_len

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_id, seq = self.samples[idx]
        inp = pad_left(seq[:-1], self.max_len)
        tgt = pad_left(seq[1:], self.max_len)
        return (
            user_id,
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )


class SASRec(nn.Module):
    """Self-attentive sequential model with causal (autoregressive) masking."""

    def __init__(self, n_items, d_model=64, n_heads=2, n_layers=2, max_len=MAX_LEN, dropout=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=PAD_TOKEN)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers, enable_nested_tensor=False)
        self.out_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.d_model = d_model
        # Small embedding init keeps the tied-weight logits near zero at the start of training.
        nn.init.normal_(self.item_emb.weight, std=d_model**-0.5)
        nn.init.normal_(self.pos_emb.weight, std=d_model**-0.5)
        with torch.no_grad():
            self.item_emb.weight[PAD_TOKEN].zero_()

    def attention_mask(self, seq):
        """Boolean mask (B * n_heads, L, L); True = this key is NOT attended to.

        A position may attend to earlier, non-padding positions. Padding positions
        may attend to themselves only, so no row is fully masked (which would give NaN).
        """
        B, L = seq.shape
        causal = torch.triu(torch.ones(L, L, dtype=torch.bool, device=seq.device), diagonal=1)
        key_is_pad = (seq == PAD_TOKEN).unsqueeze(1)  # (B, 1, L)
        mask = causal.unsqueeze(0) | key_is_pad  # (B, L, L)
        eye = torch.eye(L, dtype=torch.bool, device=seq.device).unsqueeze(0)
        mask = mask & ~eye
        return mask.repeat_interleave(self.n_heads, dim=0)

    def forward(self, seq):
        # seq: (B, L) -> hidden states (B, L, d_model)
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.dropout(self.item_emb(seq) + self.pos_emb(positions))
        return self.out_norm(self.encoder(x, mask=self.attention_mask(seq)))

    def predict_logits(self, h):
        # tied weights: score items with the item embedding table (PAD excluded)
        return h @ self.item_emb.weight[1:].T

    @torch.no_grad()
    def user_embedding(self, seq):
        """Behavioural embedding: [hidden state at the most recent item ; mean hidden state over real items].

        Sequences are left-padded, so the most recent item is the last column. The mean over
        non-padding positions adds a summary of the whole history (length, consistency of taste).
        Output dimension is 2 * d_model.
        """
        h = self.forward(seq)
        real = (seq != PAD_TOKEN).unsqueeze(-1).to(h.dtype)
        mean = (h * real).sum(dim=1) / real.sum(dim=1).clamp(min=1)
        return torch.cat([h[:, -1, :], mean], dim=-1)


def train_model(sequences, n_shows, epochs=EPOCHS, batch_size=128, lr=1e-3, device=None, seed=SEED, verbose=True):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(seed)

    dataset = SequenceDataset(sequences)
    if len(dataset) == 0:
        raise ValueError("no sequence has at least 3 items; nothing to train on")
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, generator=generator)

    model = SASRec(n_items=n_shows).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    # Targets are shifted by -1 to index the logits (which exclude PAD); padded targets become -1 and are ignored.
    criterion = nn.CrossEntropyLoss(ignore_index=-1)

    losses = []
    model.train()
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for _, inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            logits = model.predict_logits(model(inp))
            loss = criterion(logits.reshape(-1, n_shows), (tgt - 1).reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        losses.append(total_loss / n_batches)
        if verbose:
            print(f"Epoch {epoch + 1}/{epochs} - loss: {losses[-1]:.4f}")
    return model, device, losses


def extract_user_embeddings(model, sequences, device):
    """One embedding per user, computed from the user's full watch history (most recent MAX_LEN items)."""
    model.eval()
    ids = np.array([e["user_id"] for e in sequences])
    batch = []
    for e in sequences:
        batch.append(pad_left([s + 1 for s in e["sequence"]], model.max_len))
    x = torch.tensor(batch, dtype=torch.long)
    out = []
    with torch.no_grad():
        for i in range(0, len(x), 256):
            out.append(model.user_embedding(x[i : i + 256].to(device)).cpu().numpy())
    return ids, np.concatenate(out, axis=0)


@torch.no_grad()
def evaluate_next_item(model, sequences, device, k=10):
    """Hit-rate@k for predicting each user's held-out last item from the items before it.

    Returns (model_hit_rate, popularity_hit_rate, random_hit_rate). Popularity predicts the k
    shows most watched in the training portion of the histories.
    """
    model.eval()
    n_items = model.item_emb.num_embeddings - 1
    counts = np.zeros(n_items)
    inputs, targets = [], []
    for e in sequences:
        seq = e["sequence"]
        if len(seq) < 3:
            continue
        for s in seq[:-1]:
            counts[s] += 1
        inputs.append(pad_left([s + 1 for s in seq[:-1]], model.max_len))
        targets.append(seq[-1])
    if not inputs:
        raise ValueError("no sequence has at least 3 items")
    x = torch.tensor(inputs, dtype=torch.long)
    y = np.array(targets)
    hits = 0
    for i in range(0, len(x), 256):
        logits = model.predict_logits(model(x[i : i + 256].to(device)))[:, -1, :]
        top = logits.topk(k, dim=-1).indices.cpu().numpy()
        hits += (top == y[i : i + 256, None]).any(axis=1).sum()
    top_pop = set(np.argsort(-counts)[:k].tolist())
    pop_hits = sum(int(t in top_pop) for t in y)
    return hits / len(y), pop_hits / len(y), k / n_items


def main():
    with open(os.path.join(DATA_DIR, "watch_sequences.json")) as f:
        sequences = json.load(f)
    n_shows = int(max(max(e["sequence"]) for e in sequences)) + 1
    shows_path = os.path.join(DATA_DIR, "shows.csv")
    if os.path.exists(shows_path):
        with open(shows_path) as f:
            n_shows = sum(1 for _ in f) - 1

    model, device, losses = train_model(sequences, n_shows)
    hit, pop, rand = evaluate_next_item(model, sequences, device)
    print(f"Held-out last-item hit-rate@10: model {hit:.3f} | most-popular {pop:.3f} | random {rand:.3f}")

    user_ids, embeddings = extract_user_embeddings(model, sequences, device)
    np.save(os.path.join(DATA_DIR, "user_embeddings.npy"), embeddings)
    np.save(os.path.join(DATA_DIR, "user_embedding_ids.npy"), user_ids)
    with open(os.path.join(DATA_DIR, "training_summary.json"), "w") as f:
        json.dump(
            {"losses": losses, "hit_rate_at_10": hit, "popularity_hit_rate_at_10": pop, "random_hit_rate_at_10": rand},
            f,
            indent=2,
        )
    print(f"Saved embeddings for {len(user_ids)} users, dim={embeddings.shape[1]}")


if __name__ == "__main__":
    main()
