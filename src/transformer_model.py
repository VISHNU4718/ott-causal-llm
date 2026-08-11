"""
transformer_model.py
---------------------
A SASRec-style self-attentive sequential recommender (Kang & McAuley, 2018).
This is the "value-based model that captures how users derive value from the
platform" piece of the JD: it treats a user's watch history as a sequence
(like tokens in a sentence) and uses a causally-masked transformer encoder to
predict the next show they'll watch.

We train it for next-item prediction, then reuse the learned user
representations (last hidden state) as dense embeddings that summarize each
user's "taste + behavior" - these embeddings feed into the causal uplift
model in causal_uplift.py.
"""

import json
import os
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "data")
MAX_LEN = 50
PAD_TOKEN = 0  # show_ids are 0..N_SHOWS-1, so we shift real ids by +1 and reserve 0 for padding


class SequenceDataset(Dataset):
    """Each sample: (input_seq, target_seq) for next-item prediction, causally shifted."""

    def __init__(self, sequences, n_shows, max_len=MAX_LEN):
        self.samples = []
        for entry in sequences:
            seq = [s + 1 for s in entry["sequence"]]  # shift ids, reserve 0 for PAD
            seq = seq[-(max_len + 1):]  # keep most recent
            if len(seq) < 2:
                continue
            self.samples.append((entry["user_id"], seq))
        self.max_len = max_len
        self.n_shows = n_shows

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        user_id, seq = self.samples[idx]
        inp = seq[:-1]
        tgt = seq[1:]
        pad_len = self.max_len - len(inp)
        inp = [PAD_TOKEN] * pad_len + inp
        tgt = [PAD_TOKEN] * pad_len + tgt
        return (
            user_id,
            torch.tensor(inp, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )


class SASRec(nn.Module):
    """Self-attentive sequential model with causal (autoregressive) masking."""

    def __init__(self, n_items, d_model=64, n_heads=2, n_layers=2, max_len=MAX_LEN, dropout=0.2):
        super().__init__()
        self.item_emb = nn.Embedding(n_items + 1, d_model, padding_idx=PAD_TOKEN)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_model * 4,
            dropout=dropout, batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.dropout = nn.Dropout(dropout)
        self.max_len = max_len
        self.d_model = d_model

    def forward(self, seq):
        # seq: (B, L)
        B, L = seq.shape
        positions = torch.arange(L, device=seq.device).unsqueeze(0).expand(B, L)
        x = self.item_emb(seq) + self.pos_emb(positions)
        x = self.dropout(x)

        causal_mask = torch.triu(torch.ones(L, L, device=seq.device) * float("-inf"), diagonal=1)
        # NOTE: we deliberately do NOT also pass a key_padding_mask here. Since padding is
        # left-padded and the causal mask already prevents any real position from being
        # influenced by future tokens, combining both masks would let some padded query
        # positions see an all-masked row (all keys excluded), producing NaN after softmax
        # that then propagates forward through later, real positions via the residual stream.
        # Padded positions' own outputs are unused (loss ignores them via ignore_index=-1),
        # so it's safe to let them attend freely to earlier padding.
        h = self.encoder(x, mask=causal_mask)
        return h  # (B, L, d_model) hidden states

    def predict_logits(self, h):
        # tie weights with item embedding table for scoring (standard SASRec trick)
        return h @ self.item_emb.weight[1:].T  # exclude PAD from vocab of predictable items

    def user_embedding(self, seq):
        """Take the hidden state at the last NON-PAD position as the user's embedding."""
        with torch.no_grad():
            h = self.forward(seq)
            lengths = (seq != PAD_TOKEN).sum(dim=1) - 1
            lengths = lengths.clamp(min=0)
            idx = lengths.view(-1, 1, 1).expand(-1, 1, h.size(-1))
            last_hidden = h.gather(1, idx).squeeze(1)
            return last_hidden


def train_model(epochs=8, batch_size=128, lr=1e-3, device=None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    with open(os.path.join(DATA_DIR, "watch_sequences.json")) as f:
        sequences = json.load(f)
    n_shows = 400  # matches generate_data.py N_SHOWS

    dataset = SequenceDataset(sequences, n_shows)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = SASRec(n_items=n_shows).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=-1)
    # target ids are shifted by -1 to match logits vocab (which excludes PAD);
    # padded target positions become -1 and are ignored by the loss.

    model.train()
    for epoch in range(epochs):
        total_loss, n_batches = 0.0, 0
        for user_id, inp, tgt in loader:
            inp, tgt = inp.to(device), tgt.to(device)
            h = model(inp)
            logits = model.predict_logits(h)  # (B, L, n_shows)

            tgt_shifted = tgt - 1  # map to [0, n_shows-1], PAD(0)-1 = -1 -> ignored
            loss = criterion(logits.reshape(-1, n_shows), tgt_shifted.reshape(-1))

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1
        print(f"Epoch {epoch+1}/{epochs} - loss: {total_loss / n_batches:.4f}")

    return model, dataset, device


def extract_user_embeddings(model, dataset, device):
    """Run every user's sequence through the trained model to get one embedding per user."""
    model.eval()
    loader = DataLoader(dataset, batch_size=256, shuffle=False)
    all_ids, all_embs = [], []
    with torch.no_grad():
        for user_id, inp, tgt in loader:
            inp = inp.to(device)
            emb = model.user_embedding(inp)
            all_ids.extend(user_id.tolist())
            all_embs.append(emb.cpu().numpy())
    embs = np.concatenate(all_embs, axis=0)
    return np.array(all_ids), embs


if __name__ == "__main__":
    model, dataset, device = train_model(epochs=8)
    user_ids, embeddings = extract_user_embeddings(model, dataset, device)
    np.save(os.path.join(DATA_DIR, "user_embeddings.npy"), embeddings)
    np.save(os.path.join(DATA_DIR, "user_embedding_ids.npy"), user_ids)
    print(f"Saved embeddings for {len(user_ids)} users, dim={embeddings.shape[1]}")
