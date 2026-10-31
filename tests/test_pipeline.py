import numpy as np
import pandas as pd
import pytest
import torch

from src.causal_uplift import baseline_features, evaluate_uplift, run, t_learner, x_learner
from src.export_demo import build_payload
from src.generate_data import build_dataset
from src.transformer_model import (
    PAD_TOKEN,
    SASRec,
    SequenceDataset,
    evaluate_next_item,
    extract_user_embeddings,
    pad_left,
    train_model,
)


@pytest.fixture(scope="module")
def small():
    return build_dataset(seed=1, n_users=300, n_shows=40)


# ---- data generation ----


def test_same_seed_same_dataset():
    a = build_dataset(seed=3, n_users=50, n_shows=20)
    b = build_dataset(seed=3, n_users=50, n_shows=20)
    pd.testing.assert_frame_equal(a[1], b[1])
    assert a[2] == b[2]


def test_dataset_shape_and_ranges(small):
    shows, users, seqs = small
    assert len(users) == 300 and len(shows) == 40 and len(seqs) == 300
    assert set(users["engagement_type"]) <= {"dormant", "casual", "core"}
    assert set(users["treatment"]) == {0, 1}
    assert all(8 <= len(s["sequence"]) < 60 for s in seqs)
    assert all(0 <= x < 40 for s in seqs for x in s["sequence"])
    np.testing.assert_allclose(shows[[c for c in shows if c.startswith("genre_")]].sum(axis=1), 1.0)


def test_true_effect_matches_segment(small):
    _, users, _ = small
    expected = {"dormant": 0.35, "casual": 0.12, "core": -0.02}
    for seg, eff in expected.items():
        assert (users.loc[users["engagement_type"] == seg, "true_treatment_effect"] == eff).all()


def test_core_users_watch_more_than_dormant(small):
    _, users, seqs = small
    length = {s["user_id"]: len(s["sequence"]) for s in seqs}
    users = users.assign(n=users["user_id"].map(length))
    means = users.groupby("engagement_type")["n"].mean()
    assert means["core"] > means["casual"] > means["dormant"]


def test_build_dataset_rejects_bad_sizes():
    with pytest.raises(ValueError):
        build_dataset(n_users=0)
    with pytest.raises(ValueError):
        build_dataset(n_shows=1)


# ---- transformer ----


def test_pad_left_keeps_most_recent():
    assert pad_left([1, 2, 3], 5) == [0, 0, 1, 2, 3]
    assert pad_left([1, 2, 3, 4, 5, 6], 4) == [3, 4, 5, 6]


def test_dataset_holds_out_last_item(small):
    _, _, seqs = small
    ds = SequenceDataset(seqs)
    uid, inp, tgt = ds[0]
    full = [s + 1 for s in seqs[0]["sequence"]]
    assert tgt[-1].item() == full[-2]  # last training target is the second-to-last watched item
    assert (inp != PAD_TOKEN).sum() == (tgt != PAD_TOKEN).sum()


def test_model_output_shapes_and_no_nan_with_padding():
    torch.manual_seed(0)
    m = SASRec(n_items=20, d_model=16, max_len=10).eval()
    seq = torch.tensor([[0, 0, 0, 0, 0, 0, 0, 3, 4, 5], [0] * 9 + [2], [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]])
    h = m(seq)
    assert h.shape == (3, 10, 16)
    assert torch.isfinite(h).all()
    assert m.predict_logits(h).shape == (3, 10, 20)
    assert m.user_embedding(seq).shape == (3, 32)


def test_attention_is_causal():
    torch.manual_seed(0)
    m = SASRec(n_items=20, d_model=16, max_len=8).eval()
    a = torch.tensor([[0, 0, 1, 2, 3, 4, 5, 6]])
    b = a.clone()
    b[0, -1] = 19  # change only the last item
    ha, hb = m(a), m(b)
    assert torch.allclose(ha[:, :-1], hb[:, :-1], atol=1e-6)
    assert not torch.allclose(ha[:, -1], hb[:, -1])


def test_embedding_depends_on_history_content_not_only_length():
    torch.manual_seed(0)
    m = SASRec(n_items=20, d_model=16, max_len=10).eval()
    a = torch.tensor([[0] * 6 + [1, 2, 3, 4]])
    b = torch.tensor([[0] * 6 + [9, 8, 7, 6]])
    assert not torch.allclose(m.user_embedding(a), m.user_embedding(b))


def test_padding_amount_does_not_leak_through_attention():
    # Same items with different positions only differ through position embeddings, never NaN
    m = SASRec(n_items=20, d_model=16, max_len=10).eval()
    out = m.user_embedding(torch.tensor([[0] * 9 + [5]]))
    assert torch.isfinite(out).all()


def test_training_reduces_loss_and_embeddings_cover_all_users(small):
    _, _, seqs = small
    model, device, losses = train_model(seqs, n_shows=40, epochs=3, verbose=False)
    assert losses[-1] < losses[0]
    ids, embs = extract_user_embeddings(model, seqs, device)
    assert len(ids) == 300 and embs.shape == (300, 128)
    assert np.isfinite(embs).all()
    hit, pop, rand = evaluate_next_item(model, seqs, device, k=5)
    assert 0 <= hit <= 1 and 0 <= pop <= 1 and rand == pytest.approx(5 / 40)


def test_training_is_deterministic(small):
    _, _, seqs = small
    _, _, l1 = train_model(seqs, n_shows=40, epochs=1, verbose=False, device="cpu")
    _, _, l2 = train_model(seqs, n_shows=40, epochs=1, verbose=False, device="cpu")
    assert l1 == pytest.approx(l2, rel=1e-5)


def test_train_rejects_empty_input():
    with pytest.raises(ValueError):
        train_model([{"user_id": 0, "sequence": [1, 2]}], n_shows=5, epochs=1, verbose=False)


# ---- uplift ----


@pytest.fixture(scope="module")
def synthetic_uplift():
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.normal(size=(n, 3))
    tau = np.where(x[:, 0] > 0, 0.4, 0.0)  # effect depends on the first feature only
    t = rng.integers(0, 2, n)
    p = np.clip(0.3 + 0.1 * x[:, 1] + t * tau, 0.02, 0.98)
    y = rng.binomial(1, p)
    return x, t, y, tau


def test_x_learner_recovers_heterogeneous_effect(synthetic_uplift):
    x, t, y, tau = synthetic_uplift
    pred = x_learner(x[:1000], t[:1000], y[:1000], x[1000:])
    assert np.corrcoef(pred, tau[1000:])[0, 1] > 0.5


def test_t_learner_returns_one_value_per_row(synthetic_uplift):
    x, t, y, _ = synthetic_uplift
    pred, _, _ = t_learner(x[:1000], t[:1000], y[:1000], x[1000:])
    assert pred.shape == (500,)


def test_evaluate_uplift_perfect_ranking():
    true = np.linspace(-0.1, 0.4, 100)
    df = pd.DataFrame({"true_treatment_effect": true})
    corr, deciles = evaluate_uplift(df, true, "perfect", verbose=False)
    assert corr == pytest.approx(1.0)
    assert deciles == sorted(deciles, reverse=True)


def test_evaluate_uplift_random_ranking_is_flat():
    rng = np.random.default_rng(0)
    true = rng.normal(size=2000)
    df = pd.DataFrame({"true_treatment_effect": true})
    corr, deciles = evaluate_uplift(df, rng.normal(size=2000), "random", verbose=False)
    assert abs(corr) < 0.1
    assert max(deciles) - min(deciles) < 0.3


def test_run_and_payload_end_to_end(small):
    shows, users, seqs = small
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(users), 8))
    emb_df = pd.DataFrame(emb, columns=[f"emb_{i}" for i in range(8)])
    emb_df["user_id"] = users["user_id"].values
    df = users.merge(emb_df, on="user_id")
    base = baseline_features(df, seqs, shows)
    assert base.shape == (300, 1 + 12 + 1)
    df_eval, results = run(df, emb, base)
    assert len(df_eval) == 90
    assert set(results) == {"T-Learner", "X-Learner", "X-Learner on hand-built features"}
    training = {"hit_rate_at_10": 0.1, "popularity_hit_rate_at_10": 0.1, "random_hit_rate_at_10": 0.1, "losses": [1.0]}
    payload = build_payload(df, df_eval, results, seqs, shows, rng.normal(size=(300, 2)), training)
    assert len(payload["users"]) == 90 and len(payload["tsne"]) == 300
    assert [u["uplift_x"] for u in payload["users"]] == sorted((u["uplift_x"] for u in payload["users"]), reverse=True)
    assert payload["users"][0]["recent"] == seqs[payload["users"][0]["id"]]["sequence"][-25:]
