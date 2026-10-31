# OTT Retention Intelligence: transformer embeddings + causal uplift

Given a user's watch history, who should an OTT platform send a retention nudge to, and does the nudge help them? This project builds the two pieces and checks the causal one against a known ground truth:

1. A SASRec-style transformer (PyTorch, trained from scratch) over watch histories. Its hidden states become a 128-dimensional behavioural embedding per user.
2. T-Learner and X-Learner uplift models (scikit-learn) that use those embeddings to estimate how much each user's 30-day retention changes if they are contacted.

**Live results explorer: https://ott-causal-llm.vercel.app** (static page on Vercel's free tier).
The page shows results precomputed by this repository's pipeline. It does not run the models in the browser, and it does not use an LLM or any API key. See [Demo](#demo).

![Uplift validation](outputs/uplift_validation.png)

## Data

Real uplift ground truth ("what would have happened without the nudge") does not exist, so the data is simulated, as is standard when validating uplift methods (`src/generate_data.py`, seed 42):

- 4,000 users and 400 shows with 12 genres.
- Each user has a hidden segment: `dormant` (34%), `casual` (46%), `core` (20%). The segment shapes how much they watch and how consistent their taste is. The models never see it.
- A randomised 50/50 intervention with a known effect on 30-day retention: dormant +0.35, casual +0.12, core -0.02. Control retention is about 14%, 48% and 79% respectively.

## Method

**Transformer** (`src/transformer_model.py`). Causally masked self-attention, learned positional embeddings, tied input/output embeddings, 2 layers, 2 heads, d=64. Padding is masked out of attention. The last item of every history is held out from training. The user embedding is the hidden state at the most recent item concatenated with the mean hidden state over the history.

**Uplift** (`src/causal_uplift.py`). 70/30 train/test split of users. Gradient-boosted outcome models. The X-Learner uses 0.5/0.5 combination weights because assignment was randomised. Scoring is on the 1,200 held-out users, against the simulation's true per-user effect. As a reference, the same X-Learner is also run on a few hand-built features (log history length, mean genre mix, genre entropy).

## Results

Numbers from one run of the pipeline as committed (`outputs/metrics.json`, `data/training_summary.json`):

| Model | Correlation with true effect |
|---|---|
| T-Learner on embeddings | 0.62 |
| X-Learner on embeddings | 0.75 |
| X-Learner on hand-built features (reference) | 0.80 |

X-Learner on embeddings, mean **true** effect by predicted-uplift decile, top to bottom, on held-out users:

| Decile | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| True effect | 0.338 | 0.325 | 0.283 | 0.226 | 0.160 | 0.131 | 0.109 | 0.072 | 0.058 | 0.043 |

The ranking falls from top to bottom: users the model ranks highest do benefit the most.

Two results that are not flattering, and are reported as they came out:

- **The transformer does not predict the next show better than chance on this data.** Hit-rate@10 on each user's held-out last show is 2.3%; random is 2.5% and "most popular" is 2.9%. The simulated histories are close to independent draws, so there is little to predict. The embeddings still separate the segments (t-SNE below), because the shape of a history (length, consistency) differs by segment.
- **Simple hand-built features do slightly better than the embeddings** for uplift (0.80 vs 0.75). On this simulation the transformer adds no accuracy over three or four obvious features.

![Learned embeddings t-SNE](outputs/embedding_tsne.png)

`outputs/recommended_targeting.csv` is the held-out users ranked by X-Learner uplift.

### Change log note

An earlier version of this repository reported correlations of 0.74 (T) and 0.81 (X). That version took each user's embedding from a position index that pointed into the padding for short histories, so the embedding was largely a function of history length. The embedding now comes from the last real position, padding is masked from attention, and the numbers above are from the corrected code.

## Demo

`web/` is a static page (no build step) that reads `web/data.json`:

- targeting simulator: contact the top N% by predicted uplift and see the expected extra retained users against random contact and an oracle,
- decile chart and metrics table,
- segment table,
- t-SNE map of the embeddings,
- browsable held-out users with a lookup by id.

"Expected extra retained users" sums the simulation's true effect over the contacted users. It is only computable because the data is simulated.

Regenerate `web/data.json` after running the pipeline with `python -m src.export_demo`. Serve locally with `python -m http.server -d web`.

## Run it

Python 3.10+.

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python -m src.generate_data          # 1. dataset with ground truth      -> data/
python -m src.transformer_model      # 2. train transformer, embeddings  -> data/ (about 1 min on CPU)
python -m src.causal_uplift          # 3. uplift models and validation   -> outputs/
python -m src.visualize_embeddings   # 4. t-SNE plot                     -> outputs/
python -m src.export_demo            # 5. data for the web page          -> web/data.json
```

Seeds are fixed. Re-running on the same library versions gives the same numbers on CPU; other versions or GPUs can differ slightly.

## Tests

```bash
pip install -r requirements-dev.txt
ruff check . && ruff format --check .
pytest                          # data generation, transformer masking and shapes, uplift scoring, export
node --test tests/*.test.js     # the demo page's targeting maths and input validation
```

CI (`.github/workflows/ci.yml`) runs the same checks.

## Layout

```
src/        generate_data, transformer_model, causal_uplift, visualize_embeddings, export_demo
tests/      pytest and node tests
web/        static results explorer (index.html, app.js, lib.js, data.json)
data/       generated dataset and embeddings (committed so the results are inspectable)
outputs/    plots, metrics.json, recommended_targeting.csv
```

## Limitations

- All data is simulated. This shows the method can recover a known effect. It says nothing about real viewing behaviour.
- Assignment is randomised, so no propensity model is used. Observational data would need one.
- The transformer is trained from scratch and no LLM is involved, despite the repository name. Fine-tuning a pretrained language model on watch histories, or having an LLM explain why a segment responds, are possible next steps and are not done here.
- Single run, single seed; no confidence intervals on the correlations.

## License

MIT
