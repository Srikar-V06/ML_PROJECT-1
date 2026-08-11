# Ad Click-Through Rate (CTR) Prediction

Predicts the probability that a user clicks a mobile ad, using the [Avazu CTR
Prediction](https://www.kaggle.com/c/avazu-ctr-prediction) dataset schema —
site/app/device context features plus a binary `click` label.

## Why this project
Ad-tech ranking and bidding systems depend on predicting engagement
likelihood from sparse, high-cardinality categorical data under heavy class
imbalance. This project builds that pipeline end to end — from raw impression
logs to an evaluated, imbalance-aware click-probability model.

## Setup
1. Download `train.gz` from the [Avazu Kaggle
   competition](https://www.kaggle.com/c/avazu-ctr-prediction/data) (or the
   [Criteo dataset](https://www.kaggle.com/c/criteo-display-ad-challenge) —
   pipeline works for either with minor column renaming).
2. Unzip and save as `data/train.csv`. The full file is ~40M rows / 6GB — for
   fast iteration, sample it first (e.g. `pd.read_csv(..., nrows=2_000_000)`)
   and overwrite `data/train.csv` with the sample.
3. `pip install -r requirements.txt`
4. Run `ctr_pipeline.py` or open `ctr_prediction.ipynb` and run all cells.

## Approach
1. **EDA** — class balance, CTR by banner position / device type / connection type.
2. **Feature engineering** — parse `hour` into hour-of-day / day-of-week /
   weekend flag; label-encode categorical fields.
3. **Models** — Logistic Regression baseline, then XGBoost to capture
   non-linear feature interactions.
4. **Evaluation** — AUC-ROC and Log Loss (not accuracy — CTR models feed
   probability-ranked ad auctions, so accuracy at a 0.5 threshold isn't the
   right metric).
5. **Class imbalance** — handled via `class_weight="balanced"` (LR) and
   `scale_pos_weight` (XGBoost), not resampling — avoids generating
   nonsensical synthetic categorical combinations.

## Results
| Model | AUC-ROC | Log Loss |
|---|---|---|
| Logistic Regression | 0.6121 | 0.6693 |
| XGBoost | 0.7500 | 0.5899 |
Note: `day_of_week` and `is_weekend` showed no feature importance on this
2M-row sample — likely because the sample doesn't span multiple full weeks.
Would need the full 40M-row dataset or a wider time slice to evaluate
properly.
## Notes for scaling to the full dataset
- Switch label encoding to feature hashing
  (`sklearn.feature_extraction.FeatureHasher`) to control memory on the full
  40M-row set.
- Use a time-based train/test split (train on earlier hours, test on later
  ones) instead of a random split — more realistic for CTR.

## Files
- `ctr_prediction.ipynb` — full pipeline, pre-executed with outputs.
- `ctr_pipeline.py` — same pipeline as a plain script (jupytext light format).
- `outputs/` — saved plots (EDA, ROC curve, feature importance).

## Stack
Python, pandas, scikit-learn, XGBoost, matplotlib/seaborn.
