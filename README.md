# Subscription Companion

Pillar 01 — SML datathon entry. Bank Marketing dataset.

## What it is

A call-center agent picks up a customer record. The Companion tells them:

1. **How likely the customer is to subscribe** — calibrated probability.
2. **Why** — top SHAP drivers translated to plain English.
3. **What to say** — a four-stage conversation flow (opener → talking points → objection pre-empts → close), tuned to the confidence level.

**The agent uses the tool. The tool does not use the agent.** No robocalling.
No ranked call lists. No "who to call next" — only "what to say on the call
the agent is already on."

## Project layout

```
call-copilot/
├── data/
│   ├── raw/bank_additional_clean.csv      # source
│   └── processed/train_ready.csv          # cleaned + engineered
├── src/
│   ├── clean.py            # raw -> processed
│   ├── features.py         # feature config + train/val/test split
│   ├── baselines.py        # dummy + logistic regression
│   ├── train.py            # LightGBM + isotonic calibration + F2 threshold
│   ├── inference.py        # Copilot class used by the UI
│   ├── talking_points.py   # feature -> plain-English driver + hint
│   └── demo_pick.py        # picks high/mid/low demo customers
├── app/copilot.py          # Streamlit agent UI
└── artifacts/              # model, encoders, metrics, threshold
```

## Reproducing the pipeline

```bash
python3 src/clean.py
PYTHONPATH=src python3 src/baselines.py
PYTHONPATH=src python3 src/train.py
PYTHONPATH=src python3 src/demo_pick.py
python3 -m streamlit run app/copilot.py
```

## How class imbalance was handled

The "yes" class is ~11% of the data. Accuracy is meaningless here, so:

1. **Metric**: PR-AUC and F2 (recall-weighted) instead of accuracy.
2. **Class weight**: LightGBM `scale_pos_weight ≈ 7.9` (neg/pos ratio).
3. **Threshold**: chosen on the validation set to maximize F2 — not the
   default 0.5. Final threshold lands around 0.12.
4. **Calibration**: isotonic regression on validation predictions. A 0.68
   prediction really means about 68% of those customers subscribed.
5. **Stratified splits**: 80/10/10 train/val/test, ratio preserved.

## Results

| model              | val PR-AUC | val ROC-AUC | val F2 |
|--------------------|-----------:|------------:|-------:|
| dummy_majority     |       0.11 |        0.50 |   0.00 |
| logreg_balanced    |       0.43 |        0.80 |   0.54 |
| LightGBM + isotonic|       0.43 |        0.82 |   0.57 |

LightGBM test set: PR-AUC 0.47, ROC-AUC 0.81, F2 0.59.

## Leakage

`duration` (call length in seconds) is **excluded** from model features. It
is only known after the call ends, so using it would inflate scores in a way
the agent can never realize at call start. The column is kept in the dataset
for post-call analytics.

## Limits

- Data is from a 2008–2010 Portuguese campaign. Macro features (`euribor3m`,
  `emp_var_rate`) reflect that period. Production use would require
  quarterly recalibration.
- The talking-points dictionary is hand-written. Adding an LLM step to
  generate per-customer openers from the SHAP drivers is the obvious next
  upgrade.
- The model is trained on customers who answered the phone. Customers who
  never picked up are not in the training data.

## Pillar 01 compliance check

- Not a robocaller — the UI is read-only context for a human agent.
- Not an auto-prioritizer — no sorted lead lists, no "top N" displays, no
  lift-at-decile views. Customers are selected by index or manual entry,
  i.e. by the agent's own queue.
- Agent uses tool, not the other way around — every screen is information
  the agent reads while talking, not instructions the agent executes.
