# Subscription Companion — Datathon Workflow

**Pillar 01 · SML** · Bank Marketing dataset
**Deadline today: 9:00 PM**
**Goal:** Agent-facing call copilot, not robocaller, not auto-prioritizer. Tool serves agent, agent runs call.

**Pillar 01 rule check:** tool answers "what to say on this call" — never "who to call next." No ranked call lists, no lift@decile, no top-N display.

---

## Phase 0 — Setup (30 min)

- [ ] Confirm Python env: `pandas`, `scikit-learn`, `lightgbm`, `shap`, `streamlit`, `matplotlib`
- [ ] Create folder structure:
  ```
  call-copilot/
    data/        # raw + processed CSVs
    notebooks/   # EDA
    src/         # pipeline + model code
    app/         # Streamlit UI
    artifacts/   # saved model, encoders
  ```
- [ ] Move `bank_additional_clean.csv` → `data/raw/`
- [ ] Git init if not already; commit baseline

---

## Phase 1 — Data Cleaning & EDA (1 hr)

- [ ] Load CSV, profile (shape, dtypes, nulls, dupes)
- [ ] Handle missing values:
  - Keep `"unknown"` as category (signal, not noise)
  - Document choice
- [ ] `pdays` → split into:
  - `was_contacted_before` (binary, 999 = no)
  - `pdays_clean` (numeric, NaN when not contacted)
- [ ] `duration` → mark as leakage, exclude from training features
- [ ] Quick plots: target distribution, key feature vs `y`
- [ ] Save cleaned data → `data/processed/train_ready.csv`

---

## Phase 2 — Baselines (45 min)

- [ ] Stratified 80/10/10 train/val/test split (random_state fixed)
- [ ] Baseline 1: Dummy classifier (always "no") — establishes floor
- [ ] Baseline 2: Logistic regression with class weights
- [ ] Report PR-AUC, ROC-AUC, F1, F2, confusion matrix
- [ ] Lock test set — don't touch until final eval

---

## Phase 3 — Main Model (1.5 hr)

- [ ] LightGBM classifier
  - `scale_pos_weight ≈ 7.9` (36537/4639)
  - Stratified 5-fold CV on train
  - Optuna trials (cap at 30-50 trials, optimize PR-AUC)
- [ ] Calibrate probabilities (isotonic via `CalibratedClassifierCV`)
- [ ] Threshold tuning on val set — optimize F2
- [ ] Compare to baselines in single table
- [ ] Save model + encoders to `artifacts/`

---

## Phase 4 — Explainability Layer (1 hr)

- [ ] SHAP TreeExplainer on LightGBM
- [ ] Compute SHAP per validation row
- [ ] Map top-3 features per customer → plain English:
  - `poutcome=success` → "prior campaign worked — lead with that"
  - `age 30-40` → "career-building stage, emphasize term security"
  - `euribor3m low` → "good rate environment, mention it"
- [ ] Build feature→talking-point dictionary in `src/talking_points.py`

---

## Phase 5 — Agent UI (2 hr)

Streamlit app at `app/copilot.py`. Screens:

- [ ] **Customer lookup**: pick row from validation set, or enter features manually
- [ ] **Score card**:
  - Subscribe probability (calibrated %)
  - Confidence band
  - Color-coded (green/yellow/red)
- [ ] **Why panel**: top 3 drivers, plain English
- [ ] **Talking points**: suggested opener + 2 alternatives, mapped to drivers
- [ ] **Objection handler**: dropdown of common objections → counter-script
- [ ] **Conversation-quality strip**: "tool surfaces 3 personalized drivers per call so agent can tailor pitch in real time"

Tool surfaces info. Agent decides what to say. No auto-dial, no ranked call list, no "who to call next" feature.

---

## Phase 6 — Demo Stories (45 min)

Pick 3 customers from validation set:

- [ ] **High-prob lead** (~70%+): show confident script
- [ ] **Borderline** (~40-60%): show how tool helps agent navigate uncertainty
- [ ] **Low-prob** (~10%): tool says "low chance, here's a soft-touch script — don't push hard"

Walk through each on the UI. Practice talk track.

---

## Phase 7 — Buffer + Polish (30 min)

- [ ] README in repo root
- [ ] Run end-to-end fresh, confirm reproducible
- [ ] Final commit + push

---

## Cut list — do NOT do

- Deep learning (won't beat GBM on 41k tabular)
- 5-model ensemble
- Heavy hyperparam search past 50 trials
- Pretty slides over working demo
- SMOTE unless weighted boosting underperforms

---

## Judging hooks — say these out loud

- "Tool serves agent, agent doesn't serve tool."
- "Calibrated probabilities — 70% means 70%."
- "Optimized PR-AUC and F2, not accuracy — because the rare class is what matters."
- "Tool answers *what to say now*, never *who to call next* — agent owns the queue."
- "Trained on 2008-2010 data; production needs quarterly recalibration."
