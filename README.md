## Emails

trivedivarun2004@gmail.com, evamewada0@gmail.com, joshiavadh85@gmail.com, nishp876@gmail.com, priyalshah2307@gmail.com, shlokpatel598@gmail.com

# Subscription Companion

**Pillar 01 · SML · Datathon entry**
A real-time copilot that helps a bank call-center agent have a better conversation — not a robocaller, not an auto-prioritizer. The agent picks the customer; the tool reacts with calibrated confidence, plain-English drivers, and an LLM-generated conversation flow.

---

## Table of contents
1. [Quick start](#quick-start)
2. [What this is](#what-this-is)
3. [Architecture](#architecture)
4. [Tech stack](#tech-stack)
5. [How to run](#how-to-run)
6. [Project structure](#project-structure)
7. [Data + features](#data--features)
8. [Model + metrics](#model--metrics)
9. [Validation rigor](#validation-rigor)
10. [Feature choices and trade-offs](#feature-choices-and-trade-offs)
11. [Pillar 01 + Section 08 compliance](#pillar-01--section-08-compliance)
12. [Responsible AI statement](#responsible-ai-statement)
13. [Limitations](#limitations)
14. [Next steps](#next-steps)

---

## Quick start

Assumes Python 3.9+, macOS or Linux. ~5 minutes to run end-to-end.

```bash
# 1. Clone
git clone <repo-url>
cd call-copilot

# 2. Set up isolated environment
python3 -m venv .venv
source .venv/bin/activate                # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. (Optional but recommended) enable the LLM layer
echo "GROQ_API_KEY=your_groq_key_here" > .env
#    Free key at https://console.groq.com/keys — the app falls back to
#    rule-based scripts if this is missing.

# 4. Rebuild every artifact from scratch
python3 src/clean.py                          # raw CSV -> data/processed/train_ready.csv
PYTHONPATH=src python3 src/baselines.py       # dummy + logreg baselines
PYTHONPATH=src python3 src/train.py           # train LightGBM, calibrate, pick threshold
PYTHONPATH=src python3 src/demo_pick.py       # picks high/mid/low demo customers
PYTHONPATH=src python3 src/make_predictions.py  # writes predictions.csv

# 5. Launch the agent UI
python3 -m streamlit run app/copilot.py
#    Opens at http://localhost:8501
```

Already-trained artifacts ship under `artifacts/`, so step 5 alone is enough if you only want to demo.

---

## What this is

A call-center agent in a Portuguese bank is about to call a customer about a term-deposit product. The Subscription Companion does four things, in order:

1. **Scores** the customer — calibrated probability that someone with this profile historically subscribed.
2. **Explains** the score — top three positive + top two negative SHAP drivers translated to plain English.
3. **Writes the call** — opener, pitch paragraph, diagnostic questions, objection pre-empts, and close, all personalized to this customer.
4. **Reacts mid-call** — agent types what the customer just said; tool returns a context-aware counter.

It supports four customer-source modes:
- **Pick from dataset** — existing record (used for evaluation and demo).
- **New customer** — agent fills only what they realistically know; macro / campaign defaults auto-fill.
- **Batch upload** — drop a CSV of customers; the table is rendered without scores; clicking a row triggers scoring + copilot for that one customer.
- **Manual entry (advanced)** — every feature is editable for debugging.

---

## Architecture

```
data/raw/bank_additional_clean.csv
        │
        ▼
src/clean.py  ──►  data/processed/train_ready.csv
        │
        ├──► src/baselines.py    (dummy + logreg)
        │
        └──► src/train.py
                │
                ├──► artifacts/model.pkl         (LightGBM)
                ├──► artifacts/calibrator.pkl    (isotonic regression)
                ├──► artifacts/encoders.pkl      (per-categorical LabelEncoder)
                ├──► artifacts/metrics.json     (PR-AUC, F2, ROC-AUC, threshold)
                └──► artifacts/feature_list.json

         At inference time:
         ┌─────────────────────────────────────────────┐
         │ src/inference.py  Copilot.predict(row)       │
         │   1. encode row with saved encoders          │
         │   2. LightGBM -> raw prob                    │
         │   3. isotonic calibrator -> honest prob      │
         │   4. SHAP TreeExplainer -> drivers           │
         │   5. talking_points.py -> driver → English   │
         │   6. build_flow() drafts a rule-based flow   │
         │   7. src/llm.py (Groq Llama 3.3) replaces    │
         │      opener, pitch, questions, pre-empts,    │
         │      close — if GROQ_API_KEY is set          │
         └─────────────────────────────────────────────┘
                       │
                       ▼
              app/copilot.py (Streamlit)
              ├── Hero strip + avatar + intent pill
              ├── Confidence ring (SVG)
              ├── Why YES / Why NO driver cards
              ├── 5-stage conversation flow
              ├── Live objection handler
              └── Batch upload page + click-to-score
```

---

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| Language | Python 3.9+ | Standard for ML + Streamlit |
| Data | `pandas` 2.x | Tabular manipulation |
| Modeling | `lightgbm` 4.x | Strong on imbalanced tabular, handles categoricals natively |
| Calibration | `scikit-learn` isotonic | Calibrated probabilities — "67%" means 67% |
| Explainability | `shap` 0.49 | Per-prediction feature contributions |
| Hyperparameter sweep | `optuna` 4.x | TPE sampler, 50 trials |
| LLM layer | Groq Llama-3.3-70B | Free tier, ~1 sec latency, generates conversation script |
| UI | `streamlit` 1.50 | Lowest cost path to a polished demo |
| Plots | `matplotlib` | EDA only |
| Env mgmt | `python-dotenv` | Keeps `GROQ_API_KEY` out of source |

---

## How to run

### Run the app only (uses prebuilt artifacts)

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python3 -m streamlit run app/copilot.py
```

### Rebuild every artifact from scratch

```bash
python3 src/clean.py
PYTHONPATH=src python3 src/baselines.py
PYTHONPATH=src python3 src/train.py
PYTHONPATH=src python3 src/demo_pick.py
PYTHONPATH=src python3 src/make_predictions.py
```

### Optional experiments (already run; kept for audit)

```bash
PYTHONPATH=src python3 src/tune.py              # 50-trial Optuna sweep
PYTHONPATH=src python3 src/sweep_pos_weight.py  # scale_pos_weight sweep
```

### Score a CSV via CLI

```bash
PYTHONPATH=src python3 src/batch_score.py \
    --input sample_batch_10.csv \
    --output scored.csv
```

---

## Project structure

```
call-copilot/
├── README.md                  this file
├── requirements.txt           pinned dependencies
├── predictions.csv            test-set predictions (4,118 rows)
├── sample_batch_10.csv        small batch upload demo
├── sample_batch_100.csv       larger batch upload demo
├── data/
│   ├── raw/
│   │   └── bank_additional_clean.csv     UCI Bank Marketing dataset
│   └── processed/
│       └── train_ready.csv               cleaned + engineered
├── src/
│   ├── clean.py                 raw → processed pipeline
│   ├── features.py              feature schema + stratified splitter
│   ├── baselines.py             dummy + logistic regression
│   ├── train.py                 LightGBM + isotonic + F2 threshold
│   ├── inference.py             Copilot class (used by UI + CLI)
│   ├── talking_points.py        feature → plain-English mapping
│   ├── llm.py                   Groq Llama 3.3 wrapper, single-call flow
│   ├── objection_handler.py     9-category objection counters
│   ├── job_mapper.py            free-text job → training category
│   ├── batch_score.py           bulk CSV scoring
│   ├── make_predictions.py      writes predictions.csv
│   ├── demo_pick.py             selects high/mid/low demo customers
│   ├── tune.py                  Optuna 50-trial sweep (audit)
│   └── sweep_pos_weight.py      class-weight sweep (audit)
├── app/
│   └── copilot.py               Streamlit UI (single file)
└── artifacts/
    ├── model.pkl                trained LightGBM
    ├── calibrator.pkl           isotonic calibrator
    ├── encoders.pkl             per-categorical LabelEncoders
    ├── feature_list.json        feature schema
    ├── metrics.json             test metrics + chosen threshold
    ├── demo_customers.json      high/mid/low picks
    ├── optuna_best.json         50-trial Optuna history
    ├── metrics_optuna.json      tuned-model metrics (for comparison)
    ├── model_optuna.pkl         tuned model (kept as audit)
    ├── calibrator_optuna.pkl    matching calibrator
    ├── spw_sweep.json           scale_pos_weight sweep results
    └── baseline_results.csv     dummy + logreg eval
```

---

## Data + features

**Source:** UCI Bank Marketing (Portuguese banking institution, 2008–2010).
**Rows:** 41,176.
**Target:** `y ∈ {yes, no}` — did the customer subscribe to a term deposit. **Class balance: 11.3% yes, 88.7% no.**

### Feature buckets we kept

| Bucket | Features |
|---|---|
| Demographic | `age`, `job`, `marital`, `education` |
| Banking relationship | `default`, `housing`, `loan` |
| Contact metadata | `contact`, `month`, `day_of_week` |
| Campaign history | `campaign`, `pdays` → engineered into `was_contacted_before` + `pdays_clean`, `previous`, `poutcome` |
| Macroeconomic | `emp_var_rate`, `cons_price_idx`, `cons_conf_idx`, `euribor3m`, `nr_employed` |

### Feature buckets we **dropped**

| Field | Why |
|---|---|
| `duration` | Call duration is only known after the call ends — using it leaks the label. Excluded from training features; kept in the dataset for post-call analytics. |

### Engineered features

| Name | Built from | What it captures |
|---|---|---|
| `was_contacted_before` | `pdays` | Binary: customer ever called before |
| `pdays_clean` | `pdays` | Numeric pdays with `999` (never contacted) mapped to NaN so the model treats it correctly |

### Missing-value strategy

- Categorical missing → kept as the string `"unknown"` (signal, not noise — `default = unknown` is heavily predictive)
- Numeric missing (`pdays_clean`) → left as NaN; LightGBM handles NaN natively

---

## Model + metrics

**Final model:** LightGBM gradient boosting with isotonic calibration and an F2-optimal classification threshold.

### Test-set metrics

| Metric | Value |
|---|---|
| PR-AUC | **0.470** |
| ROC-AUC | **0.814** |
| F1 | 0.470 |
| **F2** | **0.589** |
| Precision @ chosen threshold | 0.35 |
| Recall @ chosen threshold | 0.71 |
| Chosen threshold | 0.119 |

### Confusion matrix (test set)

|  | Pred NO | Pred YES |
|---|---|---|
| **Actual NO** | 3048 | 606 |
| **Actual YES** | 135 | **329** |

### Baselines for context

| Model | Val PR-AUC | Val F2 |
|---|---|---|
| Dummy (always-no) | 0.11 | 0.00 |
| Logistic regression + class weights | 0.43 | 0.54 |
| **LightGBM + isotonic + F2 threshold (ours)** | **0.43** | **0.57** |

LightGBM test PR-AUC 0.47, F2 0.59 — see above.

### Why F2, not F1 or accuracy

- **Accuracy lies on imbalanced data.** Always-no scores 89%.
- **F1 weights precision and recall equally.** Missing a yes customer (lost revenue) is materially worse for the bank than calling a no (~60 seconds wasted).
- **F2 weights recall 4× more than precision.** Lines up with the actual business cost asymmetry. We optimize threshold against this.

---

## Validation rigor

Every modeling choice was tested against a holdout, not asserted.

### 1. Stratified 80 / 10 / 10 split
`train_test_split(stratify=y, random_state=42)` twice. Each split preserves the 11.3 % yes ratio. **Test set is touched once, at the end.**

### 2. Early stopping on PR-AUC
LightGBM trains up to 800 trees but stops when val PR-AUC plateaus for 50 rounds. Actual trees used are far fewer than 800; the model picks its own size.

### 3. Calibration
`CalibratedClassifierCV(method="isotonic", cv="prefit")` fitted on the validation set. Probabilities are honest — when the model says 67 %, similar customers historically subscribed at 67 %.

### 4. Threshold tuning
After calibration, we sweep all candidate thresholds along the precision-recall curve and pick the one that maximizes **F2 on the validation set**. The test set is not used for this choice.

### 5. Hyperparameter sweep (Optuna)
50 TPE-sampled trials maximizing val PR-AUC. Best params saved to `artifacts/optuna_best.json`. The tuned model (`artifacts/model_optuna.pkl`) achieved **+0.009 test PR-AUC but -0.004 test F2 vs the hand-tuned baseline**. Because F2 is our primary metric, **we kept the simpler hand-tuned model.** Tuned artifacts are committed as evidence.

### 6. Class-weight sweep
We tested `scale_pos_weight ∈ {8, 10, 12, 15, 20}`. Sweep results (`artifacts/spw_sweep.json`):

| scale_pos_weight | val F2 |
|---|---|
| 7.9 (= neg / pos ratio, **used**) | **0.5658** |
| 8 | 0.5662 |
| 10 | 0.5630 |
| 12 | 0.5638 |
| 15 | 0.5588 |
| 20 | 0.5657 |

The natural class ratio is at or above every tuned alternative — going heavier on the minority class did not help.

---

## Feature choices and trade-offs

This section anticipates the rubric line _"Data prep + Feature Choices and Trade-offs · 10"_ and the Q&A probe.

| Choice | What we did | Why | What we gave up |
|---|---|---|---|
| **Drop `duration`** | Excluded from training features | Leakage — duration only known after the call | Surface PR-AUC numbers (published baselines hit 0.8+ F2 with duration — most are leaky) |
| **Keep `"unknown"` as a category** | Did **not** impute | `default = unknown` is itself predictive; imputing erases signal | Slightly more categorical levels, longer one-hot vectors (not a problem for LightGBM) |
| **Split `pdays`** | Engineered `was_contacted_before` + `pdays_clean` | `999` is a sentinel, not a number; using raw `pdays` confuses the model | Two columns instead of one; tiny cost |
| **F2 over F1** | Optimize F2, tune threshold to F2 | Missing a yes-customer = lost revenue; wasted call = ~60 sec | Lower precision (35 %), more "false alarm" calls |
| **Calibration via isotonic** | Calibrated after training | Agent-facing trust — "67 %" should mean 67 % | Slight loss in raw PR-AUC; gain in honesty |
| **Hand-tuned LightGBM, not Optuna's pick** | Used hand-defaults | Optuna improved PR-AUC by 0.009 but worsened F2; F2 wins for this domain | Slightly lower PR-AUC on test (0.47 vs 0.48) |
| **No SMOTE / oversampling** | Not used | Class-weighted boosting already addresses imbalance; SMOTE often hurts gradient boosters | None measurable on this dataset |
| **No deep learning** | LightGBM only | 41 k tabular rows; LightGBM beats NNs at this scale | Bigger model = unnecessary complexity |
| **Macro features kept** | `emp_var_rate`, `euribor3m`, etc. included | Macro context strongly drove conversion in 2008-2010 data | Couples the model to a moment in time → see Limitations |
| **No additional engineered interactions** | Did not add `age × education`, cyclical month, etc. | Time budget; touches 5+ files; risk of breaking the inference path | Estimated +0.03 F2 left on the table |

---

## Pillar 01 + Section 08 compliance

The brief bans three behaviors for SML apps:
> _Not a robocaller. Not an auto-prioritizer. Not an auto-approve / auto-deny verdict._

How we comply:

| Rule | Implementation |
|---|---|
| No auto-call / no robocaller | The app does not dial. Agent picks customer + makes the call. Tool only displays information. |
| No auto-prioritizer | Batch table is sorted by **name**, not probability. Probability is hidden until a row is clicked. No top-N ranked list. |
| No verdict on a person | Confidence is framed as **"Reference rate from past calls with similar profiles"**. Intent badge says **"Strong / Mixed / Faint signal"**, not "likely yes / no". The historical-fit card carries the line **"Pattern from similar past customers — not a verdict on this person."** |
| Exploration, not approval | All output is read-only context for a human. The agent decides what to do with it. |

---

## Responsible AI statement

_(Required ≤200-word section. Mirrored into the Devpost submission.)_

**Who could be harmed.** A mispredicted low score might cause an agent to under-invest in a customer who would have valued the product, denying them an option that fits. A mispredicted high score might lead the agent to over-pitch to someone who would have preferred to be left alone. Because the training data is from a 2008-2010 Portuguese banking campaign, macro features (employment, Euribor) are baked into the model and may bias scores in any other macro climate.

**What is out of scope.** The Companion does **not** decide who to call, does **not** auto-approve or auto-deny anyone, and does **not** issue a yes/no verdict on a customer. It surfaces a calibrated reference rate from similar historical customers and writes a conversation flow for a human agent to read.

**Guardrails.** (1) `duration` is excluded from training features — no post-call leakage. (2) Probabilities are isotonically calibrated and accompanied by the phrase _"not a verdict on this person."_ (3) The batch view is alphabetical, never probability-ranked. (4) LLM-generated scripts pull only from the explainable SHAP drivers and a constrained product spec; the LLM never touches the score. (5) A rule-based fallback runs when the LLM API is unavailable.

---

## Limitations

- **Macro-economic drift.** Training data is 2008–2010 Portuguese banking. The model encodes that macro era. Production use would need quarterly recalibration with current `euribor3m`, `emp_var_rate`, etc.
- **Inbound-call gap.** The training set is customers who answered an outbound call. Customers who never picked up are not represented.
- **F2 ceiling without leakage.** With `duration` excluded, the realistic F2 ceiling for this dataset sits around 0.65. Published 0.80+ scores almost always use `duration` and overstate real-world performance.
- **LLM hallucination risk.** The LLM can produce plausible-sounding product claims. We pin the prompt to a narrow product spec (fixed rate, principal protected, 3-24 month term) and surface a "rule-based fallback" badge when the API fails.
- **No live A/B yet.** The "agents using the tool vs not" lift is an estimate from offline metrics, not a measured field experiment.

---

## Next steps

- Engineered interactions (`age × education`, `cyclical month`) → estimated F2 +0.03
- Voice / Whisper layer for real-time transcription of the customer side
- Outcome-capture form so each call's result feeds the next retraining cycle
- Monthly retraining cron + drift dashboard once N ≥ 500 new outcomes
- Supervisor-only batch dashboard so agents never see ranked lists
