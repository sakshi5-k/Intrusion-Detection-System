# ML-Based Intrusion Detection System (IDS)

A machine-learning-driven Intrusion Detection System that classifies network
traffic as **benign** or one of several **attack categories** (DoS, Probe,
R2L, U2R) based on flow-level statistical features — the same style of
features Wireshark/Zeek/CICFlowMeter export from packet captures (duration,
byte counts, packet counts, flag counts, error rates, etc.).

## Project Layout

```
ids_ml/
├── README.md
├── requirements.txt
├── data/
│   └── generate_dataset.py     # synthesizes a labeled traffic dataset
├── src/
│   ├── preprocess.py           # cleaning, encoding, scaling, train/test split
│   ├── train_rf.py             # Random Forest classifier (Scikit-learn)
│   ├── train_dnn.py            # Deep Neural Network classifier (TensorFlow/Keras)
│   ├── evaluate.py             # shared evaluation/report utilities
│   └── predict_live.py         # simulated "live" traffic scoring demo
├── models/                     # trained models + scaler/encoder are saved here
└── notebooks/
    └── exploration.md          # guided EDA walkthrough (markdown, no Jupyter needed)
```

## How it works

1. **Data**: `data/generate_dataset.py` builds a synthetic, NSL-KDD-style flow
   dataset (41 features → reduced to the most discriminative ones) with
   realistic statistical separation between normal traffic and four attack
   families: `DoS`, `Probe`, `R2L`, `U2R`. (Swap this out for a real PCAP-derived
   dataset such as NSL-KDD, CICIDS2017, or your own CICFlowMeter/Zeek export —
   the rest of the pipeline doesn't care where the CSV came from.)
2. **Preprocessing**: categorical features (`protocol_type`, `service`, `flag`)
   are one-hot encoded, numeric features are scaled with `StandardScaler`,
   and the data is split into stratified train/test sets.
3. **Models**:
   - A **Random Forest** classifier — fast, interpretable, strong baseline,
     gives feature importances so you can see *which* traffic features drive
     a detection.
   - A **Deep Neural Network** (Keras/TensorFlow) — a denser feedforward
     network for comparison, with dropout for regularization.
4. **Evaluation**: accuracy, precision/recall/F1 per class, confusion matrix,
   and ROC-AUC (binary attack-vs-normal view).
5. **Live demo**: `predict_live.py` streams rows one at a time (simulating
   packets/flows arriving in real time) and prints an alert whenever the
   model flags an attack, including the predicted attack category and the
   model's confidence.

## Quickstart

```bash
pip install -r requirements.txt

# 1. Generate the labeled dataset
python data/generate_dataset.py

# 2. Preprocess (creates train/test splits + saves the scaler/encoder)
python src/preprocess.py

# 3. Train models
python src/train_rf.py
python src/train_dnn.py

# 4. Evaluate (prints metrics + confusion matrix for both models)
python src/evaluate.py

# 5. Watch simulated live detection
python src/predict_live.py
```

## Using real traffic instead of synthetic data

To point this at real captures:

1. Capture traffic with **Wireshark/tshark** or **Zeek**, then export flow
   statistics with a tool like **CICFlowMeter** (produces a CSV with the
   same kind of duration/byte-count/flag-count features used here), or use
   a public labeled dataset (**NSL-KDD**, **CICIDS2017**, **UNSW-NB15**).
2. Make sure the CSV has a `label` column (`normal` or an attack category).
3. Drop it in `data/traffic.csv` and re-run `src/preprocess.py` — the schema
   in `preprocess.py` is intentionally close to the NSL-KDD column layout so
   minimal renaming is needed.

## Notes on scope

This is an educational/portfolio-grade IDS: it demonstrates the full
ML pipeline (data → features → model → evaluation → simulated deployment)
that a real SOC tool would build on. It is **not** a production packet
sniffer/inline blocker — it doesn't capture live packets itself; it expects
flow-level features as input (which Wireshark/Zeek/CICFlowMeter would
provide from real traffic).
