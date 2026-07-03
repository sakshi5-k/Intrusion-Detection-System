# Exploratory Data Analysis Walkthrough

This notes file is a guided EDA you can run as plain Python (no Jupyter
required) to get familiar with the dataset before training.

```python
import pandas as pd
df = pd.read_csv("data/traffic.csv")

df["label"].value_counts()          # class balance
df.groupby("label")["duration"].describe()
df.groupby("label")["src_bytes"].median()
df.groupby("label")["serror_rate"].mean()   # SYN-error rate spikes during DoS/Probe
df["protocol_type"].value_counts(normalize=True)
```

## What to look for

- **Class imbalance**: `normal` and `DoS` dominate; `U2R` is rare — this is
  realistic (privilege-escalation attacks are rare events) and is why the
  Random Forest is trained with `class_weight="balanced"`.
- **`serror_rate` / `rerror_rate`**: SYN/connection error rates spike sharply
  during `DoS` and `Probe` traffic — these are typically the most predictive
  features (check `models/rf_feature_importances.json` after training).
- **`num_failed_logins`**: elevated almost exclusively in `R2L` traffic,
  reflecting repeated failed authentication attempts from outside.
- **`count` / `srv_count`**: connection-rate features are the clearest signal
  for volumetric `DoS` traffic.

## Correlating with real packet captures

If you swap in a real CICFlowMeter export, the equivalent columns are
typically named `Flow Duration`, `Total Fwd Packets`, `Total Length of Fwd
Packets`, `SYN Flag Count`, etc. — rename them to match `preprocess.py`'s
expected schema (or edit `CATEGORICAL_COLS`/feature list there) to point
the same pipeline at real traffic.
