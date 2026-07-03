import numpy as np
import pandas as pd
from pathlib import Path
RNG=np.random.default_rng(42)
OUT_PATH=Path(__file__).parent / "traffic.csv"
PROTOCOLS=["tcp", "udp", "icmp"]
SERVICES=["http", "ftp", "smtp", "ssh", "dns", "telnet", "private", "other"]
FLAGS=["SF", "S0", "REJ", "RSTR", "SH"]
N_NORMAL=6000
N_DOS=2500
N_PROBE=1500
N_R2L=800
N_U2R=200
def _base_frame(n, label):
    return pd.DataFrame({
        "label": [label] * n,
        "protocol_type": RNG.choice(PROTOCOLS, n, p=[0.7, 0.25, 0.05]),
        "service": RNG.choice(SERVICES, n),
        "flag": RNG.choice(FLAGS, n, p=[0.6, 0.15, 0.1, 0.1, 0.05]),
    })
def gen_normal(n):
    df = _base_frame(n, "normal")
    df["duration"] = RNG.exponential(scale=5, size=n)
    df["src_bytes"] = RNG.lognormal(mean=6, sigma=1.2, size=n).astype(int)
    df["dst_bytes"] = RNG.lognormal(mean=6.5, sigma=1.2, size=n).astype(int)
    df["count"] = RNG.poisson(lam=8, size=n)
    df["srv_count"] = RNG.poisson(lam=8, size=n)
    df["serror_rate"] = np.clip(RNG.normal(0.02, 0.02, n), 0, 1)
    df["rerror_rate"] = np.clip(RNG.normal(0.02, 0.02, n), 0, 1)
    df["same_srv_rate"] = np.clip(RNG.normal(0.9, 0.08, n), 0, 1)
    df["diff_srv_rate"] = np.clip(RNG.normal(0.05, 0.05, n), 0, 1)
    df["dst_host_count"] = RNG.poisson(lam=20, size=n)
    df["dst_host_srv_count"] = RNG.poisson(lam=20, size=n)
    df["num_failed_logins"] = RNG.poisson(lam=0.02, size=n)
    df["logged_in"] = RNG.choice([0, 1], n, p=[0.2, 0.8])
    return df
def gen_dos(n):
    # High volume, short duration, many connections to same service, high error rates
    df = _base_frame(n, "DoS")
    df["duration"] = RNG.exponential(scale=0.5, size=n)
    df["src_bytes"] = RNG.lognormal(mean=3, sigma=1.0, size=n).astype(int)
    df["dst_bytes"] = RNG.lognormal(mean=1, sigma=0.5, size=n).astype(int)
    df["count"] = RNG.poisson(lam=200, size=n)
    df["srv_count"] = RNG.poisson(lam=200, size=n)
    df["serror_rate"] = np.clip(RNG.normal(0.85, 0.1, n), 0, 1)
    df["rerror_rate"] = np.clip(RNG.normal(0.1, 0.1, n), 0, 1)
    df["same_srv_rate"] = np.clip(RNG.normal(0.98, 0.03, n), 0, 1)
    df["diff_srv_rate"] = np.clip(RNG.normal(0.01, 0.02, n), 0, 1)
    df["dst_host_count"] = RNG.poisson(lam=255, size=n)
    df["dst_host_srv_count"] = RNG.poisson(lam=255, size=n)
    df["num_failed_logins"] = RNG.poisson(lam=0.01, size=n)
    df["logged_in"] = 0
    return df
def gen_probe(n):
    # Scanning behaviour: many different services/hosts, short, low bytes
    df = _base_frame(n, "Probe")
    df["duration"] = RNG.exponential(scale=0.2, size=n)
    df["src_bytes"] = RNG.lognormal(mean=2, sigma=0.5, size=n).astype(int)
    df["dst_bytes"] = RNG.lognormal(mean=1, sigma=0.5, size=n).astype(int)
    df["count"] = RNG.poisson(lam=50, size=n)
    df["srv_count"] = RNG.poisson(lam=5, size=n)
    df["serror_rate"] = np.clip(RNG.normal(0.3, 0.2, n), 0, 1)
    df["rerror_rate"] = np.clip(RNG.normal(0.4, 0.2, n), 0, 1)
    df["same_srv_rate"] = np.clip(RNG.normal(0.1, 0.1, n), 0, 1)
    df["diff_srv_rate"] = np.clip(RNG.normal(0.7, 0.15, n), 0, 1)
    df["dst_host_count"] = RNG.poisson(lam=200, size=n)
    df["dst_host_srv_count"] = RNG.poisson(lam=10, size=n)
    df["num_failed_logins"] = RNG.poisson(lam=0.05, size=n)
    df["logged_in"] = 0
    return df
def gen_r2l(n):
    # Remote-to-local: login attempts from outside, failed logins, low traffic
    df = _base_frame(n, "R2L")
    df["duration"] = RNG.exponential(scale=8, size=n)
    df["src_bytes"] = RNG.lognormal(mean=4, sigma=1.0, size=n).astype(int)
    df["dst_bytes"] = RNG.lognormal(mean=3, sigma=1.0, size=n).astype(int)
    df["count"] = RNG.poisson(lam=3, size=n)
    df["srv_count"] = RNG.poisson(lam=3, size=n)
    df["serror_rate"] = np.clip(RNG.normal(0.1, 0.1, n), 0, 1)
    df["rerror_rate"] = np.clip(RNG.normal(0.2, 0.15, n), 0, 1)
    df["same_srv_rate"] = np.clip(RNG.normal(0.5, 0.2, n), 0, 1)
    df["diff_srv_rate"] = np.clip(RNG.normal(0.3, 0.2, n), 0, 1)
    df["dst_host_count"] = RNG.poisson(lam=10, size=n)
    df["dst_host_srv_count"] = RNG.poisson(lam=10, size=n)
    df["num_failed_logins"] = RNG.poisson(lam=2.5, size=n)
    df["logged_in"] = RNG.choice([0, 1], n, p=[0.7, 0.3])
    return df
def gen_u2r(n):
    # User-to-root: rare, low count, but anomalous internal behaviour
    df = _base_frame(n, "U2R")
    df["duration"] = RNG.exponential(scale=15, size=n)
    df["src_bytes"] = RNG.lognormal(mean=5, sigma=1.5, size=n).astype(int)
    df["dst_bytes"] = RNG.lognormal(mean=5, sigma=1.5, size=n).astype(int)
    df["count"] = RNG.poisson(lam=2, size=n)
    df["srv_count"] = RNG.poisson(lam=2, size=n)
    df["serror_rate"] = np.clip(RNG.normal(0.05, 0.05, n), 0, 1)
    df["rerror_rate"] = np.clip(RNG.normal(0.05, 0.05, n), 0, 1)
    df["same_srv_rate"] = np.clip(RNG.normal(0.6, 0.2, n), 0, 1)
    df["diff_srv_rate"] = np.clip(RNG.normal(0.2, 0.15, n), 0, 1)
    df["dst_host_count"] = RNG.poisson(lam=5, size=n)
    df["dst_host_srv_count"] = RNG.poisson(lam=5, size=n)
    df["num_failed_logins"] = RNG.poisson(lam=0.3, size=n)
    df["logged_in"] = 1
    return df
def main():
    frames = [
        gen_normal(N_NORMAL),
        gen_dos(N_DOS),
        gen_probe(N_PROBE),
        gen_r2l(N_R2L),
        gen_u2r(N_U2R),
    ]
    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)  # shuffle
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_PATH, index=False)
    print(f"Wrote {len(df)} rows to {OUT_PATH}")
    print(df["label"].value_counts())
if __name__ == "__main__":
    main()
