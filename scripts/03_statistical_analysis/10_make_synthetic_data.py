"""
Generates small synthetic cosine_similarity.csv / cka_summary.csv files,
matching the exact schema produced by extract_embeddings_similarity.py's
run_similarity(), for smoke-testing phase_b_statistical_analysis.py without
needing real embeddings. This mirrors the project's established practice of
validating pipeline scripts against synthetic data before production use
(see preflight_check.py).
"""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)

MODELS = ["uni", "conch", "quilt_llava", "gigapath_tile"]
PERTURBATIONS = list(range(1, 19))  # 18 perturbations
SEVERITIES = ["mild", "moderate", "severe"]
N_ACC, N_GDC = 12, 10  # small synthetic tile counts per cohort

rows = []
for model in MODELS:
    # give each model a slightly different baseline + demographic gap, so
    # H1/H2/H3 all have something non-trivial to detect
    model_offset = {"uni": 0.0, "conch": 0.02, "quilt_llava": -0.05, "gigapath_tile": 0.01}[model]
    demographic_gap = {"uni": -0.01, "conch": -0.03, "quilt_llava": -0.08, "gigapath_tile": -0.015}[model]
    for pid in PERTURBATIONS:
        for severity in SEVERITIES:
            sev_penalty = {"mild": 0.0, "moderate": -0.03, "severe": -0.07}[severity]
            for i in range(N_ACC):
                val = 0.9 + model_offset + sev_penalty + demographic_gap + rng.normal(0, 0.03)
                rows.append(["", model, f"ACC_sample_{i}", "ACC", pid, severity, round(float(np.clip(val, 0, 1)), 6)])
            for i in range(N_GDC):
                val = 0.9 + model_offset + sev_penalty + rng.normal(0, 0.03)
                rows.append(["", model, f"GDC_sample_{i}", "GDC", pid, severity, round(float(np.clip(val, 0, 1)), 6)])

cosine_df = pd.DataFrame(rows, columns=["_drop", "model", "sample_id", "cohort", "perturbation_id", "severity", "cosine_similarity"])
cosine_df = cosine_df.drop(columns=["_drop"])
cosine_df.to_csv("/home/claude/synthetic_cosine_similarity.csv", index=False)

cka_rows = []
for model in MODELS:
    for pid in PERTURBATIONS:
        for severity in SEVERITIES:
            for cohort, n in [("POOLED", 22), ("ACC", 12), ("GDC", 10)]:
                cka_rows.append([model, pid, severity, cohort, n, round(float(rng.uniform(0.6, 0.95)), 6)])
cka_df = pd.DataFrame(cka_rows, columns=["model", "perturbation_id", "severity", "cohort", "n_tiles", "linear_cka"])
cka_df.to_csv("/home/claude/synthetic_cka_summary.csv", index=False)

print(f"Wrote {len(cosine_df)} cosine rows and {len(cka_df)} CKA rows.")
