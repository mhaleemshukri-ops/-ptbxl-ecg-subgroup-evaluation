import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(
    description="Compare NORM classification AUROC by age and sex."
)

parser.add_argument(
    "--input",
    required=True,
    help="Path to a CSV containing the evaluation predictions."
)

args = parser.parse_args()

df = pd.read_csv(args.input)

rng = np.random.default_rng(2026)
n_bootstrap = 2000

age_groups = [
    ("Under 40", df["age"] < 40),
    ("40–59", (df["age"] >= 40) & (df["age"] < 60)),
    ("60–79", (df["age"] >= 60) & (df["age"] < 80)),
    ("80+", df["age"] >= 80),
]


def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))

    adjusted[order] = np.minimum(
        1.0,
        np.maximum.accumulate(
            (len(p) - np.arange(len(p))) * p[order]
        )
    )

    return adjusted


results = []

for age_label, mask in age_groups:
    subgroup = df.loc[mask].copy()

    patients = subgroup["patient_id"].unique()

    patient_rows = {
        patient_id: subgroup.index[
            subgroup["patient_id"] == patient_id
        ].to_numpy()
        for patient_id in patients
    }

    def auc_difference(sample):
        male = sample[sample["sex"] == 0]
        female = sample[sample["sex"] == 1]

        if (
            male["norm_true"].nunique() < 2
            or female["norm_true"].nunique() < 2
        ):
            return np.nan

        male_auc = roc_auc_score(
            male["norm_true"],
            male["norm_max_recomputed"]
        )

        female_auc = roc_auc_score(
            female["norm_true"],
            female["norm_max_recomputed"]
        )

        return male_auc - female_auc

    observed = auc_difference(subgroup)
    bootstrap_differences = []

    for _ in range(n_bootstrap):
        sampled_patients = rng.choice(
            patients,
            size=len(patients),
            replace=True
        )

        sampled_indices = np.concatenate(
            [patient_rows[patient_id] for patient_id in sampled_patients]
        )

        bootstrap_differences.append(
            auc_difference(subgroup.loc[sampled_indices])
        )

    bootstrap_differences = np.asarray(bootstrap_differences)
    bootstrap_differences = bootstrap_differences[
        np.isfinite(bootstrap_differences)
    ]

    ci_lower, ci_upper = np.percentile(
        bootstrap_differences,
        [2.5, 97.5]
    )

    # Exploratory bootstrap sign-based p-value.
    # This is not a formal null-hypothesis test.
    p_value = min(
        1.0,
        2 * min(
            np.mean(bootstrap_differences <= 0),
            np.mean(bootstrap_differences >= 0)
        )
    )

    results.append(
        (
            age_label,
            observed,
            ci_lower,
            ci_upper,
            p_value,
            len(bootstrap_differences)
        )
    )


adjusted_pvalues = holm_adjust([row[4] for row in results])

print("Exploratory age-by-sex AUROC comparisons")
print("----------------------------------------")
print("Difference = male AUROC minus female AUROC")
print()

for row, adjusted_p in zip(results, adjusted_pvalues):
    age_label, difference, ci_lower, ci_upper, raw_p, n_valid = row

    print(f"Age group: {age_label}")
    print(f"  AUROC difference: {difference:+.4f}")
    print(f"  95% bootstrap CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
    print(f"  Approximate bootstrap p-value: {raw_p:.4f}")
    print(f"  Holm-adjusted approximate p-value: {adjusted_p:.4f}")
    print(f"  Valid bootstrap samples: {n_valid}")
    print()
