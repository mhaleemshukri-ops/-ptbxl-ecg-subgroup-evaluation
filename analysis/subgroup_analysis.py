import numpy as np
import pandas as pd
import argparse
from sklearn.metrics import roc_auc_score

parser = argparse.ArgumentParser(
    description="Compare NORM classification AUROC between age groups."
)

parser.add_argument(
    "--input",
    required=True,
    help="Path to a CSV containing the evaluation predictions."
)

args = parser.parse_args()

df = pd.read_csv(args.input)

# Age groups used in the comparison
young = df[df["age"] < 40].copy()
older = df[(df["age"] >= 60) & (df["age"] < 80)].copy()


def calculate_auc(data):
    return roc_auc_score(
        data["norm_true"],
        data["norm_max_recomputed"]
    )


# Observed difference in AUROC
observed_difference = calculate_auc(older) - calculate_auc(young)

# Patient-clustered bootstrap
rng = np.random.default_rng(2026)
n_bootstrap = 2000


def bootstrap_sample(data):
    patient_groups = [
        group
        for _, group in data.groupby("patient_id", sort=False)
    ]

    selected = rng.integers(
        0,
        len(patient_groups),
        len(patient_groups)
    )

    return pd.concat(
        [patient_groups[i] for i in selected],
        ignore_index=True
    )


bootstrap_differences = []

for _ in range(n_bootstrap):
    young_sample = bootstrap_sample(young)
    older_sample = bootstrap_sample(older)

    if (
        young_sample["norm_true"].nunique() == 2
        and older_sample["norm_true"].nunique() == 2
    ):
        difference = (
            calculate_auc(older_sample)
            - calculate_auc(young_sample)
        )
        bootstrap_differences.append(difference)

ci_lower, ci_upper = np.percentile(
    bootstrap_differences,
    [2.5, 97.5]
)

print("Age-group AUROC comparison")
print("--------------------------")
print(f"Under 40 AUROC: {calculate_auc(young):.4f}")
print(f"Age 60–79 AUROC: {calculate_auc(older):.4f}")
print(f"AUROC difference: {observed_difference:+.4f}")
print(f"95% bootstrap CI: [{ci_lower:+.4f}, {ci_upper:+.4f}]")
print(f"Valid bootstrap samples: {len(bootstrap_differences)}")
