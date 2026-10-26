# ECG Classification Across Age and Sex Groups

## About the project

In this project, we looked at how well a pretrained ECG classification model performs for different age and sex groups.

We used the PTB-XL dataset and evaluated an XResNet1D-101 model on test fold 10. We focused on the NORM label, which identifies ECGs labelled as normal.

Our research question was:

**How does classification performance of a pretrained XResNet1D-101 vary across predefined age and sex subgroups in the held-out PTB-XL test set?**

We did not train a new model. Our work was mainly about evaluating the predictions and comparing the results between groups.

## Data and method

We used 2,203 ECG recordings from 1,907 patients.

We measured performance using AUROC and divided the recordings into four age groups:

- Under 40
- 40–59
- 60–79
- 80+

We also compared male and female recordings within each age group.

To estimate uncertainty, we used 2,000 bootstrap samples, resampling by patient rather than by individual ECG.

## Results

The overall AUROC was **0.9511**, which shows that the model performed well overall.

However, performance was not the same across all age groups:

- Under 40: **0.8779**
- Age 60–79: **0.9480**

The difference was **0.0701**, with a 95% confidence interval of **[0.0194, 0.1266]**.

We also found a difference between male and female recordings in the 40–59 age group:

- Male AUROC: **0.9580**
- Female AUROC: **0.8710**

This was an exploratory comparison, so it should be interpreted carefully.

Our main finding is that a good overall model score does not necessarily mean that performance is equally good for every group.

## Code

The `analysis` folder contains two Python scripts:

- `subgroup_analysis.py` — compares AUROC between the under-40 and 60–79 age groups.
- `age_sex_comparison.py` — compares male and female AUROC within each age group.

The scripts use Python with NumPy, pandas, and scikit-learn.

To run them:

```bash
python analysis/subgroup_analysis.py --input /path/to/predictions.csv
```

```bash
python analysis/age_sex_comparison.py --input /path/to/predictions.csv
```

The input CSV needs these columns:

`patient_id`, `age`, `sex`, `norm_true`, `norm_max_recomputed`

Here, `sex` is coded as 0 for male and 1 for female.

The scripts use a fixed random seed of 2026.

## Notes

This study only evaluates one pretrained model and the NORM label on one test set. We cannot say from these results whether the model would perform the same way on other datasets or in clinical use.

We have not uploaded the PTB-XL dataset, model weights, or patient-level prediction files.

The scripts reproduce our statistical comparisons using an existing prediction CSV. They do not generate predictions directly from raw ECG signals.

## Original model

We used the pretrained model and code from the original ECG benchmarking project:

https://github.com/helme/ecg_ptbxl_benchmarking

The model was developed by the original authors. Our contribution was evaluating its performance across age and sex groups.
