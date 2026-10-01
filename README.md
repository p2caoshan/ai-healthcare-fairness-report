# Pulse oximetry fairness: aggregate results

A shareable write-up of the analyses in this project. It reports cohort sizes, bias tests, detection rates, and model metrics. It does not include patient records, identifiers, timestamps, or raw measurement tables.

**Data are not in this repository.** MIMIC-IV and the BOLD blood-gas oximetry file are credentialed PhysioNet datasets and are not redistributed here. The heart-disease appendix uses only summary statistics from a public Kaggle table.

Open the designed briefing at the site root (`index.html`) once GitHub Pages is enabled. This page is the same analysis in full.

## The question

Pulse oximeters estimate arterial oxygen saturation (SaO2) from a light signal on the finger (SpO2). If that estimate sits too high, a low arterial saturation can be missed. That miss is hidden hypoxemia. The project asks three questions, on two datasets:

1. Is SpO2 less accurate for darker skin groups, and do those groups miss more hypoxemia?
2. Can a model that combines SpO2 with labs and vital signs recover SaO2 well enough to catch more of those events?
3. Do Fairlearn group-loss constraints shrink the remaining gap between skin groups?

Skin group is a coarse proxy built from recorded race or ethnicity, used in place of a measured pigment. Causal methods in healthcare are a separate tab on the site. They are not estimated from these cohorts.

## How the numbers were produced

| | MIMIC-IV derived cohort | BOLD external cohort |
| --- | --- | --- |
| Source | MIMIC-IV v3.1 (lab events, admissions, patients, ICU chart events) | PhysioNet blood-gas and oximetry (BOLD) v1.0 |
| Pairing | SaO2 item `50817` with SpO2 item `220277`, both kept in 70–100, same patient and calendar date, within ±5 minutes, and inside the admission window | Paired SaO2–SpO2 already in the released table, same 70–100 window |
| Pairs after the admission-window join | 32,751 | — |
| Table used for modeling | 30,454 rows, 44 columns | 49,093 rows, 142 columns |
| Columns with more than 20% missing | 15 | 64 |
| Extra features | Nearest prior lab within 24 hours (9 labs) and nearest prior ICU measurement within 1 hour (8 vitals) | Demographics, admission traits, blood gas, vitals, labs |
| Split | 80/20, `random_state=42`, stratified on skin group | Same |

The modeling notebook reads the enriched MIMIC table. The Spark job writes a parquet; the executed model cells load a CSV export of that table. Both describe the same 30,454 × 44 modeling frame. This page does not try to explain the 32,751 to 30,454 drop, because the notebooks do not print that reconciliation.

### Skin-group maps

MIMIC, from the free-text `race` field:

| Group | Recorded race strings containing |
| --- | --- |
| Light | White, European, Portuguese |
| Medium | Hispanic, Latino, Asian, Pacific, Indian, South American |
| Dark | Black, African |
| Other/Unknown | Everything else |

BOLD, from `race_ethnicity`:

| Group | Source labels |
| --- | --- |
| Light | White |
| Dark | Black |
| Medium | Hispanic or Latino, Asian, American Indian / Alaska Native, Native Hawaiian / Pacific Islander |
| Other/Unknown | Unknown, More Than One Race, and any unmapped label |

Held-out test sizes (the 20% split). These are the denominators behind the group metrics.

| Skin group | MIMIC test rows | BOLD test rows |
| --- | --- | --- |
| Light | 3,939 (64.7%) | 7,476 (76.1%) |
| Other/Unknown | 1,272 (20.9%) | 713 (7.3%) |
| Dark | 463 (7.6%) | 957 (9.7%) |
| Medium | 417 (6.8%) | 673 (6.9%) |
| Test total | 6,091 | 9,819 |

Light-skin rows dominate both test sets. Medium and, on BOLD, the smaller hypoxemia counts are noisy.

### Definitions used in the notebooks

| Term | Definition in the code |
| --- | --- |
| Bias | SaO2 − SpO2, in percentage points. Negative means the pulse oximeter reads higher than the arterial value. |
| Hypoxemia | SaO2 < 88 |
| Device detection | SpO2 < 88 |
| Hidden miss (false negative) | SaO2 < 88 and SpO2 ≥ 88 |
| “Safe” reading for disparate impact | SpO2 ≥ 92, compared with SaO2 ≥ 92 |
| Model alert | Predicted SaO2 < 88 |

The false-negative rate here is “among arterial saturations under 88, the share whose paired SpO2 stayed at 88 or above.” A widely cited research definition is stricter (SaO2 < 88 and SpO2 ≥ 92). These notebooks do not use that stricter rule, so the rates below should not be quoted as if they did.

## 1. Measurement bias

SpO2 is piled up nearer 99–100 than SaO2 on both datasets. That shape is consistent with over-reading, and the group tests measure how uneven it is.

The Welch tests compare means of SaO2 − SpO2. The printed mean difference is Dark minus the comparison group. A negative difference means the Dark group is shifted further toward over-reading. The printed t-statistic is the comparison group versus Dark, so a positive t and a negative Dark-minus-other difference are the same result.

### MIMIC-IV

| Contrast | t | p | Mean difference (Dark − other) | 95% CI |
| --- | --- | --- | --- | --- |
| Dark vs Light | 5.488 | < 0.001 | −1.058 | (−1.436, −0.680) |
| Dark vs Medium | 1.699 | 0.089 | −0.450 | (−0.968, 0.069) |

On MIMIC, the Dark–Light gap is about one saturation point and is unlikely to be noise. The Dark–Medium gap is smaller and not significant at 5%.

Device error against arterial saturation, all rows: **MAE 5.943**, **RMSE 9.978**.

Disparate impact at the “safe reading” cutoff of 92:

| Quantity | Value |
| --- | --- |
| Share with SpO2 ≥ 92, Dark | 0.947 |
| Share with SpO2 ≥ 92, Light | 0.941 |
| Disparate impact ratio (Dark / Light) | 1.007 |
| Four-fifths rule | Inside 0.80–1.25, so demographic parity on this cutoff is met |
| Share with SaO2 ≥ 92, Dark / Light | 0.946 |
| Device ratio minus arterial ratio | +0.061 |

The device calls the Dark group “safe” about as often as the Light group. Arterial saturation does not. The device ratio sits 0.061 above the arterial ratio, which is the hidden-hypoxemia pattern: more Dark-group readings look reassuring than the blood gas supports.

### BOLD

| Contrast | t | p | Mean difference (Dark − other) | 95% CI |
| --- | --- | --- | --- | --- |
| Dark vs Light | 9.408 | < 0.001 | −0.630 | (−0.761, −0.499) |
| Dark vs Medium | 3.192 | 0.001 | −0.282 | (−0.456, −0.109) |

Both contrasts are significant on BOLD. The Dark–Light gap is smaller in points than on MIMIC (0.63 vs 1.06) and estimated more tightly.

Device error, all rows: **MAE 2.437**, **RMSE 4.049**. The BOLD oximeter is much closer to the arterial value than the MIMIC pairing.

| Quantity | Value |
| --- | --- |
| Share with SpO2 ≥ 92, Dark | 0.939 |
| Share with SpO2 ≥ 92, Light | 0.925 |
| Disparate impact ratio | 1.015 |
| Four-fifths rule | Met |
| Arterial safe ratio (SaO2 ≥ 92, Dark / Light) | 0.995 |
| Device ratio minus arterial ratio | +0.020 |

Same direction as MIMIC, smaller distortion.

### What is common

On both cohorts the pulse oximeter over-reads more for the Dark group than for the Light group, demographic parity at SpO2 ≥ 92 still “passes,” and that pass hides a gap versus the arterial safe rate. MIMIC shows a larger point gap and much larger absolute error. BOLD shows a smaller gap that is also present versus the Medium group.

## 2. Predicting SaO2, and catching hypoxemia

Models are ordinary linear regression and XGBoost regression. Missing predictors are filled with training-set medians. Categorical fields are one-hot encoded. Hypoxemia recall is computed on the held-out 20%.

Test-set hypoxemia (SaO2 < 88) is common in the MIMIC split and uncommon in BOLD:

| | Hypoxemia events in the test set | Share of test rows |
| --- | --- | --- |
| MIMIC | 1,283 / 6,091 | 21.1% |
| BOLD | 454 / 9,819 | 4.6% |

That mix matters. MIMIC is a sicker paired-measurement set, which lines up with its larger device error.

### MIMIC regression

Features: SpO2, gender, anchor age, hemoglobin, carboxyhemoglobin, creatinine, hematocrit, lactate, methemoglobin, pCO2, pH, pO2, and eight ICU vitals (heart rate, respiratory rate, non-invasive systolic / diastolic / mean pressure, temperature, weight, height). Rows used: 30,454.

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Linear regression | 6.178 | 8.201 | 0.031 |
| XGBoost (depth 3, 250 trees, learning rate 0.1) | 5.051 | 7.262 | 0.240 |
| Raw SpO2, all rows (not the test split) | 5.943 | 9.978 | — |

Linear regression does not beat the device on MAE. XGBoost does, modestly, and explains about a quarter of the variance in SaO2.

Group-wise error on the test set:

| Group | n | LR MAE | LR RMSE | LR R² | XGB MAE | XGB RMSE | XGB R² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dark | 463 | 6.368 | 8.548 | 0.044 | 5.064 | 7.541 | 0.256 |
| Light | 3,939 | 6.193 | 8.163 | 0.032 | 5.026 | 7.172 | 0.253 |
| Medium | 417 | 6.324 | 8.589 | 0.003 | 5.399 | 7.780 | 0.182 |
| Other/Unknown | 1,272 | 6.016 | 8.060 | 0.028 | 5.008 | 7.260 | 0.212 |

The Medium group is the weakest XGBoost fit (R² 0.18, MAE 5.40).

Hypoxemia recall (threshold 88). This is the clinical result, and it is much larger than the MAE change.

| Group | Events | SpO2 recall | LR recall | XGB recall | XGB false-negative rate |
| --- | --- | --- | --- | --- | --- |
| Dark | 108 | 0.056 | 0.083 | 0.444 | 0.556 |
| Light | 820 | 0.049 | 0.057 | 0.481 | 0.520 |
| Medium | 94 | 0.032 | 0.032 | 0.351 | 0.649 |
| Other/Unknown | 261 | 0.077 | 0.065 | 0.437 | 0.563 |

The device finds about 1 in 20 arterial hypoxemias in the Dark group (6 true positives, 102 misses). XGBoost finds about 4 in 10 (48 true positives, 60 misses), roughly eight times the device recall. Light rises from 0.049 to 0.481. Medium remains the lowest XGBoost recall (0.351). Linear regression barely moves recall in any group.

XGBoost split-count importance (weight). Gender does not appear.

| Feature | Weight | Feature | Weight |
| --- | --- | --- | --- |
| lab pO2 | 291 | Respiratory rate | 74 |
| Hemoglobin | 140 | Temperature | 73 |
| Anchor age | 120 | Systolic pressure | 72 |
| Lactate | 108 | Carboxyhemoglobin | 70 |
| SpO2 | 99 | Creatinine | 54 |
| Heart rate | 98 | Weight | 46 |
| pH | 97 | Mean pressure | 42 |
| Hematocrit | 92 | Height | 35 |
| pCO2 | 89 | Diastolic pressure | 34 |
| | | Methemoglobin | 15 |

### BOLD regression

Features: SpO2, delta SpO2, admission age, sex, BMI, heart rate, respiratory rate, mean / systolic / diastolic pressure, temperature, hemoglobin, hematocrit, creatinine, lactate, pH, pCO2, pO2, carboxyhemoglobin, methemoglobin. Rows used: 49,093.

| Model | MAE | RMSE | R² |
| --- | --- | --- | --- |
| Linear regression | 1.997 | 3.129 | 0.424 |
| XGBoost (depth 3, 100 trees, learning rate 0.1) | 0.881 | 1.339 | 0.895 |
| Raw SpO2, all rows | 2.437 | 4.049 | — |

Both models beat the device. XGBoost fits this table tightly (R² 0.89).

| Group | n | LR MAE | LR RMSE | LR R² | XGB MAE | XGB RMSE | XGB R² |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Dark | 957 | 2.229 | 3.505 | 0.430 | 0.883 | 1.336 | 0.917 |
| Light | 7,476 | 1.986 | 3.082 | 0.424 | 0.900 | 1.335 | 0.892 |
| Medium | 673 | 1.834 | 2.799 | 0.497 | 0.828 | 1.356 | 0.882 |
| Other/Unknown | 713 | 1.958 | 3.360 | 0.342 | 0.737 | 1.361 | 0.892 |

Hypoxemia recall:

| Group | Events | SpO2 recall | LR recall | XGB recall | XGB false-negative rate |
| --- | --- | --- | --- | --- | --- |
| Dark | 61 | 0.230 | 0.131 | 0.771 | 0.230 |
| Light | 337 | 0.237 | 0.116 | 0.831 | 0.169 |
| Medium | 28 | 0.393 | 0.214 | 0.786 | 0.214 |
| Other/Unknown | 28 | 0.214 | 0.107 | 0.821 | 0.179 |

XGBoost raises Dark-group recall from 0.23 to 0.77 (47 of 61 events). Linear regression is worse than the device in every group. Medium and Other each have only 28 test events, so those two recalls are unstable.

| Feature | Weight | Feature | Weight |
| --- | --- | --- | --- |
| pO2 | 237 | Temperature | 17 |
| pH | 148 | Admission age | 13 |
| SpO2 | 45 | BMI | 12 |
| pCO2 | 44 | Hematocrit | 12 |
| Methemoglobin | 36 | Delta SpO2 | 10 |
| Carboxyhemoglobin | 34 | Hemoglobin | 7 |
| Lactate | 23 | Sex (female) | 5 |
| Mean pressure | 23 | Diastolic pressure | 5 |
| Systolic pressure | 21 | Heart rate | 3 |
| | | Creatinine | 3 |
| | | Respiratory rate | 2 |

### A limit on what the accuracy means

In both fits, arterial pO2 is the most-used split, and pH or hemoglobin is next. Those blood-gas values are taken with SaO2, and pO2 is physiologically tied to saturation. The gain over raw SpO2 is therefore not a pure pulse-oximeter correction from bedside signals alone. The notebooks contain a commented feature list without labs and vitals; that list was not the run that produced the metrics above.

Even the better MIMIC detector still misses more than half of Dark-group hypoxemias (false-negative rate 0.56). BOLD’s stronger detector still misses about 1 in 4 in the Dark group, and its event counts are small.

## 3. Fairlearn constraints

A second XGBoost was trained and then wrapped in Fairlearn `GridSearch` (grid size 11) with `BoundedGroupLoss`. Fairness metrics are computed after thresholding predictions at 88, not on the regression loss the constraint actually optimizes. That mismatch is why a model can look “fairer” or “less fair” on recall while MAE barely moves.

Equal opportunity difference is the gap between the highest and lowest group recall. Average odds difference averages that gap with the false-positive-rate gap. Theil index is computed on a benefit score of the binary alert. Lower is closer.

### MIMIC

Second model: 250 trees, depth 3, learning rate 0.1, subsample 0.8, column subsample 0.8. Both constraints use an upper bound of 2.0.

| Model | Equal opportunity diff. | Average odds diff. | Theil | MAE | RMSE |
| --- | --- | --- | --- | --- | --- |
| XGBoost, no constraint | 0.0284 | 0.0273 | 0.1408 | 4.991 | 7.196 |
| GridSearch, square loss | 0.0388 | 0.0257 | 0.1508 | 5.151 | 7.313 |
| GridSearch, absolute loss | 0.0416 | 0.0269 | 0.1427 | 5.039 | 7.252 |

Recall (true positive rate) and false positive rate by group:

| Model | Group | Recall | FPR | FNR | Alert rate |
| --- | --- | --- | --- | --- | --- |
| Unconstrained | Dark | 0.475 | 0.070 | 0.526 | 0.166 |
| Unconstrained | Light | 0.472 | 0.079 | 0.528 | 0.160 |
| Unconstrained | Medium | 0.448 | 0.096 | 0.552 | 0.175 |
| Unconstrained | Other/Unknown | 0.477 | 0.091 | 0.523 | 0.169 |
| Absolute loss | Dark | 0.445 | 0.084 | 0.555 | 0.169 |
| Absolute loss | Light | 0.464 | 0.077 | 0.536 | 0.157 |
| Absolute loss | Medium | 0.422 | 0.072 | 0.578 | 0.150 |
| Absolute loss | Other/Unknown | 0.458 | 0.077 | 0.542 | 0.154 |
| Square loss | Dark | 0.453 | 0.088 | 0.547 | 0.174 |
| Square loss | Light | 0.426 | 0.077 | 0.574 | 0.149 |
| Square loss | Medium | 0.414 | 0.082 | 0.586 | 0.156 |
| Square loss | Other/Unknown | 0.430 | 0.089 | 0.570 | 0.158 |

The unconstrained model already has a small recall gap (about 3 points) and the best MAE. Both constraints widen equal opportunity difference and raise error. Square loss trims average odds difference only slightly (0.0273 to 0.0257) while making recall less even and MAE worse. Closing a regression loss gap did not produce a fairer hypoxemia detector.

### BOLD

The Fairlearn cell does not reuse the high-R² model from section 2. It refits with 250 trees, subsample 0.8, and column subsample 0.8. Square loss is bounded at 6.0. Absolute loss is bounded at 2.0. Read this block as its own specification.

| Model | Equal opportunity diff. | Average odds diff. | Theil | MAE | RMSE |
| --- | --- | --- | --- | --- | --- |
| XGBoost, no constraint | 0.0744 | 0.0384 | 0.0459 | 2.126 | 3.396 |
| GridSearch, square loss | 0.1148 | 0.0590 | 0.0451 | 2.222 | 3.429 |
| GridSearch, absolute loss | 0.1786 | 0.0900 | 0.0454 | 2.205 | 3.472 |

| Model | Group | Recall | FPR | FNR | Alert rate |
| --- | --- | --- | --- | --- | --- |
| Unconstrained | Dark | 0.033 | 0.003 | 0.967 | 0.005 |
| Unconstrained | Light | 0.068 | 0.005 | 0.932 | 0.008 |
| Unconstrained | Medium | 0.107 | 0.005 | 0.893 | 0.009 |
| Unconstrained | Other/Unknown | 0.071 | 0.006 | 0.929 | 0.008 |
| Absolute loss | Dark | 0.066 | 0.007 | 0.934 | 0.010 |
| Absolute loss | Light | 0.101 | 0.008 | 0.899 | 0.012 |
| Absolute loss | Medium | 0.214 | 0.006 | 0.786 | 0.015 |
| Absolute loss | Other/Unknown | 0.036 | 0.007 | 0.964 | 0.008 |
| Square loss | Dark | 0.115 | 0.008 | 0.885 | 0.015 |
| Square loss | Light | 0.086 | 0.005 | 0.914 | 0.008 |
| Square loss | Medium | 0.107 | 0.005 | 0.893 | 0.009 |
| Square loss | Other/Unknown | 0.000 | 0.006 | 1.000 | 0.006 |

Two facts sit side by side. The section 2 XGBoost (100 trees, no subsampling) detects about 77–83% of hypoxemias. The section 3 unconstrained refit detects about 3–11% and alerts on under 1% of rows. Its MAE (2.13) is also far from the section 2 MAE (0.88). Constraints then make the recall gap larger, not smaller. Under square loss the Other/Unknown recall is 0. Absolute loss pushes Medium recall to 0.214 while Other/Unknown falls to 0.036, which is a 0.18 equal-opportunity gap.

The notebook’s closing comment says mitigation improved some fairness metrics at an accuracy cost. The printed tables support the accuracy cost. They do not support an improvement in equal opportunity or average odds. On both datasets the unconstrained model in this cell has the smallest equal-opportunity difference and the lowest MAE.

## What the two cohorts agree on

- SpO2 over-reads more in the Dark group than in the Light group. The shift is about 1.1 points on the MIMIC pairing and 0.6 points on BOLD.
- A disparate-impact check at SpO2 ≥ 92 passes the 0.80–1.25 band on both, while the arterial safe-rate ratio is lower. The device looks more even than the blood gas. The excess of the device ratio over the arterial ratio is +0.061 on MIMIC and +0.020 on BOLD.
- Linear regression is a weak hypoxemia detector. On BOLD it is worse than SpO2 itself.
- An XGBoost SaO2 model can raise detection a lot: Dark-group recall from 6% to 44% on MIMIC, and from 23% to 77% on BOLD, under the section 2 specification. The same model widens equal opportunity relative to the device: MIMIC’s four-group recall spread goes from 0.045 to 0.130, with Medium at the floor, and BOLD’s Light–Dark recall gap goes from 0.007 to 0.060.
- Arterial blood gas, especially pO2, dominates the fit. That limits any claim that the same gain would appear from SpO2 and routine vitals alone.
- Fairlearn group-loss search, as configured, did not reduce the recall gap and did cost a little accuracy. It is not the same training as the section 2 detector, whose four-group recall spread is 0.130 on MIMIC.

## What they do not agree on

- Absolute device error is about twice as large on MIMIC (MAE 5.9 vs 2.4), and hypoxemia is about four times as common in the MIMIC test split (21% vs 5%).
- Dark versus Medium is significant on BOLD and not on MIMIC.
- The Medium group is the hardest detection problem on MIMIC (XGB recall 0.35) and is not singled out that way on BOLD.
- BOLD’s strong detector and BOLD’s Fairlearn refit are different models. Quoting only the Fairlearn cell would hide the 0.89 R² result. Quoting only section 2 would hide how fragile that operating point is to a routine hyperparameter change.

---

## Appendix A. Heart-disease models and metrics

Separate course exercise on the public Heart Failure Prediction table (918 rows, 12 columns, no missing values). No individual rows are listed here. Positive class rate is 55.3% (508 of 918). Sex is imbalanced (725 male, 193 female). Chest-pain type ASY is the largest category (496). Resting ECG is Normal in 552. Exercise angina is absent in 547 (present in 371). ST slope is Flat in 460.

Numeric summaries:

| | Mean | SD | Min | Median | Max |
| --- | --- | --- | --- | --- | --- |
| Age | 53.51 | 9.43 | 28 | 54 | 77 |
| Resting BP | 132.40 | 18.51 | 0 | 130 | 200 |
| Cholesterol | 198.80 | 109.38 | 0 | 223 | 603 |
| Fasting blood sugar (binary) | 0.233 | 0.423 | 0 | 0 | 1 |
| Max heart rate | 136.81 | 25.46 | 60 | 138 | 202 |
| Oldpeak | 0.887 | 1.067 | −2.6 | 0.6 | 6.2 |

Resting BP and cholesterol both have a minimum of 0, which is not a plausible measurement. Those zeros behave like missing values coded as zero, and they sit inside the cholesterol coefficient below.

Train/test split 80/20, `random_state=42`, predictors standardized. Categorical fields were turned into integer codes before the split. For multi-level fields the coefficient is on that code, not on a clinical contrast. Sex is alphabetical, so male is the higher code.

Hyperparameters chosen by 5-fold search on the training set, then refit:

| Model | Chosen settings | ROC AUC on hard labels |
| --- | --- | --- |
| Logistic regression, age only | C = 0.01, L2 | 0.583 |
| Logistic regression, all features | C = 0.1, L2 | 0.846 |
| Random forest | 100 trees, unlimited depth, min leaf 2, min split 10 | 0.897 |
| Gradient boosting | 50 trees, learning rate 0.1, depth 5, min leaf 2, min split 2 | 0.876 |

The AUC figures in that table are computed from predicted classes, not from probabilities, so they understate ranking performance. A 1,000-draw resample of the test set, using probabilities from the full logistic regression and not refitting the model, gives a mean AUC of 0.900 with a 95% interval of 0.852 to 0.943.

Standardized logistic coefficients, all features:

| Feature | Coefficient | Feature | Coefficient |
| --- | --- | --- | --- |
| Exercise angina | +0.582 | Resting ECG | −0.137 |
| Oldpeak | +0.455 | Max heart rate | −0.226 |
| Sex (male = higher code) | +0.439 | Cholesterol | −0.425 |
| Fasting blood sugar | +0.359 | Chest-pain type (coded) | −0.501 |
| Age | +0.143 | ST slope (coded) | −0.885 |
| Resting BP | +0.058 | | |

Exercise angina has the largest positive coefficient. ST slope has the largest negative one, which matches an upsloping code sitting at the high end of the alphabetical encoding and carrying lower risk. The negative cholesterol coefficient should be read next to the zero values in that column, not as a clinical claim that higher cholesterol protects.

Five metrics on the same hard predictions. They rank the models the same way. Random forest leads on every column. Brier score here is the error rate of the 0/1 prediction, not the probability Brier score.

| Model | Precision | Recall | F1 | AUC (labels) | Brier (labels) |
| --- | --- | --- | --- | --- | --- |
| Logistic, age only | 0.642 | 0.738 | 0.687 | 0.583 | 0.391 |
| Logistic, all features | 0.898 | 0.822 | 0.859 | 0.846 | 0.158 |
| Random forest | 0.923 | 0.897 | 0.910 | 0.897 | 0.103 |
| Gradient boosting | 0.912 | 0.869 | 0.890 | 0.876 | 0.125 |

Calibration curves (quantile bins) put the two logistic models near the diagonal. Random forest and gradient boosting sit over-confident: predicted probabilities run higher than observed frequencies. That is expected when the training loss is a classification loss rather than a probability loss, and it is why the forest can win the table above and still be the wrong model when a risk is shown to a patient.

Net benefit and expected cost were plotted against treat-all and treat-none, not tabulated. The three full-feature models sit above both reference strategies across a wide threshold band, roughly 0.1 to 0.9, and their expected-cost curves fall inside the treat-all / treat-none triangle. On that decision view the tree models outrank logistic regression. Age alone does not.

## Appendix B. Salk vaccine field trial

Design questions only. No patient data.

Giving the 1954 polio vaccine to a large group of children and then counting cases the next year would not identify the vaccine’s effect. There is no control series, and nothing blocks the other reasons polio rates change from year to year. A causal claim would need consistency (SUTVA), positivity, and ignorability. An uncontrolled rollout supplies none of them. It is also the wrong first step in children for a product that had only laboratory evidence.

Assigning vaccine by parental permission is not randomization. Parents who consent can differ in exposure, schooling, and trust in medicine. Those differences are exactly the kind that break ignorability, because they can change polio risk and the chance of being vaccinated.

The workable alternative is to ask permission to participate, and only then randomize consenting children to vaccine or control.

## Limits

- Race and ethnicity are proxies for skin pigment. They misclassify people, and the two datasets do not use the same map.
- One train-test split. No confidence intervals on the fairness metrics. Medium-group and BOLD hypoxemia counts are small.
- Group calibration was plotted in the notebooks and not printed as a slope, so calibration fairness is unscored here. The section 2 equal-opportunity spreads are arithmetic on rounded recalls.
- Fairness constraints optimize regression loss. The reported gaps are for a thresholded alert. Those are different tasks.
- On BOLD, section 2 and section 3 are different XGBoost settings. They must not be read as “the good model, then the mitigated version of that same model.”
- Blood-gas features, above all pO2, drive the fit.
- High missingness (15 columns on MIMIC, 64 on BOLD) is handled by median fill, which invents typical values for the patients who were not measured.
- Nothing here is a device evaluation, a treatment policy, or a recommendation to change monitoring practice.

## What is not in this repository

- MIMIC-IV tables, BOLD rows, and the enriched extracts
- Subject identifiers, admission times, and the sample rows printed during preprocessing
- Model binaries

Credentialed copies stay with the PhysioNet user who downloaded them: [MIMIC-IV v3.1](https://physionet.org/content/mimiciv/3.1/) and [BOLD v1.0](https://physionet.org/content/blood-gas-oximetry/1.0/). The heart-disease file is the public [Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction).

## Environment behind the printed results

MIMIC and BOLD notebooks were run with Python 3.10, pandas 2.3, NumPy 2.2, scikit-learn 1.7, XGBoost 3.2, and a Fairlearn build whose classification metrics were partly computed by hand. The Spark enrichment used PySpark 4.0 in local mode with 8 GB driver and executor memory. The heart-disease notebook recorded pandas 2.2.2, NumPy 2.0.2, seaborn 0.13.2, scikit-learn 1.6.1, and matplotlib 3.10.0.
