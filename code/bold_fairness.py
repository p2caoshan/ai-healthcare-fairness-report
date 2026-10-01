import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ttest_ind
import numpy as np
from scipy import stats
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

pd.set_option('display.max_columns', None)
DATA_PATH = "physionet.org/files/blood-gas-oximetry/1.0/bold_dataset.csv"
df = pd.read_csv(DATA_PATH)

df.shape

data_len = len(df)
missing_pct = df.isnull().sum()/data_len
missing_data = pd.DataFrame({'missing_count': df.isnull().sum(), 'missing_pct': missing_pct})

len(missing_pct[missing_pct > 0.2])

plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
sns.histplot(df['SaO2'], bins=30, kde=True, stat='density')
sns.histplot(df['SpO2'], bins=30, kde=True, stat='density')
plt.grid(False)
plt.legend(['SaO2', 'SpO2'])
plt.title('Distribution of O2 Levels (Full Range)')
plt.subplot(1, 2, 2)
sns.histplot(df['SaO2'], bins=30, kde=True, stat='density')
sns.histplot(df['SpO2'], bins=30, kde=True, stat='density')
plt.grid(False)
plt.xlim(85, 100)
plt.legend(['SaO2', 'SpO2'])
plt.title('Distribution of O2 Levels (Narrow Range)')

def apply_skin_tone_mapping(df, column_name='race'):

    race_map = {
        'White': 'Light',
        'Black': 'Dark',
        'Hispanic OR Latino': 'Medium',
        'Asian': 'Medium',
        'American Indian / Alaska Native': 'Medium',
        'Native Hawaiian / Pacific Islander': 'Medium',
        'Unknown': 'Other/Unknown',
        'More Than One Race': 'Other/Unknown'
    }

    df['skin_group'] = df[column_name].map(race_map).fillna('Other/Unknown')
    return df

df = apply_skin_tone_mapping(df, 'race_ethnicity')
df['skin_group'].value_counts()
df['SapO2_SpO2_diff'] = df['SaO2'] - df['SpO2']
df['is_hypoxic'] = (df['SaO2'] < 88).astype(int)
df['detected_hypoxia'] = (df['SpO2'] < 88).astype(int)
fig = plt.figure(figsize=(3, 3))
df['skin_group'].value_counts().plot(kind='pie', autopct='%1.1f%%', colors=['#ffffd2','#63b3ff','#99ff99','#ffcc99']    )

plt.figure(figsize=(10, 5))

sns.violinplot(
    data=df,
    x='SapO2_SpO2_diff',
    y='skin_group',
    hue='skin_group',
    palette='PuBu',
    inner='box',
    bw_adjust=.5,
    cut=0
)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Bias')

plt.title('Oxygen Saturation Bias by Skin Group\n(Positive = Pulse Ox Underestimates | Negative = Pulse Ox Overestimates)', fontsize=14)
plt.xlabel('Difference: SaO2 (Arterial) - SpO2 (Pulse Oximeter)', fontsize=12)
plt.ylabel('Skin Tone Group', fontsize=12)
plt.legend()
plt.tight_layout()
plt.show()

light_group = df[df['skin_group'] == 'Light']['SapO2_SpO2_diff']
medium_group = df[df['skin_group'] == 'Medium']['SapO2_SpO2_diff']
dark_group = df[df['skin_group'] == 'Dark']['SapO2_SpO2_diff']

print("Dark vs Light:")
t_stat, p_value = ttest_ind(light_group, dark_group, equal_var=False)
print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
n1, n2 = len(dark_group), len(light_group)
m1, m2 = np.mean(dark_group), np.mean(light_group)
v1, v2 = np.var(dark_group, ddof=1), np.var(light_group, ddof=1)
diff = m1 - m2
se = np.sqrt(v1/n1 + v2/n2)
df_welch = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
t_crit = stats.t.ppf(0.975, df_welch)
ci_lower = diff - t_crit * se
ci_upper = diff + t_crit * se
print(f"Mean Difference: {diff:.3f}")
print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")

print("\n")
print("Dark vs Medium:")
t_stat, p_value = ttest_ind(medium_group, dark_group, equal_var=False)
print(f"T-statistic: {t_stat:.3f}, P-value: {p_value:.3f}")
n1, n2 =len(dark_group), len(medium_group)
m1, m2 = np.mean(dark_group), np.mean(medium_group)
v1, v2 = np.var(dark_group, ddof=1), np.var(medium_group, ddof=1)
diff = m1 - m2
se = np.sqrt(v1/n1 + v2/n2)
df_welch = (v1/n1 + v2/n2)**2 / ((v1/n1)**2/(n1-1) + (v2/n2)**2/(n2-1))
t_crit = stats.t.ppf(0.975, df_welch)
ci_lower = diff - t_crit * se
ci_upper = diff + t_crit * se
print(f"Mean Difference: {diff:.3f}")
print(f"95% CI: ({ci_lower:.3f}, {ci_upper:.3f})")

fig = plt.figure(figsize=(10, 6))
for group in df['skin_group'].unique():
    subset = df[df['skin_group'] == group]

    fn = len(subset[(subset['is_hypoxic'] == 1) & (subset['detected_hypoxia'] == 0)])
    tp = len(subset[(subset['is_hypoxic'] == 1) & (subset['detected_hypoxia'] == 1)])
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    plt.bar(group, fnr, label=f'{group} (FNR={fnr:.2f})',color=sns.color_palette('PuBu')[['Light', 'Medium', 'Dark', 'Other/Unknown'].index(group)])
plt.title('False Negative Rate by Skin Group')
plt.xlabel('Skin Group')
plt.ylabel('False Negative Rate')
plt.legend(loc='lower left')

plt.figure(figsize=(10, 6))
sns.regplot(data=df[df['skin_group']=='Light'], x='SpO2', y='SaO2', label='Light', scatter_kws={'alpha':0.1})
sns.regplot(data=df[df['skin_group']=='Medium'], x='SpO2', y='SaO2', label='Medium', scatter_kws={'alpha':0.1})
sns.regplot(data=df[df['skin_group']=='Dark'], x='SpO2', y='SaO2', label='Dark', scatter_kws={'alpha':0.1})
sns.regplot(data=df[df['skin_group']=='Other/Unknown'], x='SpO2', y='SaO2', label='Other/Unknown', scatter_kws={'alpha':0.1})
plt.plot([70, 100], [70, 100], 'k--', label='Perfect Calibration')
plt.legend()
plt.title('Calibration Check: SpO2 vs SaO2')

threshold = 92
df['favorable_outcome'] = (df['SpO2'] >= threshold).astype(int)

selection_rates = df.groupby('skin_group')['favorable_outcome'].mean()

unprivileged_rate = selection_rates['Dark']
privileged_rate = selection_rates['Light']

dir_ratio = unprivileged_rate / privileged_rate

print(f"Selection Rate (Dark): {unprivileged_rate:.3f}")
print(f"Selection Rate (Light): {privileged_rate:.3f}")
print(f"Disparate Impact Ratio: {dir_ratio:.3f}")

if dir_ratio < 0.8:
    print("Result: Significant Disparate Impact detected (Below 0.8)")
elif dir_ratio > 1.25:
    print("Result: Significant Disparate Impact detected (Above 1.25)")
else:
    print("Result: Demographic Parity is generally met (within 0.8 - 1.25 range)")

threshold = 92

selection_rate_dark = (df[df['skin_group'] == 'Dark']['SpO2'] >= threshold).mean()
selection_rate_light = (df[df['skin_group'] == 'Light']['SpO2'] >= threshold).mean()
dir_ratio = selection_rate_dark / selection_rate_light

prevalence_rate_dark = (df[df['skin_group'] == 'Dark']['SaO2'] >= threshold).mean()
prevalence_rate_light = (df[df['skin_group'] == 'Light']['SaO2'] >= threshold).mean()
prevalence_ratio = prevalence_rate_dark / prevalence_rate_light

print(f"Disparate Impact Ratio (SpO2): {dir_ratio:.3f}")
print(f"Actual Prevalence Ratio (SaO2): {prevalence_ratio:.3f}")
print(f"Difference: {dir_ratio - prevalence_ratio:.3f}")

mae = mean_absolute_error(df['SaO2'], df['SpO2'])
rmse = mean_squared_error(df['SaO2'], df['SpO2']) ** 0.5
print(f"Baseline MAE: {mae:.3f}")
print(f"Baseline RMSE: {rmse:.3f}")

target_col = "SaO2"
group_col = "skin_group"

candidate_features = [
    "SpO2", "delta_SpO2", "admission_age", "sex_female", "BMI_admission",
    "vitals_heart_rate", "vitals_resp_rate", "vitals_mbp_ni", "vitals_sbp_ni", "vitals_dbp_ni",
    "vitals_tempc", "cbc_hemoglobin", "cbc_hematocrit", "bmp_creatinine", "bmp_lactate",
    "pH", "pCO2", "pO2", "Carboxyhemoglobin", "Methemoglobin"
]

use_features = [c for c in candidate_features if c in df.columns]
required_cols = list(dict.fromkeys(use_features + [target_col, group_col, "SpO2"]))
work = df[required_cols].copy()

numeric_cols = [c for c in use_features if c != group_col]
for c in [target_col, "SpO2"] + numeric_cols:
    if c in work.columns:
        work[c] = pd.to_numeric(work[c], errors="coerce")

work = work.dropna(subset=[target_col, group_col])
print("Rows used:", len(work))
print("Features used:", use_features)

X = work[use_features].copy()
y = work[target_col].copy()
g = work[group_col].copy()
spo2_for_eval = work["SpO2"].copy()

X_train_raw, X_test_raw, y_train, y_test, g_train, g_test, spo2_train, spo2_test = train_test_split(
    X, y, g, spo2_for_eval,
    test_size=0.2,
    random_state=42,
    stratify=g
)

X_train = pd.get_dummies(X_train_raw, drop_first=False)
X_test = pd.get_dummies(X_test_raw, drop_first=False)
X_train, X_test = X_train.align(X_test, join="left", axis=1, fill_value=0)

train_medians = X_train.median(numeric_only=True)
X_train = X_train.fillna(train_medians).fillna(0)
X_test = X_test.fillna(train_medians).fillna(0)

lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
pred_lr = lr_model.predict(X_test)

xgb_model = XGBRegressor(max_depth=3, n_estimators=100, learning_rate=0.1, random_state=42, n_jobs=-1)
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)

def reg_metrics(y_true, y_pred):
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": rmse,
        "R2": r2_score(y_true, y_pred)
    }

overall = pd.DataFrame([
    {"model": "LR", **reg_metrics(y_test, pred_lr)},
    {"model": "XGB", **reg_metrics(y_test, pred_xgb)}
]).set_index("model")

print("\nOverall performance")
print(overall.round(4).to_string())

def by_group_metrics(y_true, y_pred, groups, model_name):
    out = []
    tmp = pd.DataFrame({"y": y_true.values, "pred": y_pred, "group": groups.values})
    for grp, d in tmp.groupby("group"):
        if len(d) < 10:
            continue
        m = reg_metrics(d["y"], d["pred"])
        out.append({"model": model_name, "group": grp, "n": len(d), **m})
    return pd.DataFrame(out)

gm = pd.concat([
    by_group_metrics(y_test, pred_lr, g_test, "LR"),
    by_group_metrics(y_test, pred_xgb, g_test, "XGB")
], ignore_index=True)

print("\nGroup-wise performance")
print(gm.sort_values(["group", "model"]).round(4).to_string(index=False))

threshold = 88.0
true_hypox = y_test < threshold

alert_spo2 = spo2_test < threshold
alert_lr = pred_lr < threshold
alert_xgb = pred_xgb < threshold

def detection_report(true_event, alert, group_values, label):
    rows = []
    tmp = pd.DataFrame({"true": true_event.values, "alert": alert, "group": group_values.values})
    for grp, d in tmp.groupby("group"):
        n_true = int(d["true"].sum())
        if n_true == 0:
            continue
        tp = int(((d["true"] == 1) & (d["alert"] == 1)).sum())
        fn = int(((d["true"] == 1) & (d["alert"] == 0)).sum())
        recall = tp / (tp + fn) if (tp + fn) > 0 else np.nan
        rows.append({
            "model": label,
            "group": grp,
            "true_events": n_true,
            "TP": tp,
            "FN": fn,
            "Recall": recall
        })
    return pd.DataFrame(rows)

hypoxemia_rep = pd.concat([
    detection_report(true_hypox, alert_spo2, g_test, "SpO2 baseline"),
    detection_report(true_hypox, alert_lr, g_test, "LR predicted SaO2"),
    detection_report(true_hypox, alert_xgb, g_test, "XGB predicted SaO2")
], ignore_index=True)

print("\nHypoxemia recall by group")
print(hypoxemia_rep.sort_values(["group", "model"]).round(4).to_string(index=False))

eval_df = pd.DataFrame({
    "group": g_test.to_numpy(),
    "true_hypox": true_hypox.to_numpy(),
    "alert_xgb": np.asarray(alert_xgb),
    "y_true": y_test.to_numpy(),
    "y_pred": np.asarray(pred_xgb)
}).reset_index(drop=True)

plt.figure(figsize=(10, 6))
order = [g for g in ["Light", "Medium", "Dark", "Other/Unknown"] if g in eval_df["group"].unique()]
palette = sns.color_palette("PuBu", n_colors=max(4, len(order)))

for i, group in enumerate(order):
    subset = eval_df[eval_df["group"] == group]
    fn = ((subset["true_hypox"] == 1) & (subset["alert_xgb"] == 0)).sum()
    tp = ((subset["true_hypox"] == 1) & (subset["alert_xgb"] == 1)).sum()
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
    plt.bar(group, fnr, label=f"{group} (FNR={fnr:.2f})", color=palette[i])

plt.title("False Negative Rate by Skin Group (XGB Predicted SaO2)")
plt.xlabel("Skin Group")
plt.ylabel("False Negative Rate")
plt.legend(loc="lower left")
plt.show()

plt.figure(figsize=(10, 6))
for group in ["Light","Medium", "Dark", "Other/Unknown"]:
    subset = eval_df[eval_df["group"] == group]
    if len(subset) > 0:
        sns.regplot(
            x=subset["y_pred"],
            y=subset["y_true"],
            label=group,
            ci=None,
            scatter_kws={"alpha": 0.08, "s": 20},
            line_kws={"linewidth": 2}
        )

plt.plot([70, 100], [70, 100], "k--", label="Perfect Calibration")
plt.legend()
plt.title("Calibration Check: XGB Predicted SaO2 vs Actual SaO2")
plt.xlabel("Predicted SaO2")
plt.ylabel("Actual SaO2")
plt.show()

xgb_importance = xgb_model.get_booster().get_score(importance_type="weight")
importance_df = pd.DataFrame({
    "feature": list(xgb_importance.keys()),
    "importance": list(xgb_importance.values())
})
importance_df = importance_df.sort_values("importance", ascending=False)
print("\nXGB Feature Importance (by weight)")
print(importance_df.round(4).to_string(index=False))

from xgboost import XGBRegressor
from fairlearn.metrics import (
    MetricFrame,
    true_positive_rate,
    false_positive_rate,
    false_negative_rate,
    selection_rate,
)
from fairlearn.reductions import GridSearch, BoundedGroupLoss, SquareLoss, AbsoluteLoss
from sklearn.metrics import mean_absolute_error, mean_squared_error

hh_threshold = 88.0

y_hypox_train = (y_train < hh_threshold).astype(int)
y_hypox_test = (y_test < hh_threshold).astype(int)

def theil_index_manual(y_true_binary, y_pred_binary):

    b = np.asarray(y_pred_binary, dtype=float) - np.asarray(y_true_binary, dtype=float) + 1.0
    b = np.clip(b, 1e-12, None)
    mu = np.mean(b)
    r = b / mu
    return np.mean(r * np.log(r))

param = {
    "n_estimators": 250,
    "max_depth": 3,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "random_state": 42,
    "n_jobs": -1
}
xgb_reg = XGBRegressor(
    **param
)
xgb_reg.fit(X_train, y_train)
pred_reg = xgb_reg.predict(X_test)
alert_reg_hypox = (pred_reg < hh_threshold).astype(int)

constraints_sq = BoundedGroupLoss(
    loss=SquareLoss(min_val=50, max_val=100),
    upper_bound=6.0,
)
mitigator_sq = GridSearch(
    estimator=XGBRegressor(
         **param
    ),
    constraints=constraints_sq,
    grid_size=11,
)
mitigator_sq.fit(X_train, y_train, sensitive_features=g_train)
pred_fair_sq = mitigator_sq.predict(X_test)
alert_fair_sq = (pred_fair_sq < hh_threshold).astype(int)

constraints_abs = BoundedGroupLoss(
    loss=AbsoluteLoss(min_val=50, max_val=100),
    upper_bound=2.0,
)
mitigator_abs = GridSearch(
    estimator=XGBRegressor(
       **param
    ),
    constraints=constraints_abs,
    grid_size=11,
)
mitigator_abs.fit(X_train, y_train, sensitive_features=g_train)
pred_fair_abs = mitigator_abs.predict(X_test)
alert_fair_abs = (pred_fair_abs < hh_threshold).astype(int)

models = {
    "XGBRegressor (unmitigated)": (pred_reg, alert_reg_hypox),
    "XGBRegressor + GridSearch(SquareLoss)": (pred_fair_sq, alert_fair_sq),
    "XGBRegressor + GridSearch(AbsoluteLoss)": (pred_fair_abs, alert_fair_abs),
}

rows = []
for name, (pred_regression, y_pred_alert) in models.items():
    mf_tpr = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_hypox_test,
        y_pred=y_pred_alert,
        sensitive_features=g_test,
    )
    mf_fpr = MetricFrame(
        metrics=false_positive_rate,
        y_true=y_hypox_test,
        y_pred=y_pred_alert,
        sensitive_features=g_test,
    )
    mf_fnr = MetricFrame(
        metrics=false_negative_rate,
        y_true=y_hypox_test,
        y_pred=y_pred_alert,
        sensitive_features=g_test,
    )
    mf_sel = MetricFrame(
        metrics=selection_rate,
        y_true=y_hypox_test,
        y_pred=y_pred_alert,
        sensitive_features=g_test,
    )

    by_group = pd.DataFrame({
        "group": mf_tpr.by_group.index,
        "TPR_recall": mf_tpr.by_group.values,
        "FPR": mf_fpr.by_group.values,
        "FNR": mf_fnr.by_group.values,
        "SelectionRate": mf_sel.by_group.values,
    })

    mae = mean_absolute_error(y_test, pred_regression)
    rmse = mean_squared_error(y_test, pred_regression) ** 0.5

    by_group["MAE_overall"] = mae
    by_group["RMSE_overall"] = rmse
    by_group["model"] = name
    rows.append(by_group)

    eod = mf_tpr.difference()
    aod = 0.5 * (mf_tpr.difference() + mf_fpr.difference())
    theil = theil_index_manual(y_hypox_test, y_pred_alert)

    print(f"\n{name}")
    print("Equal Opportunity Difference:", round(eod, 4))
    print("Average Odds Difference:", round(aod, 4))
    print("Theil Index:", round(theil, 4))
    print("Recall disparity (max-min):", round(mf_tpr.difference(), 4))
    print("FNR disparity (max-min):", round(mf_fnr.difference(), 4))
    print("MAE overall:", round(mae, 4), "| RMSE overall:", round(rmse, 4))

fairlearn_compare = pd.concat(rows, ignore_index=True)
display(
    fairlearn_compare.sort_values(["model","group"]).reset_index(drop=True).round(4)
)

summary_rows = []
for name, (pred_regression, y_pred_alert) in models.items():
    mf_tpr = MetricFrame(
        metrics=true_positive_rate,
        y_true=y_hypox_test,
        y_pred=y_pred_alert,
        sensitive_features=g_test,
    )
    mf_fpr = MetricFrame(
        metrics=false_positive_rate,
        y_true=y_hypox_test,
        y_pred=y_pred_alert,
        sensitive_features=g_test,
    )
    summary_rows.append({
        "model": name,
        "EqualOpportunityDiff": mf_tpr.difference(),
        "AverageOddsDiff": 0.5 * (mf_tpr.difference() + mf_fpr.difference()),
        "TheilIndex": theil_index_manual(y_hypox_test, y_pred_alert),
        "MAE": mean_absolute_error(y_test, pred_regression),
        "RMSE": mean_squared_error(y_test, pred_regression) ** 0.5,
    })

summary_df = pd.DataFrame(summary_rows).round(4)
print("\nModel-level fairness/accuracy summary")
print(summary_df.to_string(index=False))
