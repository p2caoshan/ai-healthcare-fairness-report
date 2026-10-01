import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.linear_model import LogisticRegression
import sklearn.metrics as metrics
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import calibration_curve
from sklearn.utils import resample
from scipy.ndimage import uniform_filter1d

import warnings
warnings.filterwarnings('ignore')
import kagglehub

path = 'kaggle/input/heart-failure-prediction'

print("Path to dataset files:", path)
print("Pandas version:", pd.__version__)
print("Numpy version:", np.__version__)
print("Seaborn version:", sns.__version__)
print("Scikit-learn version:", sklearn.__version__)
print("KaggleHub version:", kagglehub.__version__)
print("Matplotlib version:", matplotlib.__version__)

df = pd.read_csv(path + "/heart.csv")

print(df.shape)

df.isnull().sum().sort_values(ascending=False)

df.select_dtypes(include=['object']).describe().T

df.select_dtypes(exclude=['object']).describe().T

df.hist(figsize=(12,8))

plt.figure(figsize=(12,7))

i = 1
for col in df.select_dtypes(include=['object']).columns:
    plt.subplot(2, 3, i)
    i+=1
    df[col].value_counts().plot(kind='bar')
    plt.xticks(rotation=0)
    plt.xlabel(None)
    plt.title(col)
plt.show()

categorical_cols = df.select_dtypes(include=['object']).columns

for col in categorical_cols:
    df[col] = df[col].astype('category').cat.codes
sns.heatmap(df.corr(), cmap='coolwarm')

X = df.drop('HeartDisease', axis=1)
y = df['HeartDisease']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

lr = LogisticRegression(random_state=42)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': [ 'l2', None]}
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(X_train[:,0].reshape(-1, 1), y_train)
print("Best parameters: ", grid_search.best_params_)

lr_age = LogisticRegression(**grid_search.best_params_, random_state=42)
lr_age.fit(X_train[:,0].reshape(-1, 1), y_train)

y_pred = lr_age.predict(X_test[:,0].reshape(-1, 1))
lr_age_auc = roc_auc_score(y_test, y_pred)
print("AUC: ", lr_age_auc)

lr = LogisticRegression(random_state=42)
param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100],
              'penalty': [ 'l2', None]}
grid_search = GridSearchCV(lr, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_params_)

lr = LogisticRegression(**grid_search.best_params_, random_state=42)
lr.fit(X_train, y_train)

y_pred = lr.predict(X_test)
lr_all_auc = roc_auc_score(y_test, y_pred)
print("AUC: ", lr_all_auc)

rf = RandomForestClassifier(random_state=42)
param_grid = {'n_estimators': [5, 10, 20, 50, 100],
              'max_depth': [2, 5, 10, None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

grid_search = GridSearchCV(rf, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_params_)

rf = RandomForestClassifier(**grid_search.best_params_, random_state=42)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)
rf_auc = roc_auc_score(y_test, y_pred)
print("AUC: ", rf_auc)

gb = GradientBoostingClassifier(random_state=42)
param_grid = {'n_estimators': [5, 10, 20, 50, 100],
              'learning_rate': [0.01, 0.001, 0.1],
              'max_depth': [2, 5, 10, None],
              'min_samples_split': [2, 5, 10],
              'min_samples_leaf': [1, 2, 4]}

grid_search = GridSearchCV(gb, param_grid, cv=5)
grid_search.fit(X_train, y_train)
print("Best parameters: ", grid_search.best_params_)

gb = GradientBoostingClassifier(**grid_search.best_params_, random_state=42)
gb.fit(X_train, y_train)

y_pred = gb.predict(X_test)
gb_auc = roc_auc_score(y_test, y_pred)
print("AUC: ", gb_auc)

print("AUC of logistic regression with age only: ", lr_age_auc)
print("AUC of logistic regression with all features: ", lr_all_auc)
print("AUC of random forest: ", rf_auc)
print("AUC of gradient boosting: ", gb_auc)

feature_importance = pd.DataFrame({'feature': X.columns, 'importance': lr.coef_[0]})
feature_importance = feature_importance.sort_values(by='importance', ascending=False)
feature_importance

figure, axes = plt.subplots(1, 4, figsize=(15, 5))
for ax, model in zip(figure.axes, [lr_age, lr, rf, gb]):
  if model == lr_age:
    prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test[:,0].reshape(-1, 1))[:, 1], n_bins=10, strategy="quantile")
  else:
    prob_true, prob_pred = calibration_curve(y_test, model.predict_proba(X_test)[:, 1], n_bins=10, strategy="quantile")
  ax.scatter(prob_pred, prob_true)
  ax.plot([0, 1], [0,1], 'k--')
  ax.set_xlabel('Predicted probabilities')
  ax.set_ylabel('True probabilities')
  ax.set_title(type(model).__name__)

plt.tight_layout()
plt.show()

matrix_df = pd.DataFrame(columns=['Model','Precision', 'Recall', 'F1', 'AUC', 'Brier'])

for model in [lr_age, lr, rf, gb]:
    if model == lr_age:
        y_pred = model.predict(X_test[:,0].reshape(-1, 1))
        model_nm = type(model).__name__ + '_age'

    else:
        y_pred = model.predict(X_test)
        model_nm = type(model).__name__
    new_df = pd.DataFrame([[model_nm,
                            metrics.precision_score(y_test, y_pred),
                            metrics.recall_score(y_test, y_pred),
                            metrics.f1_score(y_test, y_pred),
                            metrics.roc_auc_score(y_test, y_pred),
                            metrics.brier_score_loss(y_test, y_pred)]],
                          columns=['Model','Precision', 'Recall', 'F1', 'AUC', 'Brier'])
    matrix_df = pd.concat([matrix_df, new_df])

matrix_df.set_index('Model')

def UtilPerfBin(y, p, cut, costratio):
    NB = np.mean((p >= cut) * (y == 1)) - (cut / (1 - cut)) * np.mean((p >= cut) * (y == 0))
    SNB = NB / np.mean(y)
    risksort = np.sort(p)
    ecpts = pd.DataFrame(np.nan, index=range(len(risksort)), columns=range(3))
    for i in range(len(risksort)):
        ecpts.iloc[i, 0] = np.sum(p[y == 1] < risksort[i]) / np.sum(y == 1)
        ecpts.iloc[i, 1] = np.sum(p[y == 0] >= risksort[i]) / np.sum(y == 0)
        ecpts.iloc[i, 2] = (ecpts.iloc[i, 0] * np.mean(y) * (costratio if costratio > 1 else 1) + ecpts.iloc[i, 1] * (1 - np.mean(y)) * (1 if costratio > 1 else costratio))
    EC = np.min(ecpts.iloc[:, 2])
    ECthreshold = risksort[np.argmin(ecpts.iloc[:, 2])]
    return pd.DataFrame([[NB, SNB, EC, ECthreshold]], columns=["Net benefit", "Standardized net benefit", "Expected cost", "Threshold for EC"])

def net_benefit(y_true, probabilities, thresholds):
    """Calculate the Net Benefit (NB) at each threshold."""
    nb_values = []
    for threshold in thresholds:
        tp = np.sum((probabilities >= threshold) & (y_true == 1))
        fp = np.sum((probabilities >= threshold) & (y_true == 0))
        nb = (tp / len(y_true)) - (fp / len(y_true)) * (threshold / (1 - threshold))
        nb_values.append(nb)

    return np.array(nb_values)

def ecplotv(y, p, ncostfp):
    risksort = np.sort(p)
    ec_res = pd.DataFrame(np.nan, index=[0, 1], columns=range(len(ncostfp)))
    for i, cost_fp in enumerate(ncostfp):
        ecpts = pd.DataFrame(np.nan, index=range(len(risksort)), columns=[0, 1, 2])

        for j, threshold in enumerate(risksort):
            fn_rate = np.sum(p[y == 1] < threshold) / np.sum(y == 1)
            fp_rate = np.sum(p[y == 0] >= threshold) / np.sum(y == 0)
            ec_value = (fn_rate * np.mean(y) * (1 - cost_fp) +
                        fp_rate * (1 - np.mean(y)) * cost_fp)

            ecpts.iloc[j, 0] = fn_rate
            ecpts.iloc[j, 1] = fp_rate
            ecpts.iloc[j, 2] = ec_value

        ec_res.iloc[0, i] = ecpts.iloc[:, 2].min()
        ec_res.iloc[1, i] = risksort[ecpts.iloc[:, 2].idxmin()]

    return ec_res

def plot_single_model_curves(ax_nb, ax_ec, y_test, y_prob, model_name):
    thresholds = np.linspace(0, 1, 101)

    nb_values = net_benefit(y_test, y_prob, thresholds)

    nb_smoothed = uniform_filter1d(nb_values, size=5)

    ncostfp = np.arange(1, 100) / 100
    ecmodel = ecplotv(y = y_test, p = y_prob,
                      ncostfp = ncostfp)

    ax_nb.plot(thresholds, nb_values, label=model_name, lw=2)
    ax_nb.plot(thresholds, nb_smoothed, label=f"{model_name} (Smoothed)", linestyle="--", lw=1)

    x_vals = 1 - ncostfp
    ax_ec.plot(x_vals, ecmodel.iloc[0, :], lw=2, label=model_name)

fig, (ax_nb, ax_ec) = plt.subplots(1, 2, figsize=(18, 7))

thresholds = np.linspace(0, 1, 101)
mean_outcome = y_test.mean()
ncostfp = np.arange(1, 100) / 100

ax_nb.axhline(0, color='red', linestyle='-', lw=1, label='Treat None')

nb_treat_all = mean_outcome - (1 - mean_outcome) * (thresholds / (1 - thresholds))

nb_treat_all[thresholds == 1] = mean_outcome
ax_nb.plot(thresholds, nb_treat_all, color='orange', linestyle='-', lw=1, label='Treat All')

ecTA = ncostfp * (mean_outcome / (1 - mean_outcome))
ecTN = (1 - ncostfp) * ((1 - mean_outcome) / mean_outcome)
x_vals = 1 - ncostfp
ax_ec.plot(x_vals, ecTA, color='orange', linewidth=1, label='Treat All')
ax_ec.plot(x_vals, ecTN, color='red', linewidth=1, label='Treat None')

for model in [lr_age, lr, rf, gb]:
    if model == lr_age:
        y_prob = model.predict_proba(X_test[:,0].reshape(-1, 1))[:, 1]
        model_name = type(model).__name__ + '_age'
    else:
        y_prob = model.predict_proba(X_test)[:, 1]
        model_name = type(model).__name__
    plot_single_model_curves(ax_nb, ax_ec, y_test, y_prob, model_name)

ax_nb.set_xlabel("Decision threshold")
ax_nb.set_ylabel("Net Benefit")
ax_nb.set_ylim(-0.05, 0.5)
ax_nb.set_xlim(0, 1)
ax_nb.legend(loc="lower left")
ax_nb.set_title("Net Benefit Curve")
ax_nb.grid(True, linestyle='--', alpha=0.6)

ax_ec.set_xlabel("Normalized cost of false negative", fontsize=12, labelpad=10)
ax_ec.set_ylabel("Expected cost", fontsize=12, labelpad=10)
ax_ec.set_xlim(0, 1)
ax_ec.set_ylim(0, 0.5)
ax_ec.set_title("Expected Cost Curve")
ax_ec.legend(loc="upper right")
ax_ec.grid(True, linestyle='--', alpha=0.6)

plt.tight_layout()
plt.show()

from sklearn.utils import resample

n_iterations = 1000
scores = []
y_pred_proba = lr.predict_proba(X_test)[:, 1]

for i in range(n_iterations):

    y_test_resampled, y_pred_resampled = resample(y_test, y_pred_proba, replace=True, stratify=y_test)

    score = roc_auc_score(y_test_resampled, y_pred_resampled)
    scores.append(score)

lower = np.percentile(scores, 2.5)
upper = np.percentile(scores, 97.5)

print(f"AUC: {np.mean(scores):.3f} \n95% CI: ({lower:.3f} - {upper:.3f})")
