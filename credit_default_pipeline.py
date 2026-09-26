# %% [markdown]
# # Predictive Modelling for Credit Card Default Using Machine Learning
# **MSc Artificial Intelligence – Introduction to Artificial Intelligence (Practical skills assessment)**
#
# **How to replicate:** (1) put `Credit_Card.csv` in the same folder as this notebook (or change `DATA_PATH`),
# (2) `pip install pandas numpy scikit-learn matplotlib seaborn scipy shap`,
# (3) run all cells top to bottom. Everything is seeded with `RANDOM_STATE = 42`.
# Figures are saved to `figures/` and tables to `tables/`.

# %%
# ---- Imports and configuration -------------------------------------------------------------
import os, time, json, warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from sklearn.base import clone
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold, cross_val_predict
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
                             roc_curve, ConfusionMatrixDisplay, precision_recall_curve)
import shap

warnings.filterwarnings("ignore")
sns.set_theme(style="whitegrid", context="notebook")

DATA_PATH = "Credit_Card.csv"     # semicolon-separated file supplied with the assignment
TARGET = "default.payment.next.month"
RANDOM_STATE = 42
FIG_DIR, TAB_DIR = "figures", "tables"
os.makedirs(FIG_DIR, exist_ok=True); os.makedirs(TAB_DIR, exist_ok=True)
KEY = {}                          # key numbers collected for the report


def savefig(name):
    plt.tight_layout()
    plt.savefig(f"{FIG_DIR}/{name}.png", dpi=150, bbox_inches="tight")
    plt.show()


def savetab(df_, name, index=True):
    df_.to_csv(f"{TAB_DIR}/{name}.csv", index=index)

# %% [markdown]
# ## 0. Loading and repairing the raw file
# The raw CSV has several quality problems that are not mentioned in the brief. They are handled here
# *before* any analysis and are documented in the report:
# 1. Fields are separated by `;` (not `,`).
# 2. **4,788 exact duplicate rows** (IDs repeated) → the file has 34,788 rows but only 30,000 unique clients.
# 3. `LIMIT_BAL_LOG` and `risk_leak` were damaged by a spreadsheet export (e.g. `10.819.798.284.210.200`
#    instead of `10.8197982842102`, or `-2,00E+11`).
# 4. In 288 rows `LIMIT_BAL` is exactly 10x larger than the value implied by `LIMIT_BAL_LOG` (injected outliers).
# 5. `BILL_AMT_SUM` disagrees with the sum of `BILL_AMT1..6` in ~1% of rows.

# %%
raw = pd.read_csv(DATA_PATH, sep=";", dtype=str)
print("Raw shape:", raw.shape)
df = raw.drop_duplicates().copy()
print("After removing exact duplicates:", df.shape, "| duplicated IDs left:", df["ID"].duplicated().sum())
KEY["rows_raw"], KEY["rows_dedup"] = len(raw), len(df)

# --- generic numeric columns -------------------------------------------------------------
special = ["CITY", "risk_leak", "LIMIT_BAL_LOG"]
for c in df.columns.difference(special):
    df[c] = pd.to_numeric(df[c], errors="coerce")


# --- LIMIT_BAL_LOG: dots were inserted as thousands separators -> restore the decimal point ----
def parse_limit_log(x):
    if pd.isna(x):
        return np.nan
    digits = x.replace(".", "")
    for k in (1, 2):                              # ln(limit) lies between ~9 and ~15
        v = float(digits[:k] + "." + digits[k:])
        if 8 <= v <= 15:
            return v
    return np.nan


# --- risk_leak: three formats: normal float / dotted (damaged) / scientific (irrecoverable) --
def parse_risk_leak(x):
    if pd.isna(x):
        return np.nan
    if "E" in x.upper():                          # e.g. '-2,00E+11' -> information destroyed -> missing
        return np.nan
    if x.count(".") >= 2:                         # e.g. '1.075.675.276.433.290' -> 1.07567527643329
        d = x.replace(".", "")
        return float(d[0] + "." + d[1:])
    return float(x)


fmt = pd.Series(np.where(df["risk_leak"].str.upper().str.contains("E", na=False), "scientific",
                np.where(df["risk_leak"].str.count(r"\.") >= 2, "dotted", "normal")), index=df.index)
print("risk_leak formats:", fmt.value_counts().to_dict())
KEY["risk_leak_formats"] = fmt.value_counts().to_dict()

df["risk_leak"] = df["risk_leak"].map(parse_risk_leak)
df["LIMIT_BAL_LOG"] = df["LIMIT_BAL_LOG"].map(parse_limit_log)

# --- LIMIT_BAL: fix values inflated x10 (LIMIT_BAL_LOG gives the true value) -------------------
ratio = df["LIMIT_BAL"] / np.exp(df["LIMIT_BAL_LOG"])
inflated = np.isclose(ratio, 10, rtol=0.01)
print("LIMIT_BAL inflated x10:", inflated.sum(), "| max before:", df["LIMIT_BAL"].max())
KEY["limit_bal_inflated_rows"] = int(inflated.sum())
limit_example = df.loc[inflated, ["ID", "LIMIT_BAL", "LIMIT_BAL_LOG"]].head(3).copy()
df.loc[inflated, "LIMIT_BAL"] = df.loc[inflated, "LIMIT_BAL"] / 10
limit_example["LIMIT_BAL_fixed"] = df.loc[limit_example.index, "LIMIT_BAL"]
print(limit_example.to_string(index=False)); print("max after:", df["LIMIT_BAL"].max())

# --- BILL_AMT_SUM: recompute from its components ------------------------------------------------
bill_cols = [f"BILL_AMT{i}" for i in range(1, 7)]
pay_amt_cols = [f"PAY_AMT{i}" for i in range(1, 7)]
mismatch = ((df[bill_cols].sum(axis=1, min_count=6) - df["BILL_AMT_SUM"]).abs() >= 1).sum()
print("BILL_AMT_SUM inconsistent rows:", mismatch)
KEY["bill_sum_inconsistent_rows"] = int(mismatch)
df["BILL_AMT_SUM"] = df[bill_cols].sum(axis=1)

df[TARGET] = df[TARGET].astype(int)
df = df.reset_index(drop=True)
print("Clean shape:", df.shape)

# %% [markdown]
# ## Task 1 – Exploratory Data Analysis

# %%
# ---- 1.1 Target distribution and class imbalance -------------------------------------------------
vc = df[TARGET].value_counts().sort_index()
pct = df[TARGET].value_counts(normalize=True).sort_index() * 100
print(pd.DataFrame({"count": vc, "percent": pct.round(2)}))
print(f"Imbalance ratio (non-default : default) = {vc[0] / vc[1]:.2f} : 1")
KEY["default_rate"] = float(pct[1] / 100); KEY["imbalance_ratio"] = float(vc[0] / vc[1])
KEY["n_default"], KEY["n_nodefault"] = int(vc[1]), int(vc[0])

fig, ax = plt.subplots(figsize=(5, 4))
sns.barplot(x=["No default (0)", "Default (1)"], y=vc.values, palette=["#4C72B0", "#C44E52"], ax=ax)
for i, (v, p) in enumerate(zip(vc.values, pct.values)):
    ax.text(i, v + 300, f"{v:,}\n({p:.1f}%)", ha="center")
ax.set_ylabel("Clients"); ax.set_title("Target distribution"); ax.set_ylim(0, vc.max() * 1.15)
savefig("fig01_target_distribution")

# %%
# ---- 1.2 Descriptive statistics of key numerical features ---------------------------------------
key_num = ["LIMIT_BAL", "AGE", "BILL_AMT_SUM", "LIMIT_BAL_LOG", "risk_leak"]


def iqr_outliers(s):
    q1, q3 = s.quantile([.25, .75]); i = q3 - q1
    return int(((s < q1 - 1.5 * i) | (s > q3 + 1.5 * i)).sum())


desc = pd.DataFrame({
    "mean": df[key_num].mean(), "median": df[key_num].median(), "std": df[key_num].std(),
    "min": df[key_num].min(), "max": df[key_num].max(), "skewness": df[key_num].skew(),
    "IQR outliers": [iqr_outliers(df[c].dropna()) for c in key_num],
    "missing": df[key_num].isna().sum()})
print(desc.round(3).to_string()); savetab(desc.round(3), "table01_descriptive_stats")

# %%
# ---- 1.3 Histograms and box plots (skewness / outliers) -----------------------------------------
fig, axes = plt.subplots(2, 5, figsize=(20, 7))
for j, c in enumerate(key_num):
    sns.histplot(df[c].dropna(), kde=True, bins=40, ax=axes[0, j], color="#4C72B0")
    axes[0, j].set_title(f"{c}\nskew={df[c].skew():.2f}")
    sns.boxplot(x=df[TARGET], y=df[c], ax=axes[1, j], palette=["#4C72B0", "#C44E52"])
    axes[1, j].set_xlabel("default (0/1)")
savefig("fig02_hist_box_key_numeric")

# %%
# ---- 1.4 Categorical features: distribution, default rate, association with target ---------------
cats = ["SEX", "EDUCATION", "MARRIAGE", "RISK_RATING"]
fig, axes = plt.subplots(2, 4, figsize=(20, 8))
for j, c in enumerate(cats):
    order = sorted(df[c].dropna().unique())
    sns.countplot(x=df[c], order=order, ax=axes[0, j], color="#4C72B0"); axes[0, j].set_title(f"{c}: counts")
    rate = df.groupby(c)[TARGET].mean().reindex(order) * 100
    sns.barplot(x=rate.index, y=rate.values, ax=axes[1, j], color="#C44E52")
    axes[1, j].axhline(df[TARGET].mean() * 100, ls="--", c="k"); axes[1, j].set_title(f"{c}: default rate (%)")
savefig("fig03_categorical_distributions")

city_rate = df.groupby("CITY")[TARGET].agg(["mean", "count"]).sort_values("mean", ascending=False)
fig, ax = plt.subplots(figsize=(16, 4))
sns.barplot(x=city_rate.index, y=city_rate["mean"] * 100, ax=ax, color="#8172B2")
ax.axhline(df[TARGET].mean() * 100, ls="--", c="k"); ax.set_ylabel("Default rate (%)")
ax.set_title(f"CITY ({df.CITY.nunique()} categories): default rate"); plt.xticks(rotation=90)
savefig("fig04_city_default_rate")


def cramers_v(a, b):
    ct = pd.crosstab(a, b); chi2, p, _, _ = stats.chi2_contingency(ct)
    return np.sqrt(chi2 / (ct.values.sum() * (min(ct.shape) - 1))), p


assoc = pd.DataFrame({c: cramers_v(df[c].fillna("missing"), df[TARGET]) for c in cats + ["CITY"]},
                     index=["Cramer's V", "chi2 p-value"]).T
print(assoc.round(4)); savetab(assoc.round(4), "table02_categorical_association")
print("\nDefault rate by RISK_RATING (%):"); print((df.groupby("RISK_RATING")[TARGET].mean() * 100).round(2))

# %%
# ---- 1.5 Correlation of numerical features with the target -------------------------------------
num_all = [c for c in df.select_dtypes("number").columns if c not in ("ID", TARGET)]
corr_t = df[num_all].corrwith(df[TARGET]).sort_values(key=np.abs, ascending=False)
spear = df[num_all].apply(lambda s: stats.spearmanr(s, df[TARGET], nan_policy="omit")[0])
corr_tab = pd.DataFrame({"pearson": corr_t, "spearman": spear[corr_t.index]})
print(corr_tab.round(3).to_string()); savetab(corr_tab.round(3), "table03_correlation_with_target")

fig, ax = plt.subplots(figsize=(7, 9))
sns.barplot(x=corr_t.values, y=corr_t.index, palette=["#C44E52" if v > 0 else "#4C72B0" for v in corr_t.values], ax=ax)
ax.set_title("Pearson correlation with default"); ax.set_xlabel("r")
savefig("fig05_correlation_with_target")

top = list(corr_t.index[:10])
plt.figure(figsize=(9, 7))
sns.heatmap(df[top + [TARGET]].corr(), annot=True, fmt=".2f", cmap="coolwarm", center=0)
plt.title("Correlation matrix: top-10 features + target")
savefig("fig06_correlation_heatmap")

# %%
# ---- 1.6 Payment history, RISK_RATING and risk_leak vs default ---------------------------------
pay_cols = ["PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6"]
fig, axes = plt.subplots(1, 3, figsize=(18, 4.5))
rates = pd.DataFrame({c: df.groupby(c)[TARGET].mean() * 100 for c in pay_cols})
sns.heatmap(rates, annot=True, fmt=".0f", cmap="Reds", ax=axes[0], cbar_kws={"label": "default rate %"})
axes[0].set_title("Default rate (%) by payment-status code"); axes[0].set_ylabel("status code")
sns.boxplot(x=df[TARGET], y=df["risk_leak"], ax=axes[1], palette=["#4C72B0", "#C44E52"])
axes[1].set_title("risk_leak by default status")
sns.barplot(x=df["RISK_RATING"], y=df[TARGET] * 100, ax=axes[2], color="#C44E52", errorbar=None)
axes[2].set_title("Default rate (%) by RISK_RATING")
savefig("fig07_pay_risk_leak")

# leakage diagnostics for risk_leak / RISK_RATING
rl = df.dropna(subset=["risk_leak"])
KEY["risk_leak_auc_alone"] = float(roc_auc_score(rl[TARGET], rl["risk_leak"]))
KEY["risk_leak_max_class0"] = float(rl.loc[rl[TARGET] == 0, "risk_leak"].max())
KEY["risk_leak_min_class1"] = float(rl.loc[rl[TARGET] == 1, "risk_leak"].min())
KEY["risk_rating_auc_alone"] = float(roc_auc_score(df[TARGET], df["RISK_RATING"]))
KEY["pay0_auc_alone"] = float(roc_auc_score(df[TARGET], df["PAY_0"].fillna(0)))
print({k: round(v, 4) for k, v in KEY.items() if "alone" in k or "risk_leak_m" in k})

# %% [markdown]
# ## Task 2 – Data preparation
# **Order matters:** the dataset is split (stratified) *first*, and imputation / scaling / encoding are
# *fitted on the training set only*, then applied to the test set. This avoids data leakage from the test set.

# %%
# ---- 2.1 Missing values ---------------------------------------------------------------------------
miss = df.isna().sum(); miss = miss[miss > 0]
miss_tab = pd.DataFrame({"missing": miss, "percent": (miss / len(df) * 100).round(2)})
inf = []
for c in miss.index:                                   # is missingness related to the target? (MCAR check)
    m = df[c].isna()
    p = stats.chi2_contingency(pd.crosstab(m, df[TARGET]))[1]
    inf.append((df.loc[m, TARGET].mean() * 100, df.loc[~m, TARGET].mean() * 100, p))
miss_tab[["default% if missing", "default% if present", "chi2 p"]] = np.round(inf, 4)
method = {"LIMIT_BAL": "median (right-skewed)", "LIMIT_BAL_LOG": "dropped (redundant with LIMIT_BAL)",
          "AGE": "median", "PAY_AMT1": "median (right-skewed)", "PAY_AMT2": "median (right-skewed)",
          "SEX": "mode", "EDUCATION": "mode", "MARRIAGE": "mode",
          "risk_leak": "median (only used in the leakage experiment)"}
miss_tab["chosen method"] = [method.get(c, "median") for c in miss_tab.index]
print(miss_tab.to_string()); savetab(miss_tab, "table04_missing_values")
KEY["rows_with_any_missing"] = int(df.isna().any(axis=1).sum())
KEY["pct_rows_any_missing"] = float(df.isna().any(axis=1).mean() * 100)

# %%
# ---- 2.2 Rare / undocumented category codes -------------------------------------------------------
print("EDUCATION before:", df["EDUCATION"].value_counts(dropna=False).sort_index().to_dict())
print("MARRIAGE  before:", df["MARRIAGE"].value_counts(dropna=False).sort_index().to_dict())
df["EDUCATION"] = df["EDUCATION"].replace({0: 4, 5: 4, 6: 4})   # 0,5,6 are undocumented -> 'others'
df["MARRIAGE"] = df["MARRIAGE"].replace({0: 3})                  # 0 undocumented -> 'others'
print("EDUCATION after :", df["EDUCATION"].value_counts(dropna=False).sort_index().to_dict())
print("MARRIAGE  after :", df["MARRIAGE"].value_counts(dropna=False).sort_index().to_dict())

# %%
# ---- 2.3 Feature sets, stratified split, preprocessing pipeline ----------------------------------
USE_CITY = False        # CITY: see Task 1 chi-square test; set True to one-hot encode it
cat_cols = ["SEX", "EDUCATION", "MARRIAGE"] + (["CITY"] if USE_CITY else [])
num_base = (["LIMIT_BAL", "AGE"] + pay_cols + bill_cols + pay_amt_cols + ["BILL_AMT_SUM", "RISK_RATING"])
num_leak = num_base + ["risk_leak"]     # scenario A only (leakage experiment)

X_all, y_all = df.drop(columns=[TARGET, "ID", "LIMIT_BAL_LOG"]), df[TARGET]
X_train, X_test, y_train, y_test = train_test_split(
    X_all, y_all, test_size=0.20, stratify=y_all, random_state=RANDOM_STATE)
print(f"Train {X_train.shape} | Test {X_test.shape}")
print("Default rate  train: %.4f | test: %.4f" % (y_train.mean(), y_test.mean()))
KEY.update(n_train=len(X_train), n_test=len(X_test), n_test_defaults=int(y_test.sum()))


def build_preprocessor(num_cols, cat_cols):
    num = Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())])
    cat = Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))])
    ct = ColumnTransformer([("num", num, num_cols), ("cat", cat, cat_cols)])
    return ct.set_output(transform="pandas")


prep = build_preprocessor(num_base, cat_cols).fit(X_train)     # fitted on TRAIN only
Xtr_t = prep.transform(X_train)
print("Transformed training matrix:", Xtr_t.shape, "| any NaN left:", bool(Xtr_t.isna().any().any()))

# %%
# ---- 2.4 Before-and-after examples ----------------------------------------------------------------
show_cols = ["LIMIT_BAL", "AGE", "SEX", "EDUCATION", "PAY_AMT1"]
rows_missing = X_train[X_train[show_cols].isna().any(axis=1)].head(4)
before = rows_missing[show_cols]
after = Xtr_t.loc[rows_missing.index, [c for c in Xtr_t.columns if c.split("__")[1].split("_")[0] in
                                       ("LIMIT", "AGE", "SEX", "EDUCATION", "PAY") and
                                       any(k in c for k in ("LIMIT_BAL", "AGE", "SEX", "EDUCATION", "PAY_AMT1"))]]
print("BEFORE (raw, with NaN):"); print(before.to_string())
print("\nAFTER (imputed, scaled, one-hot):"); print(after.round(3).to_string())
savetab(before, "table05a_before_example"); savetab(after.round(3), "table05b_after_example")

ba = pd.DataFrame({"mean before": X_train[["LIMIT_BAL", "AGE", "PAY_AMT1"]].mean(),
                   "std before": X_train[["LIMIT_BAL", "AGE", "PAY_AMT1"]].std(),
                   "mean after": Xtr_t[["num__LIMIT_BAL", "num__AGE", "num__PAY_AMT1"]].mean().values,
                   "std after": Xtr_t[["num__LIMIT_BAL", "num__AGE", "num__PAY_AMT1"]].std().values})
print("\nScaling effect:"); print(ba.round(3).to_string()); savetab(ba.round(3), "table06_scaling_effect")

# %% [markdown]
# ## Task 3 – Model training (default hyper-parameters)
# Six classifiers are trained with scikit-learn default hyper-parameters (only `random_state` is fixed for
# reproducibility). Every model sits in a `Pipeline` with the preprocessing step, so imputation and scaling are
# always fitted on the training data only.

# %%
def make_models():
    return {
        "Logistic Regression": LogisticRegression(),
        "SVM (RBF)": SVC(random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(random_state=RANDOM_STATE),
        "KNN": KNeighborsClassifier(),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }


def scores_of(pipe, X):
    return pipe.predict_proba(X)[:, 1] if hasattr(pipe, "predict_proba") else pipe.decision_function(X)


def evaluate(pipe, X, y, thr=None):
    s = scores_of(pipe, X)
    pred = pipe.predict(X) if thr is None else (s >= thr).astype(int)
    return {"Accuracy": accuracy_score(y, pred), "Precision": precision_score(y, pred, zero_division=0),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred), "AUC-ROC": roc_auc_score(y, s)}


models = make_models()
# record all initial hyper-parameters BEFORE tuning
init_params = pd.DataFrame({n: pd.Series({k: str(v) for k, v in m.get_params().items()}) for n, m in models.items()})
init_params = init_params.fillna("-")
savetab(init_params, "table07_initial_hyperparameters"); print(init_params.to_string())

fitted, fit_time = {}, {}
for name, m in models.items():
    t0 = time.time()
    pipe = Pipeline([("prep", clone(prep)), ("clf", m)]).fit(X_train, y_train)
    fitted[name], fit_time[name] = pipe, time.time() - t0
    print(f"{name:20s} trained in {fit_time[name]:6.1f}s")

# %% [markdown]
# ## Task 4 – Evaluation and visualisation

# %%
# ---- 4.1 Comparative metrics table (test set) -----------------------------------------------------
res = pd.DataFrame({n: evaluate(p, X_test, y_test) for n, p in fitted.items()}).T
res["Train time (s)"] = pd.Series(fit_time)
res = res.sort_values("AUC-ROC", ascending=False)
print(res.round(4).to_string()); savetab(res.round(4), "table08_model_comparison_default")
KEY["default_results"] = res.round(4).to_dict("index")
# overfitting check: train vs test AUC
gap = pd.DataFrame({"train AUC": {n: roc_auc_score(y_train, scores_of(p, X_train)) for n, p in fitted.items()},
                    "test AUC": res["AUC-ROC"]}).round(4)
gap["gap"] = (gap["train AUC"] - gap["test AUC"]).round(4)
print("\nTrain vs test AUC:"); print(gap.to_string()); savetab(gap, "table09_train_test_auc_gap")
KEY["train_test_gap"] = gap.to_dict("index")

# %%
# ---- 4.2 Confusion matrices and ROC curves for each model --------------------------------------
names = list(fitted)
fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, n in zip(axes.ravel(), names):
    ConfusionMatrixDisplay.from_predictions(y_test, fitted[n].predict(X_test), ax=ax, cmap="Blues",
                                            display_labels=["No default", "Default"], colorbar=False)
    ax.set_title(n); ax.grid(False)
savefig("fig08_confusion_matrices")

fig, axes = plt.subplots(2, 3, figsize=(15, 9))
for ax, n in zip(axes.ravel(), names):
    fpr, tpr, _ = roc_curve(y_test, scores_of(fitted[n], X_test))
    ax.plot(fpr, tpr, lw=2, label=f"AUC = {res.loc[n, 'AUC-ROC']:.3f}"); ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_title(n); ax.set_xlabel("False positive rate"); ax.set_ylabel("True positive rate"); ax.legend(loc="lower right")
savefig("fig09_roc_curves_each_model")

plt.figure(figsize=(7, 6))
for n in names:
    fpr, tpr, _ = roc_curve(y_test, scores_of(fitted[n], X_test)); plt.plot(fpr, tpr, label=f"{n} ({res.loc[n, 'AUC-ROC']:.3f})")
plt.plot([0, 1], [0, 1], "k--"); plt.legend(loc="lower right"); plt.title("ROC curves – all models (default settings)")
plt.xlabel("False positive rate"); plt.ylabel("True positive rate")
savefig("fig10_roc_curves_overlay")

# %%
# ---- 4.3 Hyper-parameter tuning of the best model(s) (RandomizedSearchCV, ROC-AUC, 3-fold CV) -----
param_space = {
    "Gradient Boosting": {"clf__n_estimators": [100, 200, 300], "clf__learning_rate": [0.03, 0.05, 0.1],
                          "clf__max_depth": [2, 3, 4], "clf__subsample": [0.7, 0.85, 1.0],
                          "clf__min_samples_leaf": [1, 20, 50]},
    "Random Forest": {"clf__n_estimators": [150, 300], "clf__max_depth": [8, 12, 16, None],
                      "clf__min_samples_leaf": [1, 5, 10, 20], "clf__max_features": ["sqrt", "log2", 0.3]},
    "Logistic Regression": {"clf__C": [0.01, 0.1, 1, 10, 100]},
    "Decision Tree": {"clf__max_depth": [3, 5, 8, 12], "clf__min_samples_leaf": [1, 10, 50, 100]},
    "KNN": {"clf__n_neighbors": [5, 15, 31, 51], "clf__weights": ["uniform", "distance"]},
    "SVM (RBF)": {"clf__C": [0.5, 1, 5], "clf__gamma": ["scale", 0.01]},
}
N_TUNE_MODELS, N_ITER = 2, 8
to_tune = list(res.index[:N_TUNE_MODELS])
cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
tuned, tune_rows, search_logs = {}, [], {}
for n in to_tune:
    t0 = time.time()
    base = Pipeline([("prep", clone(prep)), ("clf", make_models()[n])])
    rs = RandomizedSearchCV(base, param_space[n], n_iter=min(N_ITER, int(np.prod([len(v) for v in param_space[n].values()]))),
                            scoring="roc_auc", cv=cv, random_state=RANDOM_STATE, n_jobs=1, refit=True)
    rs.fit(X_train, y_train); tuned[n] = rs.best_estimator_
    b, a = res.loc[n], evaluate(rs.best_estimator_, X_test, y_test)
    tune_rows.append({"Model": n, "Stage": "default", **{k: b[k] for k in a}})
    tune_rows.append({"Model": n, "Stage": "tuned", **a})
    search_logs[n] = {k.replace("clf__", ""): v for k, v in rs.best_params_.items()}
    search_logs[n]["cv_auc"] = round(float(rs.best_score_), 4)
    print(f"{n}: best CV AUC={rs.best_score_:.4f} | params={search_logs[n]} | {time.time() - t0:.0f}s")
tune_tab = pd.DataFrame(tune_rows).set_index(["Model", "Stage"])
print(tune_tab.round(4).to_string()); savetab(tune_tab.round(4), "table10_before_after_tuning")
KEY["tuning"] = tune_tab.round(4).reset_index().to_dict("records"); KEY["best_params"] = search_logs

# %%
# ---- 4.4 Final model: pick by test AUC after tuning, handle class imbalance with a threshold ----
final_name = max(tuned, key=lambda n: evaluate(tuned[n], X_test, y_test)["AUC-ROC"])
final = tuned[final_name]
print("Final model:", final_name)
KEY["final_model"] = final_name

# threshold chosen on TRAINING data only (out-of-fold probabilities) -> no test leakage
oof = cross_val_predict(clone(final), X_train, y_train, cv=cv, method="predict_proba")[:, 1]
prec, rec, thr = precision_recall_curve(y_train, oof)
f1s = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
best_thr = float(thr[np.argmax(f1s)]); print(f"F1-optimal threshold from CV on train: {best_thr:.3f}")
thr_tab = pd.DataFrame({"threshold 0.50": evaluate(final, X_test, y_test),
                        f"threshold {best_thr:.2f}": evaluate(final, X_test, y_test, thr=best_thr)}).T
print(thr_tab.round(4).to_string()); savetab(thr_tab.round(4), "table11_threshold_effect")
KEY["threshold"] = {"value": best_thr, "results": thr_tab.round(4).to_dict("index")}

fig, axes = plt.subplots(1, 3, figsize=(17, 5))
s_final = scores_of(final, X_test)
ConfusionMatrixDisplay.from_predictions(y_test, (s_final >= .5).astype(int), ax=axes[0], cmap="Blues", colorbar=False,
                                        display_labels=["No default", "Default"]); axes[0].set_title(f"{final_name} tuned, thr=0.50"); axes[0].grid(False)
ConfusionMatrixDisplay.from_predictions(y_test, (s_final >= best_thr).astype(int), ax=axes[1], cmap="Blues", colorbar=False,
                                        display_labels=["No default", "Default"]); axes[1].set_title(f"thr={best_thr:.2f}"); axes[1].grid(False)
for n in to_tune:
    for lab, mdl in (("default", fitted[n]), ("tuned", tuned[n])):
        f_, t_, _ = roc_curve(y_test, scores_of(mdl, X_test))
        axes[2].plot(f_, t_, ls="--" if lab == "default" else "-", label=f"{n} {lab} ({roc_auc_score(y_test, scores_of(mdl, X_test)):.3f})")
axes[2].plot([0, 1], [0, 1], "k:"); axes[2].legend(loc="lower right"); axes[2].set_title("ROC before vs after tuning")
savefig("fig11_final_model_diagnostics")

# %% [markdown]
# ### 4.5 Interpretability with SHAP
# **Scenario A – with `risk_leak`** (leakage demonstration) and **Scenario B – without `risk_leak`** (realistic model).

# %%
def shap_pos(model, X):
    sv = shap.TreeExplainer(model).shap_values(X, check_additivity=False)
    if isinstance(sv, list): sv = sv[1]
    elif getattr(sv, "ndim", 2) == 3: sv = sv[:, :, 1]
    return sv


def original_name(c):
    c = c.split("__", 1)[1]
    return c.rsplit("_", 1)[0] if c.split("_")[0] in ("SEX", "EDUCATION", "MARRIAGE", "CITY") else c


def shap_report(pipe, X, tag, n=1200):
    Xt = pipe["prep"].transform(X).sample(n, random_state=RANDOM_STATE)
    sv = shap_pos(pipe["clf"], Xt)
    imp = pd.Series(np.abs(sv).mean(axis=0), index=Xt.columns)
    grp = imp.groupby([original_name(c) for c in imp.index]).sum().sort_values(ascending=False)
    plt.figure(); shap.summary_plot(sv, Xt.rename(columns=lambda c: original_name(c) if c.startswith("num__") else c[5:]),
                                    show=False, max_display=15); plt.title(f"SHAP summary – {tag}")
    savefig(f"fig_shap_beeswarm_{tag}")
    plt.figure(figsize=(7, 6)); grp.head(15)[::-1].plot.barh(color="#4C72B0"); plt.xlabel("mean |SHAP value|")
    plt.title(f"Mean |SHAP| – {tag}"); savefig(f"fig_shap_bar_{tag}")
    return grp, Xt, sv


# Scenario B: tuned final model, no risk_leak
grp_B, Xt_B, sv_B = shap_report(final, X_test, "B_final_model_no_leak")
tabB = (grp_B / grp_B.sum() * 100).round(2).rename("share of total |SHAP| (%)").to_frame()
tabB["mean |SHAP|"] = grp_B.round(4); print(tabB.head(12).to_string()); savetab(tabB, "table12_shap_importance_final")
KEY["shap_B_top"] = tabB.head(12).to_dict("index")

# dependence plot for PAY_0
plt.figure(figsize=(6, 4.5))
shap.dependence_plot("num__PAY_0", sv_B, Xt_B, show=False, interaction_index=None); plt.title("SHAP dependence: PAY_0")
savefig("fig_shap_dependence_PAY_0")

# %%
# ---- Scenario A: leakage experiment (risk_leak included) -----------------------------------------
prepA = build_preprocessor(num_leak, cat_cols)
XA_train, XA_test = X_train[num_leak + cat_cols], X_test[num_leak + cat_cols]
leak_models = {n: make_models()[n] for n in ["Logistic Regression", "Random Forest", "Gradient Boosting"]}
leak_fit = {n: Pipeline([("prep", clone(prepA)), ("clf", m)]).fit(XA_train, y_train) for n, m in leak_models.items()}
leak_res = pd.DataFrame({n: evaluate(p, XA_test, y_test) for n, p in leak_fit.items()}).T
print("Scenario A (WITH risk_leak):"); print(leak_res.round(4).to_string()); savetab(leak_res.round(4), "table13_leakage_experiment")
KEY["leak_results"] = leak_res.round(4).to_dict("index")
grp_A, _, _ = shap_report(leak_fit["Gradient Boosting"], XA_test, "A_with_risk_leak")
tabA = (grp_A / grp_A.sum() * 100).round(2).rename("share of total |SHAP| (%)").to_frame()
print(tabA.head(8).to_string()); savetab(tabA, "table14_shap_importance_with_leak")
KEY["shap_A_top"] = tabA.head(8).to_dict("index")

# %%
# ---- Save everything needed for the report ------------------------------------------------------------
with open(f"{TAB_DIR}/key_results.json", "w") as f:
    json.dump(KEY, f, indent=2, default=str)
print("Done. Figures in ./figures, tables in ./tables")
