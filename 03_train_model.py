#!/usr/bin/env python3

import numpy as np
import pandas as pd
import joblib
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, GroupKFold
from sklearn.metrics import precision_recall_curve, average_precision_score
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, roc_curve, roc_auc_score


#############################################
### Variables
#############################################
BORZOI_CADD_GPN_MSA = "/workspace/preprocessed_data/train/borzoi_with_cadd_gpn_msa.txt"
TG_DATA = "/workspace/data/TraitGym/TG.txt"
TRAIN_CLF_DATA = "/workspace/preprocessed_data/train/borzoi_with_cadd_gpn_msa_clf.txt"


MODEL = "/workspace/results/final_model.joblib"
PRAUC_CHROM = "/workspace/results/prauc_per_chrom.csv"
MODEL_COEF = "/workspace/results/final_model_mean_coefficients.csv"

PLOT_F1 = "/workspace/results/f1_threshold_curve.png"
PLOT_DISTR = "/workspace/results/probs_distribution.png"
PLOT_ROC = "/workspace/results/roc_curve.png"
PLOT_PR = "/workspace/results/pr_curve.png"



#############################################
### Ensemble_model(CADD+GPN-MSA+Borzoi)
#############################################

def predict(clf, X):
    return clf.predict_proba(X)[:, 1]


def train_predict(V_train, V_test, features, train_f):
    clf = train_f(V_train[features], V_train.label, V_train.chrom)
    return predict(clf, V_test[features])


def train_logistic_regression(X, y, groups):

    pipeline = Pipeline([
        ('imputer', SimpleImputer(
            missing_values=np.nan,
            strategy='mean',
            keep_empty_features=True,  
        )),
        ('scaler', StandardScaler()),
        ('linear', LogisticRegression(
            class_weight="balanced",
            random_state=42
        ))
    ])


    Cs = np.logspace(-8, 0, 10)
    param_grid = {'linear__C': Cs}

    clf = GridSearchCV(
        pipeline,
        param_grid,
        scoring="average_precision",
        cv=GroupKFold(),
        n_jobs=-1,
    )


    clf.fit(X, y, groups=groups)


    print(f"Лучшие параметры: {clf.best_params_}")


    linear = clf.best_estimator_.named_steps["linear"]
    coef = pd.DataFrame({
        "feature": X.columns,
        "coef": linear.coef_[0],
    }).sort_values("coef", ascending=False, key=abs)

    print("Коэффициенты:")
    print(coef.head(10))

    return clf




#############################################
###  Step 1: Add label to training dataset 
#############################################

print("=== Step 1: Add label to training dataset ===")
# df_train = pd.read_csv(BORZOI_CADD_GPN_MSA, sep='\t')
# TG = pd.read_csv(TG_DATA, sep='\t')
# TG['hash'] = TG.iloc[:,:4].astype(str).agg("_".join, axis=1)
# df_train.merge(TG[['hash', 'label']], on='hash', how='left').to_csv(TRAIN_CLF_DATA, sep='\t', index=False)


#############################################
###  Step 2: Train model
#############################################


print("=== Step 2: Training model ===")


# =============================
# 2.1. Загружаем данные
# =============================
df = pd.read_csv(TRAIN_CLF_DATA, sep="\t")
features = [c for c in df.columns if c not in ["hash", "chrom", "label"]]
X = df[features]
y = df["label"]
groups = df["chrom"]

best_cs = []
coef_list = []
intercept_list = []

results = []  

# =============================
# 2.2. LOCO (по хромосомам)
# =============================
for chrom in df["chrom"].unique():
    train_mask = df["chrom"] != chrom
    test_mask = ~train_mask

    clf = train_logistic_regression(
        X=X.loc[train_mask],
        y=y.loc[train_mask],
        groups=groups.loc[train_mask],
    )


    best_cs.append(clf.best_params_["linear__C"])


    linear = clf.best_estimator_.named_steps["linear"]
    coef_list.append(linear.coef_[0])
    intercept_list.append(linear.intercept_[0])


    probs = predict(clf, X.loc[test_mask])
    prauc = average_precision_score(y.loc[test_mask], probs)

    results.append({"chrom": chrom, "n": test_mask.sum(), "prauc": prauc})
    print(f"Chrom {chrom}: AUPRC={prauc:.4f}, best_C={best_cs[-1]}")

# =============================
# 2.3. Считаем взвешенное среднее AUPRC + SE
# =============================
res = pd.DataFrame(results)

weights = res["n"] / res["n"].sum()
weighted_mean = (res["prauc"] * weights).sum()

bootstraps = []
for i in range(1000):
    sampled = res.sample(len(res), replace=True, random_state=i)
    w = sampled["n"] / sampled["n"].sum()
    bootstraps.append((sampled["prauc"] * w).sum())
se = np.std(bootstraps)

print("\nAUPRC по каждой хромосоме:")
print(res)
print(f"\nВзвешенное среднее AUPRC: {weighted_mean:.4f} ± {se:.4f}")

res.to_csv(PRAUC_CHROM, index=False)


# =============================
# 2.4. Финальная модель
# =============================


final_C = float(np.mean(best_cs))
mean_coef = np.mean(np.vstack(coef_list), axis=0).reshape(1, -1)
mean_intercept = float(np.mean(intercept_list))

print(f"\nФинальный C: {final_C:.2e}")
print("Пример средний коэффициентов:")
print(pd.Series(mean_coef[0], index=features).sort_values(key=abs, ascending=False).head(10))


final_model = LogisticRegression(
    class_weight="balanced",
    random_state=42,
    C=final_C
)


final_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean', keep_empty_features=True)),
    ('scaler', StandardScaler()),
    ('linear', final_model)
])

final_pipeline.fit(X, y)

final_model = final_pipeline.named_steps["linear"]
final_model.classes_ = np.array([0, 1])
final_model.coef_ = mean_coef
final_model.intercept_ = np.array([mean_intercept])


joblib.dump(final_pipeline, MODEL)
print("✔ Сохранено: final_model.joblib")


coef_df = pd.DataFrame({
    "feature": features,
    "coef": mean_coef[0]
}).sort_values("coef", ascending=False, key=abs)
coef_df.to_csv(MODEL_COEF, index=False)
print("✔ Сохранено: final_model_mean_coefficients.csv")



#############################################
###  Step 3: Visualisation
#############################################

print("=== Step 3: Visualisation ===")


# =============================
# 3.1. Apply model to train dataset 
# =============================

# Загружаем данные и модель
df = pd.read_csv(TRAIN_CLF_DATA, sep="\t")
features = [c for c in df.columns if c not in ["hash", "chrom", "label"]]
final_model = joblib.load(MODEL)


probs = final_model.predict_proba(df[features])[:, 1]

# =============================
# 3.2. F1 curve 
# =============================

thresholds = np.linspace(0, 1, 101)
f1_scores = [f1_score(y, probs >= t) for t in thresholds]

best_t = thresholds[np.argmax(f1_scores)]
print(f"Лучший порог по F1: {best_t:.2f}, F1 = {max(f1_scores):.3f}")

plt.figure(figsize=(8, 5))
plt.plot(thresholds, f1_scores, marker="o", markersize=3, linewidth=1.8)  
plt.axvline(best_t, color="red", linestyle="--", label=f"Лучший порог = {best_t:.2f}")
plt.xlabel("Порог вероятности")
plt.ylabel("F1-score")
plt.legend()
plt.grid(alpha=0.3)

plt.savefig(PLOT_F1, dpi=300, bbox_inches="tight")
plt.close()

print("✅ График сохранён: f1_threshold_curve.png")


# =============================
# 3.3. Probability Distribution 
# =============================


plt.figure(figsize=(8, 5))
sns.kdeplot(probs[df["label"] == 0], fill=True, alpha=0.5, label="Класс 0 (Контроль)")
sns.kdeplot(probs[df["label"] == 1], fill=True, alpha=0.5, label="Класс 1 (Каузальные)")
plt.axvline(best_t, color="red", linestyle="--", label=f"Порог = {best_t:.2f}")
plt.xlabel("Вероятность класса 1")
plt.ylabel("Плотность вероятности")
plt.legend()
plt.savefig(PLOT_DISTR, dpi=300, bbox_inches="tight")
plt.close()

print("✅ График сохранён: probs_distribution.png")

# =============================
# 3.4. ROC curve
# =============================
probs = final_model.predict_proba(df[features])[:, 1]

# ROC
fpr, tpr, _ = roc_curve(df["label"], probs)
roc_auc = roc_auc_score(df["label"], probs)
print(f"ROC AUC = {roc_auc:.4f}")

plt.figure(figsize=(6, 6))
plt.plot(fpr, tpr, color="#1f77b4", linewidth=2, label=f"ROC curve (AUC = {roc_auc:.3f})")
plt.plot([0, 1], [0, 1], color="#d62728", linestyle="--", linewidth=1.2, label="Случайный классификатор")

plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC-кривая")
plt.legend(frameon=False)


plt.grid(True, color="gray", linestyle="--", linewidth=0.5, alpha=0.3)

plt.savefig(PLOT_ROC, dpi=300, bbox_inches="tight")
plt.close()

print("✅ График сохранён: roc_curve.png")



# =============================
# 3.5. PR curve
# =============================

precision, recall, _ = precision_recall_curve(df["label"], probs)
prauc = average_precision_score(df["label"], probs)
print(f"PR AUC = {prauc:.4f}")

plt.figure(figsize=(6, 6))
plt.plot(recall, precision, label=f"PR curve (AUC = {prauc:.3f})", color="blue")
plt.xlabel("Recall (полнота)")
plt.ylabel("Precision (точность)")
plt.title("Precision-Recall кривая")
plt.legend()
plt.grid(True)
plt.savefig(PLOT_PR, dpi=300, bbox_inches="tight")
plt.close()

print("✅ График сохранён: pr_curve.png")

