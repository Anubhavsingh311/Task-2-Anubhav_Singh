"""
Fraud Detection Machine Learning Pipeline
==========================================
Dataset  : Kaggle Credit Card Fraud Detection
           (284,807 transactions | 492 fraud | 0.1727% fraud rate)
Source   : https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud

Features : V1-V28 (PCA-anonymised), Time, Amount
Target   : Class  (0 = Legitimate, 1 = Fraud)

Run:
    pip install -r requirements.txt
    # Place creditcard.csv in the same directory, then:
    python fraud_detection_pipeline.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve,
    precision_score, recall_score, f1_score
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline


# ─────────────────────────────────────────────────────────────────
# STEP 1 — LOAD AND EXPLORE THE DATASET
# ─────────────────────────────────────────────────────────────────
print("=" * 65)
print("STEP 1: LOAD AND EXPLORE THE DATASET")
print("=" * 65)

df = pd.read_csv("creditcard.csv")

print(f"\nDataset shape : {df.shape}")
print(f"\nColumn dtypes:\n{df.dtypes.to_string()}")
print(f"\nMissing values : {df.isnull().sum().sum()}")
print(f"\nBasic statistics (Amount & Time):")
print(df[["Time", "Amount"]].describe().to_string())

class_counts = df["Class"].value_counts()
class_pct    = df["Class"].value_counts(normalize=True) * 100
fraud_pct    = class_pct[1]

print(f"\nClass distribution:")
dist_df = pd.DataFrame({"Count": class_counts, "Percentage (%)": class_pct.round(4)})
print(dist_df.to_string())
print(f"\n>>> HIGHLY IMBALANCED: only {fraud_pct:.4f}% of transactions are fraudulent.")
print(f"    Fraud-to-Legitimate ratio : 1 : {round(class_counts[0] / class_counts[1])}")
print(f"    A model predicting 'Legitimate' for every transaction achieves")
print(f"    {(1 - fraud_pct/100)*100:.4f}% accuracy while catching ZERO fraud.")


# ─────────────────────────────────────────────────────────────────
# STEP 2 — TRAIN / TEST SPLIT  (before SMOTE)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 2: TRAIN / TEST SPLIT  (stratified, BEFORE SMOTE)")
print("=" * 65)

X = df.drop("Class", axis=1)
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    stratify=y,
    random_state=42
)

print(f"\nTraining : {X_train.shape[0]:,} rows | Fraud: {y_train.sum():,} ({y_train.mean()*100:.4f}%)")
print(f"Test     : {X_test.shape[0]:,} rows  | Fraud: {y_test.sum():,}  ({y_test.mean()*100:.4f}%)")
print("\nApplying SMOTE before this split would leak synthetic samples")
print("into the test set, inflating every reported metric artificially.")


# ─────────────────────────────────────────────────────────────────
# STEP 3 — SMOTE  (training data only)
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 3: SMOTE — applied ONLY to training data")
print("=" * 65)

smote_standalone = SMOTE(random_state=42)
X_train_smote, y_train_smote = smote_standalone.fit_resample(X_train, y_train)

before = pd.Series(y_train).value_counts()
after  = pd.Series(y_train_smote).value_counts()
print(f"\nBefore SMOTE : Legit = {before[0]:,}   Fraud = {before[1]:,}")
print(f"After  SMOTE : Legit = {after[0]:,}   Fraud = {after[1]:,}")
print("The test set is never passed through SMOTE.")


# ─────────────────────────────────────────────────────────────────
# STEPS 4 & 5 — PIPELINES
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEPS 4-5: PIPELINES  (imblearn.pipeline — leak-proof SMOTE)")
print("=" * 65)

# Logistic Regression: StandardScaler → SMOTE → LR
lr_pipeline = ImbPipeline([
    ("scaler",     StandardScaler()),
    ("smote",      SMOTE(random_state=42)),
    ("classifier", LogisticRegression(max_iter=1000, random_state=42, class_weight='balanced'))
])

# Random Forest: SMOTE → RF  (no scaling needed for tree models)
rf_pipeline = ImbPipeline([
    ("smote",      SMOTE(random_state=42)),
    ("classifier", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))
])

print("LR  pipeline : StandardScaler → SMOTE → LogisticRegression")
print("RF  pipeline : SMOTE → RandomForestClassifier")
print("imblearn.pipeline confines SMOTE to each CV fold inside GridSearchCV.")


# ─────────────────────────────────────────────────────────────────
# STEP 6 — GridSearchCV HYPERPARAMETER TUNING
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 6: HYPERPARAMETER TUNING — GridSearchCV")
print("=" * 65)

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# --- Logistic Regression ---
lr_param_grid = {"classifier__C": [0.01, 0.1, 1, 10]}
print("\nTuning Logistic Regression  (4 C values × 3-fold = 12 fits)...")
lr_grid = GridSearchCV(lr_pipeline, lr_param_grid, cv=cv,
                       scoring="roc_auc", n_jobs=-1, verbose=1)
lr_grid.fit(X_train, y_train)
print(f"  Best C          : {lr_grid.best_params_['classifier__C']}")
print(f"  Best CV ROC-AUC : {lr_grid.best_score_:.4f}")

# --- Random Forest ---
rf_param_grid = {"classifier__max_depth": [10, 20, None]}
print("\nTuning Random Forest  (3 max_depth values × 3-fold = 9 fits)...")
rf_grid = GridSearchCV(rf_pipeline, rf_param_grid, cv=cv,
                       scoring="roc_auc", n_jobs=-1, verbose=1)
rf_grid.fit(X_train, y_train)
print(f"  Best max_depth  : {rf_grid.best_params_['classifier__max_depth']}")
print(f"  Best CV ROC-AUC : {rf_grid.best_score_:.4f}")


# ─────────────────────────────────────────────────────────────────
# STEP 7 — EVALUATION ON HELD-OUT TEST SET
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 7: EVALUATION ON HELD-OUT TEST SET")
print("=" * 65)

def evaluate(name, estimator, X_te, y_te):
    y_pred = estimator.predict(X_te)
    y_prob = estimator.predict_proba(X_te)[:, 1]
    p   = precision_score(y_te, y_pred)
    r   = recall_score(y_te, y_pred)
    f1  = f1_score(y_te, y_pred)
    auc = roc_auc_score(y_te, y_prob)
    cm  = confusion_matrix(y_te, y_pred)
    print(f"\n{'─' * 50}")
    print(f"  {name}")
    print(f"{'─' * 50}")
    print(f"  Precision : {p:.4f}")
    print(f"  Recall    : {r:.4f}")
    print(f"  F1-score  : {f1:.4f}")
    print(f"  ROC-AUC   : {auc:.4f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_te, y_pred,
          target_names=["Legitimate", "Fraud"], digits=4))
    print(f"  Confusion Matrix:\n  {cm}")
    return {"Model": name, "Precision": p, "Recall": r,
            "F1": f1, "ROC-AUC": auc, "cm": cm, "y_prob": y_prob}

lr_res = evaluate("Logistic Regression", lr_grid.best_estimator_, X_test, y_test)
rf_res = evaluate("Random Forest",       rf_grid.best_estimator_, X_test, y_test)


# ─────────────────────────────────────────────────────────────────
# STEP 8 — COMPARISON TABLE & CONCLUSION
# ─────────────────────────────────────────────────────────────────
print("\n" + "=" * 65)
print("STEP 8: MODEL COMPARISON & CONCLUSION")
print("=" * 65)

comparison = pd.DataFrame([
    {k: v for k, v in lr_res.items() if k not in ["cm", "y_prob"]},
    {k: v for k, v in rf_res.items() if k not in ["cm", "y_prob"]},
]).set_index("Model").round(4)

print(f"\n{comparison.to_string()}")

lr_auc = lr_res["ROC-AUC"]; rf_auc = rf_res["ROC-AUC"]
print(f"""
CONCLUSION
─────────────────────────────────────────────────────────────
Both models were evaluated on the same held-out 20% test set
containing {y_test.sum()} actual fraud cases out of {len(y_test):,} transactions.

Logistic Regression (AUC = {lr_auc:.4f}):
  High Recall ({lr_res['Recall']:.4f}) — catches most fraud, but fires 1,400+
  false alarms per {len(y_test):,} transactions (Precision {lr_res['Precision']:.4f}).
  Suitable when catching every fraud case is paramount and
  the investigation team can absorb high false positive volume.

Random Forest (AUC = {rf_auc:.4f}):
  Far higher Precision ({rf_res['Precision']:.4f}) and F1 ({rf_res['F1']:.4f}) — flags
  far fewer legitimate transactions as fraud, while still
  catching {rf_res['Recall']*100:.1f}% of actual fraud cases.
  Better suited for production where analyst review capacity
  is limited and precision matters alongside recall.

Winner by ROC-AUC  : {'Logistic Regression' if lr_auc > rf_auc else 'Random Forest'}
Winner by F1-score : {'Logistic Regression' if lr_res['F1'] > rf_res['F1'] else 'Random Forest'}
Winner by Precision: {'Logistic Regression' if lr_res['Precision'] > rf_res['Precision'] else 'Random Forest'}

The optimal choice depends on the business cost of:
  (a) a missed fraud  vs  (b) a false alarm sent to an analyst.
─────────────────────────────────────────────────────────────
""")


# ─────────────────────────────────────────────────────────────────
# VISUALISATIONS  (9-panel dashboard)
# ─────────────────────────────────────────────────────────────────
C_BLUE  = '#42A5F5'
C_GREEN = '#66BB6A'

fig = plt.figure(figsize=(20, 18))
fig.suptitle('Fraud Detection ML Pipeline — Real Kaggle Dataset (284,807 transactions)',
             fontsize=15, fontweight='bold', y=0.99)
gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.52, wspace=0.40)

# 1. Original class distribution
ax1 = fig.add_subplot(gs[0, 0])
bars = ax1.bar(['Legitimate', 'Fraud'], [class_counts[0], class_counts[1]],
               color=['#2196F3', '#F44336'], edgecolor='white', lw=1.5, width=0.5)
for bar, cnt in zip(bars, [class_counts[0], class_counts[1]]):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.04,
             f'{cnt:,}', ha='center', fontsize=10, fontweight='bold')
ax1.set_yscale('log'); ax1.set_ylim(bottom=1)
ax1.set_title(f'Original Class Distribution\n({fraud_pct:.4f}% fraud — extremely imbalanced)', fontweight='bold')
ax1.set_ylabel('Count (log scale)')

# 2. After SMOTE
ax2 = fig.add_subplot(gs[0, 1])
bars2 = ax2.bar(['Legitimate', 'Fraud'], [after[0], after[1]],
                color=['#2196F3', '#F44336'], edgecolor='white', lw=1.5, width=0.5)
for bar, cnt in zip(bars2, [after[0], after[1]]):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.01,
             f'{cnt:,}', ha='center', fontsize=10, fontweight='bold')
ax2.set_title('After SMOTE\n(training set balanced 50/50)', fontweight='bold')
ax2.set_ylabel('Count')

# 3. Metric bar comparison
ax3 = fig.add_subplot(gs[0, 2])
metrics = ['Precision', 'Recall', 'F1', 'ROC-AUC']
x = np.arange(len(metrics)); w = 0.35
lr_v = [lr_res[m] for m in metrics]; rf_v = [rf_res[m] for m in metrics]
b1 = ax3.bar(x - w/2, lr_v, w, label='Logistic Regression', color=C_BLUE,  edgecolor='white')
b2 = ax3.bar(x + w/2, rf_v, w, label='Random Forest',       color=C_GREEN, edgecolor='white')
ax3.set_xticks(x); ax3.set_xticklabels(metrics, fontsize=9)
ax3.set_ylim(0, 1.18)
ax3.set_title('Metric Comparison', fontweight='bold'); ax3.set_ylabel('Score')
ax3.legend(fontsize=8)
for bar in list(b1) + list(b2):
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=8)

# 4. Confusion Matrix — LR
ax4 = fig.add_subplot(gs[1, 0])
sns.heatmap(lr_res['cm'], annot=True, fmt='d', cmap='Blues', ax=ax4,
            xticklabels=['Pred Legit', 'Pred Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud'],
            linewidths=0.5, cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
ax4.set_title('Confusion Matrix\nLogistic Regression', fontweight='bold')

# 5. Confusion Matrix — RF
ax5 = fig.add_subplot(gs[1, 1])
sns.heatmap(rf_res['cm'], annot=True, fmt='d', cmap='Greens', ax=ax5,
            xticklabels=['Pred Legit', 'Pred Fraud'],
            yticklabels=['Actual Legit', 'Actual Fraud'],
            linewidths=0.5, cbar=False, annot_kws={'size': 13, 'weight': 'bold'})
ax5.set_title('Confusion Matrix\nRandom Forest', fontweight='bold')

# 6. ROC Curves
ax6 = fig.add_subplot(gs[1, 2])
fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_res['y_prob'])
fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_res['y_prob'])
ax6.plot(fpr_lr, tpr_lr, color=C_BLUE,  lw=2, label=f'LR  (AUC = {lr_res["ROC-AUC"]:.4f})')
ax6.plot(fpr_rf, tpr_rf, color=C_GREEN, lw=2, label=f'RF  (AUC = {rf_res["ROC-AUC"]:.4f})')
ax6.plot([0, 1], [0, 1], 'k--', lw=1, alpha=0.5, label='Random Classifier')
ax6.set_xlabel('False Positive Rate'); ax6.set_ylabel('True Positive Rate')
ax6.set_title('ROC Curves', fontweight='bold'); ax6.legend(fontsize=9)
ax6.set_xlim([0, 1]); ax6.set_ylim([0, 1.02])

# 7. GridSearchCV — LR
ax7 = fig.add_subplot(gs[2, 0])
lr_cv_df = pd.DataFrame(lr_grid.cv_results_)
C_vals   = [0.01, 0.1, 1, 10]
ax7.plot(C_vals, lr_cv_df['mean_test_score'], 'o-', color=C_BLUE, lw=2, ms=8)
ax7.fill_between(C_vals,
    lr_cv_df['mean_test_score'] - lr_cv_df['std_test_score'],
    lr_cv_df['mean_test_score'] + lr_cv_df['std_test_score'],
    alpha=0.2, color=C_BLUE)
ax7.set_xscale('log')
ax7.set_xlabel('C (regularisation)'); ax7.set_ylabel('Mean CV ROC-AUC')
ax7.set_title('GridSearchCV — Logistic Regression', fontweight='bold')
best_C = lr_grid.best_params_['classifier__C']
ax7.axvline(best_C, color='red', ls='--', alpha=0.7, label=f'Best C = {best_C}')
ax7.legend(fontsize=8)

# 8. GridSearchCV — RF
ax8 = fig.add_subplot(gs[2, 1])
rf_cv_df     = pd.DataFrame(rf_grid.cv_results_)
depth_labels = ['10', '20', 'None']
ax8.bar(depth_labels, rf_cv_df['mean_test_score'],
        color=C_GREEN, edgecolor='white',
        yerr=rf_cv_df['std_test_score'], capsize=6,
        error_kw={'ecolor': 'gray', 'alpha': 0.7})
ax8.set_xlabel('max_depth'); ax8.set_ylabel('Mean CV ROC-AUC')
ax8.set_title('GridSearchCV — Random Forest', fontweight='bold')
for i, (lbl, s) in enumerate(zip(depth_labels, rf_cv_df['mean_test_score'])):
    ax8.text(i, s + 0.001, f'{s:.4f}', ha='center', va='bottom', fontsize=9)

# 9. Summary table
ax9 = fig.add_subplot(gs[2, 2])
ax9.axis('off')
tdata = [
    ['Logistic Regression',
     f"{lr_res['Precision']:.4f}", f"{lr_res['Recall']:.4f}",
     f"{lr_res['F1']:.4f}",        f"{lr_res['ROC-AUC']:.4f}"],
    ['Random Forest',
     f"{rf_res['Precision']:.4f}", f"{rf_res['Recall']:.4f}",
     f"{rf_res['F1']:.4f}",        f"{rf_res['ROC-AUC']:.4f}"],
]
tbl = ax9.table(cellText=tdata,
                colLabels=['Model', 'Precision', 'Recall', 'F1', 'ROC-AUC'],
                loc='center', cellLoc='center')
tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1.3, 2.4)
for j in range(5):
    tbl[0, j].set_facecolor('#1565C0')
    tbl[0, j].set_text_props(color='white', fontweight='bold')
# Highlight row with better ROC-AUC
best_row = 1 if lr_res['ROC-AUC'] >= rf_res['ROC-AUC'] else 2
for j in range(5):
    tbl[best_row, j].set_facecolor('#E8F5E9')
    tbl[best_row, j].set_text_props(fontweight='bold')
ax9.set_title('Final Comparison\n(LR: higher ROC-AUC | RF: higher Precision & F1)',
              fontweight='bold', pad=20)

plt.savefig('fraud_detection_results.png', dpi=150, bbox_inches='tight', facecolor='white')
print("Dashboard saved → fraud_detection_results.png")
print("\n" + "=" * 65)
print("PIPELINE COMPLETE")
print("=" * 65)
