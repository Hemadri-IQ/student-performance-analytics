"""
Pipeline prototype and metrics extractor for:
AI-Powered Student Performance Analytics & Early Risk Prediction System
Author: Hemadri
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)
from sklearn.inspection import permutation_importance

# Ensure figures folder exists
os.makedirs("report/figures", exist_ok=True)

# Styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Arial'
plt.rcParams['axes.edgecolor'] = '#CCCCCC'
plt.rcParams['axes.linewidth'] = 0.8

print("1. Loading raw dataset...")
df_raw = pd.read_csv("data/student_performance.csv")
raw_rows, raw_cols = df_raw.shape
raw_duplicates = int(df_raw.duplicated().sum())
raw_missing = int(df_raw.isnull().sum().sum())
missing_by_col = df_raw.isnull().sum()[df_raw.isnull().sum() > 0].to_dict()

print(f"Raw shape: {raw_rows} rows, {raw_cols} columns")
print(f"Duplicates: {raw_duplicates}")
print(f"Missing values: {raw_missing}")

# 2. Data Cleaning
print("2. Performing data cleaning...")
df_clean = df_raw.copy()

# Deduplicate
df_clean = df_clean.drop_duplicates().reset_index(drop=True)
dedup_rows = len(df_clean)

# Impute missing values for clean analytical dataframe
# Numericals: median
num_cols_with_missing = ["study_hours_per_day", "sleep_hours", "previous_score"]
for c in num_cols_with_missing:
    median_val = df_clean[c].median()
    df_clean[c] = df_clean[c].fillna(median_val)

# Categoricals: mode
cat_cols_with_missing = ["parental_education", "extracurricular_activity"]
for c in cat_cols_with_missing:
    mode_val = df_clean[c].mode()[0]
    df_clean[c] = df_clean[c].fillna(mode_val)

clean_rows, clean_cols = df_clean.shape
clean_duplicates = int(df_clean.duplicated().sum())
clean_missing = int(df_clean.isnull().sum().sum())

print(f"Clean shape: {clean_rows} rows, {clean_cols} columns")
print(f"Clean duplicates: {clean_duplicates}, Clean missing: {clean_missing}")

# 3. KPI Computation
print("3. Computing KPIs...")
kpis = {
    "avg_final_score": float(df_clean["final_score"].mean()),
    "avg_attendance": float(df_clean["attendance_percentage"].mean()),
    "avg_study_hours": float(df_clean["study_hours_per_day"].mean()),
    "high_performing_pct": float((df_clean["final_score"] >= 80.0).mean() * 100),
    "high_risk_pct": float((df_clean["academic_risk"] == "High Risk").mean() * 100),
    "avg_assignment_score": float(df_clean["assignment_score"].mean()),
    "avg_midterm_score": float(df_clean["midterm_score"].mean()),
    "avg_previous_score": float(df_clean["previous_score"].mean())
}

for k, v in kpis.items():
    print(f"  {k}: {v:.2f}")

# Risk distribution counts and percentages
risk_counts = df_clean["academic_risk"].value_counts().to_dict()
risk_pcts = (df_clean["academic_risk"].value_counts(normalize=True) * 100).to_dict()
print("Risk distribution:", risk_counts)
print("Risk percentages:", {k: f"{v:.1f}%" for k, v in risk_pcts.items()})

# 4. EDA Visualizations
print("4. Generating EDA visualizations...")

# Figure 1: Score Distributions
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
sns.histplot(df_clean["final_score"], kde=True, color="#1f77b4", ax=axes[0], bins=25)
axes[0].set_title("Distribution of Final Academic Scores", fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel("Final Score (0-100)", fontsize=10)
axes[0].set_ylabel("Student Count", fontsize=10)
axes[0].axvline(df_clean["final_score"].mean(), color='red', linestyle='--', label=f'Mean: {kpis["avg_final_score"]:.1f}')
axes[0].legend()

sns.histplot(df_clean["attendance_percentage"], kde=True, color="#2ca02c", ax=axes[1], bins=25)
axes[1].set_title("Distribution of Student Attendance Rate", fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel("Attendance Percentage (%)", fontsize=10)
axes[1].set_ylabel("Student Count", fontsize=10)
axes[1].axvline(df_clean["attendance_percentage"].mean(), color='red', linestyle='--', label=f'Mean: {kpis["avg_attendance"]:.1f}%')
axes[1].legend()
plt.tight_layout()
plt.savefig("report/figures/fig1_distributions.png", dpi=300)
plt.close()

# Figure 2: Key Correlations with Final Score
fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
sns.regplot(data=df_clean, x="attendance_percentage", y="final_score",
            scatter_kws={'alpha': 0.35, 'color': '#2ca02c'}, line_kws={'color': '#d62728', 'linewidth': 2}, ax=axes[0])
axes[0].set_title("Attendance vs. Final Score", fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel("Attendance Percentage (%)", fontsize=10)
axes[0].set_ylabel("Final Score", fontsize=10)

sns.regplot(data=df_clean, x="study_hours_per_day", y="final_score",
            scatter_kws={'alpha': 0.35, 'color': '#1f77b4'}, line_kws={'color': '#d62728', 'linewidth': 2}, ax=axes[1])
axes[1].set_title("Daily Study Hours vs. Final Score", fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel("Study Hours / Day", fontsize=10)
axes[1].set_ylabel("Final Score", fontsize=10)
plt.tight_layout()
plt.savefig("report/figures/fig2_bivariate_relationships.png", dpi=300)
plt.close()

# Figure 3: Correlation Heatmap
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
corr_matrix = df_clean[numeric_cols].corr()

plt.figure(figsize=(9, 7))
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="Blues", cbar=True, square=True, linewidths=0.5)
plt.title("Correlation Heatmap of Academic & Behavioral Features", fontsize=12, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right', fontsize=9)
plt.yticks(fontsize=9)
plt.tight_layout()
plt.savefig("report/figures/fig3_correlation_matrix.png", dpi=300)
plt.close()

# Figure 4: Academic Risk Distribution
plt.figure(figsize=(7, 4.5))
order = ["Low Risk", "Moderate Risk", "High Risk"]
palette = {"Low Risk": "#2ca02c", "Moderate Risk": "#ff7f0e", "High Risk": "#d62728"}
ax = sns.countplot(data=df_clean, x="academic_risk", order=order, palette=palette)
plt.title("Academic Risk Category Distribution (N = 2,023)", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Academic Risk Category", fontsize=10)
plt.ylabel("Number of Students", fontsize=10)
for p in ax.patches:
    h = p.get_height()
    pct = (h / len(df_clean)) * 100
    ax.annotate(f"{h:,}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., h / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=10)
plt.tight_layout()
plt.savefig("report/figures/fig4_risk_distribution.png", dpi=300)
plt.close()

# 5. Machine Learning Modeling
print("5. Running Machine Learning Modeling...")
# Features and Target
X = df_clean.drop(columns=["student_id", "final_score", "academic_risk"])
y = df_clean["academic_risk"]

feature_names_original = list(X.columns)
print(f"Features ({len(feature_names_original)}): {feature_names_original}")

# Train/Test Split (Stratified, 80/20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical_features),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ]), categorical_features)
    ]
)

models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
}

results = {}
trained_pipelines = {}

classes = ["High Risk", "Moderate Risk", "Low Risk"]

for name, clf in models.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    
    results[name] = {
        "Accuracy": float(acc),
        "Precision": float(prec),
        "Recall": float(rec),
        "F1-Score": float(f1),
        "ROC-AUC": float(roc_auc)
    }
    print(f"\n--- {name} ---")
    print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

# Model comparison table
df_metrics = pd.DataFrame(results).T
print("\nModel Comparison Table:")
print(df_metrics.round(4))

# Select best model based on F1-Score & ROC-AUC
best_model_name = df_metrics["F1-Score"].idxmax()
best_pipe = trained_pipelines[best_model_name]
print(f"\nSelected Champion Model: {best_model_name}")

y_pred_best = best_pipe.predict(X_test)
cm = confusion_matrix(y_test, y_pred_best, labels=classes)
cm_norm = confusion_matrix(y_test, y_pred_best, labels=classes, normalize="true")

# Figure 5: Confusion Matrix
plt.figure(figsize=(6.5, 5))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes,
            cbar=False, annot_kws={"size": 11, "fontweight": "bold"})
plt.title(f"Confusion Matrix: {best_model_name} (Test Set)", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Predicted Academic Risk Category", fontsize=10)
plt.ylabel("Actual Academic Risk Category", fontsize=10)
plt.tight_layout()
plt.savefig("report/figures/fig5_confusion_matrix.png", dpi=300)
plt.close()

# Feature Importance
print("6. Extracting Feature Importance...")
# Get feature names after one-hot encoding
cat_encoder = best_pipe.named_steps["preprocessor"].named_transformers_["cat"].named_steps["encoder"]
encoded_cat_names = list(cat_encoder.get_feature_names_out(categorical_features))
all_feature_names = numerical_features + encoded_cat_names

# Permutation importance
perm_imp = permutation_importance(best_pipe, X_test, y_test, n_repeats=10, random_state=42, scoring="f1_weighted")
perm_importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Importance": perm_imp.importances_mean,
    "Std": perm_imp.importances_std
}).sort_values(by="Importance", ascending=False)

print("\nPermutation Feature Importance (Top 10):")
print(perm_importance_df.head(10).to_string(index=False))

# Figure 6: Feature Importance Plot
plt.figure(figsize=(8, 5))
sns.barplot(data=perm_importance_df.head(10), x="Importance", y="Feature", palette="Blues_r")
plt.title(f"Permutation Feature Importance ({best_model_name})", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Mean Decrease in Weighted F1-Score", fontsize=10)
plt.ylabel("Academic & Behavioral Features", fontsize=10)
plt.tight_layout()
plt.savefig("report/figures/fig6_feature_importance.png", dpi=300)
plt.close()

# Save computed results to json for report generator
summary_stats = {
    "raw_rows": raw_rows,
    "raw_cols": raw_cols,
    "raw_duplicates": raw_duplicates,
    "raw_missing": raw_missing,
    "missing_by_col": missing_by_col,
    "clean_rows": clean_rows,
    "clean_cols": clean_cols,
    "clean_duplicates": clean_duplicates,
    "clean_missing": clean_missing,
    "kpis": kpis,
    "risk_counts": risk_counts,
    "risk_pcts": risk_pcts,
    "metrics": {k: {m: round(v, 4) for m, v in vals.items()} for k, vals in results.items()},
    "best_model": best_model_name,
    "confusion_matrix": cm.tolist(),
    "confusion_matrix_classes": classes,
    "top_features": perm_importance_df.head(10).to_dict(orient="records")
}

with open("report/metrics_summary.json", "w") as f:
    json.dump(summary_stats, f, indent=2)

print("\nMetrics summary saved to report/metrics_summary.json")
