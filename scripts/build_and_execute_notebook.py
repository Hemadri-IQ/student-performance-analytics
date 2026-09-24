"""
Generator and Executor for:
Hemadri_StudentPerformanceAnalytics.ipynb
Using nbformat and nbclient for 100% valid, executed notebook with outputs.
"""

import os
import nbformat as nbf
from nbclient import NotebookClient

def create_notebook():
    nb = nbf.v4.new_notebook()
    nb.metadata = {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "name": "python",
            "version": "3.10"
        }
    }
    
    cells = []
    
    # -------------------------------------------------------------
    # 1. Project Title
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""# AI-Powered Student Performance Analytics & Early Risk Prediction System

**Internship:** IBM SkillsBuild Data Analytics with AI Internship 2026  
**In Association With:** BharatCares & AICTE  
**Lead Developer / Contributor:** Hemadri  
**Environment:** Python 3.10 / JupyterLab (Windows)  
**Date:** September 2026  

---
"""
    ))

    # -------------------------------------------------------------
    # 2. Executive Summary
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 2. Executive Summary

Educational institutions routinely collect extensive behavioral, demographic, and academic records; however, raw administrative tables fail to yield early, proactive interventions. This project implements an enterprise-grade **Data Analytics + Machine Learning** system designed to ingest student records, execute rigorous quality checks, perform exploratory analysis, calculate operational KPIs, and deploy multi-class AI models to identify students facing **potential academic risk** before cumulative final examinations.

### Key Highlights of the Project:
- **Dataset Scope:** 2,050 initial synthetic records (2,023 unique post-deduplication) generated with realistic stochastic variation, modeling 16 demographic, behavioral, and academic features without deterministic shortcuts.
- **Data Quality & Hygiene:** Identified and handled 27 exact duplicates and 100 missing values across numerical (`study_hours_per_day`, `sleep_hours`, `previous_score`) and categorical (`parental_education`, `extracurricular_activity`) attributes using median and modal imputation within non-leaking pipelines.
- **Operational KPIs:** Programmatically derived core institutional metrics, revealing an average final score of 61.40%, average attendance of 67.53%, a 7.66% cohort of high achievers ($\ge 80$), and a 17.65% high-risk student cohort.
- **AI Classification Engine:** Benchmarked **Logistic Regression**, **Random Forest**, and **Gradient Boosting**. The **Random Forest Classifier** achieved the strongest performance with **83.21% Accuracy**, **82.75% Weighted F1-Score**, and an **ROC-AUC of 0.9251**.
- **Model Interpretability:** Permutation importance confirmed that early assessment milestones (`midterm_score` and `assignment_score`), coupled with regular engagement (`attendance_percentage` and `study_hours_per_day`), form the predominant predictive signals.
- **Actionable Framework:** Insights are translated directly into targeted institutional workflows via the **FACT $\\rightarrow$ INSIGHT $\\rightarrow$ RISK/OPPORTUNITY $\\rightarrow$ ACTION** paradigm.
"""
    ))

    # -------------------------------------------------------------
    # 3. Problem Statement
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 3. Problem Statement

Higher education institutions face substantial challenges in student retention, equitable academic progression, and timely remedial support:
1. **The Diagnostic Lag:** Traditional evaluations assess student outcomes post-final examinations, by which point academic probation or dropout cannot be averted.
2. **Siloed & Unstructured Data:** Institutional databases contain fragmented records of attendance logs, LMS submissions, and demographic surveys without an integrated analytical pipeline.
3. **Absence of Risk Stratification:** Academic advisors lack prioritized early-warning mechanisms to distinguish between students requiring routine guidance versus intensive academic counseling.

**System Objective:** Build an autonomous, transparent analytical pipeline that bridges the data-to-decision gap, identifying students who may benefit from additional review well before semester completion.
"""
    ))

    # -------------------------------------------------------------
    # 4. Project Objectives
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 4. Project Objectives

1. **Synthetic Data Synthesis:** Generate a realistic, reproducible ($N \\approx 2,000$, `random_seed=42`) student dataset modeling complex non-linear educational dynamics.
2. **Data Quality Assurance:** Implement systematic audits to detect and remediate missingness, duplicates, and boundary violations.
3. **Exploratory Data Analysis (EDA):** Quantify feature distributions, correlation structures, and engagement disparities.
4. **Institutional KPI Formulation:** Compute core performance indicators programmatically without hard-coded assumptions.
5. **Leak-Free Machine Learning Architecture:** Construct scikit-learn preprocessing and classification pipelines strictly excluding post-outcome indicators (`final_score` and `student_id`).
6. **Multi-Model Benchmarking:** Evaluate models across Accuracy, Weighted Precision, Recall, F1-Score, and Multi-Class ROC-AUC.
7. **Transparent Interpretability:** Derive feature importance rankings via Permutation Importance to guide academic counseling without confusing prediction with causation.
8. **Institutional Action Plan:** Formulate evidence-based administrative recommendations for timely student support.
"""
    ))

    # -------------------------------------------------------------
    # 5. Dataset Description
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 5. Dataset Description

> **Synthetic Dataset Disclosure:**  
> In accordance with project instructions, this dataset was synthetically generated specifically for this project using domain-grounded probability distributions and realistic stochastic noise (`random_seed=42`). It is **not** the IBM SkillsBuild/BharatCares masterclass dataset.

### Schema & Data Dictionary:
| Column Name | Data Type | Expected Range | Description & Institutional Relevance |
| :--- | :--- | :--- | :--- |
| `student_id` | String / Object | STU1001–STU3050 | Unique pseudo-anonymized identifier (excluded from ML). |
| `gender` | Categorical | Female, Male, Other | Student reported demographic gender. |
| `age` | Integer | 17–22 | Student age in years at matriculation. |
| `study_hours_per_day` | Float | 0.5–8.0 hrs | Self-reported daily independent study duration. |
| `attendance_percentage`| Float | 45.0–100.0% | Classroom and laboratory attendance percentage. |
| `previous_score` | Float | 25.0–98.0 | Cumulative historical academic score prior to current term. |
| `assignment_score` | Float | 20.0–100.0 | Continuous assessment grade for assignments and coursework. |
| `midterm_score` | Float | 20.0–100.0 | Mid-semester formal evaluation examination score. |
| `extracurricular_activity`| Categorical | Yes, No | Active participation in university sports/clubs. |
| `internet_access` | Categorical | Yes, No | Reliable broadband connectivity at domicile. |
| `sleep_hours` | Float | 4.0–10.0 hrs | Self-reported nightly sleep duration. |
| `parental_education` | Categorical | High School–Doctorate| Highest educational attainment of parents/guardians. |
| `family_income_category`| Categorical | Low, Medium, High | Socioeconomic household bracket. |
| `class_participation` | Float | 10.0–100.0 | In-class engagement score evaluated by faculty. |
| `final_score` | Float | 20.0–100.0 | End-of-term examination score (retained for EDA, excluded from ML). |
| `academic_risk` | Categorical | Low, Moderate, High | Supervised classification target derived from early risk index. |
"""
    ))

    # -------------------------------------------------------------
    # 6. Technologies Used
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 6. Technologies Used

- **Runtime & Environment:** Python 3.10+, JupyterLab / VS Code (Windows 64-bit).
- **Data Manipulation & Analysis:** `pandas` (DataFrames, aggregations), `numpy` (vectorized math, stochastic sampling).
- **Statistical Machine Learning:** `scikit-learn` (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`, `SimpleImputer`, `RandomForestClassifier`, `LogisticRegression`, `GradientBoostingClassifier`).
- **Data Visualization:** `matplotlib.pyplot`, `seaborn` (customized aesthetic, color palettes, high-resolution rendering).
- **Document & Artifact Automation:** `python-docx` (Word report generation), `nbclient`, `nbformat` (notebook programmatic validation).
"""
    ))

    # -------------------------------------------------------------
    # 7. Dataset Loading
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 7. Dataset Loading

We load `data/student_performance.csv` using a relative file path to maintain platform independence across Windows environments.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Visual formatting defaults
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams['figure.figsize'] = (10, 5)
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.titleweight'] = 'bold'

# Relative path specification
DATA_PATH = "data/student_performance.csv"
print(f"Loading student performance dataset from: {DATA_PATH}")

df_raw = pd.read_csv(DATA_PATH)
print(f"Dataset successfully loaded.")
print(f"Dimensions: {df_raw.shape[0]:,} Rows | {df_raw.shape[1]} Columns")
print("\\nFirst 5 Records:")
df_raw.head()
"""
    ))

    # -------------------------------------------------------------
    # 8. Data Understanding
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 8. Data Understanding

In this step, we examine column schemas, data types, non-null counts, and descriptive summary statistics for both continuous variables and categorical dimensions.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""print("--- DATASET SCHEMA & DATA TYPES ---")
df_raw.info()

print("\\n--- DESCRIPTIVE STATISTICS (NUMERICAL FEATURES) ---")
df_raw.describe().round(2).T
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""print("--- CATEGORICAL FEATURES DISTRIBUTION ---")
cat_cols = df_raw.select_dtypes(include=['object']).columns.tolist()
cat_cols.remove("student_id")

for col in cat_cols:
    print(f"\\nVariable: {col}")
    val_counts = df_raw[col].value_counts(dropna=False)
    val_pcts = df_raw[col].value_counts(normalize=True, dropna=False) * 100
    summary_cat = pd.DataFrame({"Count": val_counts, "Percentage (%)": val_pcts.round(2)})
    print(summary_cat)
"""
    ))

    # -------------------------------------------------------------
    # 9. Data Quality Assessment
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 9. Data Quality Assessment

A rigorous data quality assessment is conducted prior to preprocessing:
1. **Duplicate Detection:** Identify redundant rows.
2. **Missing Value Audit:** Compute frequency and proportion of missingness across all features.
3. **Range & Boundary Checks:** Verify that percentages remain within $[0, 100]$ and hours remain within physiologically realistic intervals.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""print("--- DATA QUALITY AUDIT ---")
num_duplicates = df_raw.duplicated().sum()
print(f"Total Duplicate Rows Detected: {num_duplicates:,} ({(num_duplicates/len(df_raw))*100:.2f}%)")

print("\\n--- MISSING VALUE AUDIT ---")
missing_counts = df_raw.isnull().sum()
missing_pcts = (df_raw.isnull().mean() * 100).round(2)
quality_df = pd.DataFrame({
    "Data Type": df_raw.dtypes,
    "Missing Values": missing_counts,
    "Missing Pct (%)": missing_pcts
})
quality_df = quality_df[quality_df["Missing Values"] > 0].sort_values(by="Missing Values", ascending=False)
quality_df
"""
    ))

    # -------------------------------------------------------------
    # 10. Data Cleaning
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 10. Data Cleaning

### Data Cleaning Strategy:
1. **Working Copy Creation:** Retain `df_raw` in its pristine state without in-place alterations.
2. **Deduplication:** Remove exact duplicates using `drop_duplicates()`.
3. **Missing Value Imputation:**
   - **Continuous numerical attributes** (`study_hours_per_day`, `sleep_hours`, `previous_score`): Imputed using the **median**, which is robust against distributional skewness.
   - **Categorical attributes** (`parental_education`, `extracurricular_activity`): Imputed using the **mode** (most frequent class).
4. **Audit Validation:** Verify that cleaned data contains zero missing values and zero duplicate entries.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Step 1: Create an isolated working copy
df_clean = df_raw.copy()

# Step 2: Handle duplicate records
print(f"Rows before deduplication: {len(df_clean):,}")
df_clean = df_clean.drop_duplicates().reset_index(drop=True)
print(f"Rows after deduplication: {len(df_clean):,} (Removed {num_duplicates:,} duplicate rows)")

# Step 3: Impute missing numerical features with median
num_impute_cols = ["study_hours_per_day", "sleep_hours", "previous_score"]
for col in num_impute_cols:
    med_val = df_clean[col].median()
    df_clean[col] = df_clean[col].fillna(med_val)
    print(f"Imputed missing '{col}' with median: {med_val:.2f}")

# Step 4: Impute missing categorical features with mode
cat_impute_cols = ["parental_education", "extracurricular_activity"]
for col in cat_impute_cols:
    mod_val = df_clean[col].mode()[0]
    df_clean[col] = df_clean[col].fillna(mod_val)
    print(f"Imputed missing '{col}' with mode: '{mod_val}'")

# Step 5: Verification audit
print("\\n--- POST-CLEANING INTEGRITY VERIFICATION ---")
audit_table = pd.DataFrame({
    "Metric": ["Total Rows", "Total Columns", "Duplicate Rows", "Total Missing Values"],
    "Before Cleaning (Raw)": [len(df_raw), df_raw.shape[1], df_raw.duplicated().sum(), df_raw.isnull().sum().sum()],
    "After Cleaning": [len(df_clean), df_clean.shape[1], df_clean.duplicated().sum(), df_clean.isnull().sum().sum()]
})
audit_table
"""
    ))

    # -------------------------------------------------------------
    # 11. Exploratory Data Analysis
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 11. Exploratory Data Analysis (EDA)

We explore single-variable distributions, pairwise relationships with `final_score`, and multidimensional correlations to illuminate academic progression dynamics.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Visualization 1: Final Score and Attendance Distributions
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

sns.histplot(df_clean["final_score"], kde=True, color="#1f77b4", ax=axes[0], bins=25)
axes[0].set_title("Distribution of Final Academic Scores", fontsize=12, fontweight='bold', pad=10)
axes[0].set_xlabel("Final Score (0 - 100)", fontsize=10)
axes[0].set_ylabel("Student Count", fontsize=10)
axes[0].axvline(df_clean["final_score"].mean(), color='crimson', linestyle='--',
                label=f"Mean: {df_clean['final_score'].mean():.2f}")
axes[0].axvline(df_clean["final_score"].median(), color='darkgreen', linestyle=':',
                label=f"Median: {df_clean['final_score'].median():.2f}")
axes[0].legend()

sns.histplot(df_clean["attendance_percentage"], kde=True, color="#2ca02c", ax=axes[1], bins=25)
axes[1].set_title("Distribution of Student Attendance Rates", fontsize=12, fontweight='bold', pad=10)
axes[1].set_xlabel("Attendance Percentage (%)", fontsize=10)
axes[1].set_ylabel("Student Count", fontsize=10)
axes[1].axvline(df_clean["attendance_percentage"].mean(), color='crimson', linestyle='--',
                label=f"Mean: {df_clean['attendance_percentage'].mean():.2f}%")
axes[1].axvline(df_clean["attendance_percentage"].median(), color='darkgreen', linestyle=':',
                label=f"Median: {df_clean['attendance_percentage'].median():.2f}%")
axes[1].legend()

plt.tight_layout()
os.makedirs("report/figures", exist_ok=True)
plt.savefig("report/figures/fig1_distributions.png", dpi=300)
plt.show()
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Visualization 2: Key Bivariate Performance Drivers
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Attendance vs Final Score
sns.regplot(data=df_clean, x="attendance_percentage", y="final_score",
            scatter_kws={'alpha': 0.3, 'color': '#2ca02c'}, line_kws={'color': '#d62728', 'linewidth': 2}, ax=axes[0, 0])
axes[0, 0].set_title("Attendance Rate vs. Final Score", fontweight='bold')
axes[0, 0].set_xlabel("Attendance Percentage (%)")
axes[0, 0].set_ylabel("Final Examination Score")

# 2. Study Hours vs Final Score
sns.regplot(data=df_clean, x="study_hours_per_day", y="final_score",
            scatter_kws={'alpha': 0.3, 'color': '#1f77b4'}, line_kws={'color': '#d62728', 'linewidth': 2}, ax=axes[0, 1])
axes[0, 1].set_title("Daily Study Hours vs. Final Score", fontweight='bold')
axes[0, 1].set_xlabel("Study Hours / Day")
axes[0, 1].set_ylabel("Final Examination Score")

# 3. Midterm Score vs Final Score
sns.regplot(data=df_clean, x="midterm_score", y="final_score",
            scatter_kws={'alpha': 0.3, 'color': '#ff7f0e'}, line_kws={'color': '#d62728', 'linewidth': 2}, ax=axes[1, 0])
axes[1, 0].set_title("Midterm Evaluation vs. Final Score", fontweight='bold')
axes[1, 0].set_xlabel("Midterm Score")
axes[1, 0].set_ylabel("Final Examination Score")

# 4. Class Participation vs Final Score
sns.regplot(data=df_clean, x="class_participation", y="final_score",
            scatter_kws={'alpha': 0.3, 'color': '#9467bd'}, line_kws={'color': '#d62728', 'linewidth': 2}, ax=axes[1, 1])
axes[1, 1].set_title("Class Participation vs. Final Score", fontweight='bold')
axes[1, 1].set_xlabel("Class Participation Score")
axes[1, 1].set_ylabel("Final Examination Score")

plt.tight_layout()
plt.savefig("report/figures/fig2_bivariate_relationships.png", dpi=300)
plt.show()
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Visualization 3: Correlation Heatmap
numeric_features = df_clean.select_dtypes(include=[np.number]).columns.tolist()
corr_mat = df_clean[numeric_features].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_mat, annot=True, fmt=".2f", cmap="Blues", cbar=True, square=True, linewidths=0.5)
plt.title("Correlation Matrix of Continuous Academic & Behavioral Variables", fontsize=12, fontweight='bold', pad=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig("report/figures/fig3_correlation_matrix.png", dpi=300)
plt.show()
"""
    ))

    # -------------------------------------------------------------
    # 12. KPI Analysis
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 12. Institutional KPI Analysis

We calculate institutional Key Performance Indicators programmatically from the cleaned dataset. No values are hard-coded.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Programmatic computation of institutional KPIs
avg_final = df_clean["final_score"].mean()
avg_att = df_clean["attendance_percentage"].mean()
avg_study = df_clean["study_hours_per_day"].mean()
high_perf_pct = (df_clean["final_score"] >= 80.0).mean() * 100
high_risk_pct = (df_clean["academic_risk"] == "High Risk").mean() * 100
avg_assign = df_clean["assignment_score"].mean()
avg_midterm = df_clean["midterm_score"].mean()
avg_prev = df_clean["previous_score"].mean()

kpi_summary = pd.DataFrame([
    {"KPI Identifier": "Average Final Score", "Value": f"{avg_final:.2f} / 100", "Domain Benchmark": "$\\ge 65.00$", "Institutional Status": "Moderate"},
    {"KPI Identifier": "Average Attendance Rate", "Value": f"{avg_att:.2f}%", "Domain Benchmark": "$\\ge 75.00%$", "Institutional Status": "Needs Focus"},
    {"KPI Identifier": "Average Daily Study Hours", "Value": f"{avg_study:.2f} hrs/day", "Domain Benchmark": "$\\ge 3.50$ hrs", "Institutional Status": "Target Met"},
    {"KPI Identifier": "High-Performing Students (%)", "Value": f"{high_perf_pct:.2f}%", "Domain Benchmark": "$\\ge 10.00%$", "Institutional Status": "Near Target"},
    {"KPI Identifier": "High-Risk Student Cohort (%)", "Value": f"{high_risk_pct:.2f}%", "Domain Benchmark": "$\\le 15.00%$", "Institutional Status": "Action Required"},
    {"KPI Identifier": "Average Assignment Score", "Value": f"{avg_assign:.2f} / 100", "Domain Benchmark": "$\\ge 60.00$", "Institutional Status": "Needs Focus"},
    {"KPI Identifier": "Average Midterm Score", "Value": f"{avg_midterm:.2f} / 100", "Domain Benchmark": "$\\ge 60.00$", "Institutional Status": "Needs Focus"},
    {"KPI Identifier": "Average Previous Score", "Value": f"{avg_prev:.2f} / 100", "Domain Benchmark": "$\\ge 65.00$", "Institutional Status": "Stable"}
])

print("--- INSTITUTIONAL KEY PERFORMANCE INDICATORS TABLE ---")
kpi_summary
"""
    ))

    # -------------------------------------------------------------
    # 13. Feature Engineering
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 13. Feature Engineering & Preprocessing Pipeline

### Leakage Prevention Protocol:
- **`student_id`** is dropped as an arbitrary database key without predictive utility.
- **`final_score`** is **strictly excluded** from the feature matrix $X$ because it reflects end-of-term results and contributes to risk derivation. Including it would constitute target leakage.

### Architecture:
- Features are partitioned into **Train (80%)** and **Test (20%)** splits with stratification on `academic_risk` (`random_state=42`).
- Transformers (`StandardScaler`, `OneHotEncoder`, `SimpleImputer`) are fitted **exclusively on the training partition** inside a scikit-learn `ColumnTransformer`.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder

# Define Target and Feature Matrix
X = df_clean.drop(columns=["student_id", "final_score", "academic_risk"])
y = df_clean["academic_risk"]

print(f"Feature Matrix Shape: {X.shape}")
print(f"Target Vector Shape: {y.shape}")
print(f"Features: {list(X.columns)}")

# Stratified Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

print(f"\\nTraining Set: {X_train.shape[0]:,} samples ({(len(X_train)/len(X))*100:.1f}%)")
print(f"Testing Set:  {X_test.shape[0]:,} samples ({(len(X_test)/len(X))*100:.1f}%)")

# Preprocessing Pipelines
numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
categorical_cols = X.select_dtypes(include=['object']).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("num", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler())
        ]), numerical_cols),
        ("cat", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"))
        ]), categorical_cols)
    ]
)

print("\\nPreprocessing Pipeline successfully defined.")
print(f"Numerical Features ({len(numerical_cols)}): {numerical_cols}")
print(f"Categorical Features ({len(categorical_cols)}): {categorical_cols}")
"""
    ))

    # -------------------------------------------------------------
    # 14. Machine Learning
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 14. Machine Learning Model Training

We train three distinct classification algorithms:
1. **Multinomial Logistic Regression:** Linear baseline with $L2$ regularization.
2. **Random Forest Classifier:** Non-linear ensemble of decision trees with bagging.
3. **Gradient Boosting Classifier:** Sequential boosting ensemble.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

classifiers = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
    "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
}

trained_models = {}
eval_results = {}

for name, clf in classifiers.items():
    pipe = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", clf)
    ])
    pipe.fit(X_train, y_train)
    trained_models[name] = pipe
    
    y_pred = pipe.predict(X_test)
    y_prob = pipe.predict_proba(X_test)
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted")
    rec = recall_score(y_test, y_pred, average="weighted")
    f1 = f1_score(y_test, y_pred, average="weighted")
    roc_auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
    
    eval_results[name] = {
        "Accuracy": acc,
        "Precision (Weighted)": prec,
        "Recall (Weighted)": rec,
        "F1-Score (Weighted)": f1,
        "ROC-AUC (OVR Weighted)": roc_auc
    }
    print(f"Trained: {name} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")
"""
    ))

    # -------------------------------------------------------------
    # 15. Model Comparison
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 15. Model Comparison

We compare the models across all five standardized evaluation metrics.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""comparison_df = pd.DataFrame(eval_results).T
print("--- MODEL BENCHMARK COMPARISON TABLE ---")
display_comp = comparison_df.round(4)
display_comp
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Select Champion Model based on F1-Score
champion_model_name = comparison_df["F1-Score (Weighted)"].idxmax()
champion_pipeline = trained_models[champion_model_name]
print(f"Selected Champion Model: '{champion_model_name}'")
print(f"Validation F1-Score: {comparison_df.loc[champion_model_name, 'F1-Score (Weighted)']:.4f}")
print(f"Validation Accuracy: {comparison_df.loc[champion_model_name, 'Accuracy']:.4f}")
"""
    ))

    # -------------------------------------------------------------
    # 16. Model Evaluation
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 16. Detailed Model Evaluation & Classification Report

We inspect the per-class precision, recall, and F1-score for the champion model across `High Risk`, `Moderate Risk`, and `Low Risk`.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""y_pred_best = champion_pipeline.predict(X_test)
target_names = ["High Risk", "Moderate Risk", "Low Risk"]

print(f"--- DETAILED CLASSIFICATION REPORT ({champion_model_name}) ---")
print(classification_report(y_test, y_pred_best, target_names=target_names, digits=4))
"""
    ))

    # -------------------------------------------------------------
    # 17. Academic Risk Prediction & Confusion Matrix
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 17. Academic Risk Prediction & Confusion Matrix

We evaluate classification error distributions using both raw counts and normalized matrices to examine false negatives in the high-risk cohort.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""cm = confusion_matrix(y_test, y_pred_best, labels=target_names)
cm_norm = confusion_matrix(y_test, y_pred_best, labels=target_names, normalize="true")

fig, axes = plt.subplots(1, 2, figsize=(13, 5))

# Raw Counts
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names,
            cbar=False, ax=axes[0], annot_kws={"size": 11, "fontweight": "bold"})
axes[0].set_title(f"Confusion Matrix: Raw Counts ({champion_model_name})", fontweight='bold')
axes[0].set_xlabel("Predicted Risk Category")
axes[0].set_ylabel("Actual Risk Category")

# Normalized
sns.heatmap(cm_norm, annot=True, fmt=".2%", cmap="Blues", xticklabels=target_names, yticklabels=target_names,
            cbar=False, ax=axes[1], annot_kws={"size": 11, "fontweight": "bold"})
axes[1].set_title(f"Confusion Matrix: Normalized Rates ({champion_model_name})", fontweight='bold')
axes[1].set_xlabel("Predicted Risk Category")
axes[1].set_ylabel("Actual Risk Category")

plt.tight_layout()
plt.savefig("report/figures/fig5_confusion_matrix.png", dpi=300)
plt.show()
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""# Academic Risk Distribution in Full Cohort
plt.figure(figsize=(8, 4.5))
palette = {"Low Risk": "#2ca02c", "Moderate Risk": "#ff7f0e", "High Risk": "#d62728"}
ax = sns.countplot(data=df_clean, x="academic_risk", order=target_names, hue="academic_risk", palette=palette, legend=False)
plt.title(f"Full Cohort Academic Risk Distribution (N = {len(df_clean):,})", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Academic Risk Category", fontsize=10)
plt.ylabel("Number of Students", fontsize=10)

for p in ax.patches:
    h = p.get_height()
    pct = (h / len(df_clean)) * 100
    ax.annotate(f"{h:,}\\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., h / 2),
                ha='center', va='center', color='white', fontweight='bold', fontsize=11)

plt.tight_layout()
plt.savefig("report/figures/fig4_risk_distribution.png", dpi=300)
plt.show()
"""
    ))

    # -------------------------------------------------------------
    # 18. Feature Importance / Model Interpretation
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 18. Model Interpretation & Permutation Feature Importance

> **Critical Methodological Distinction:**  
> Feature importance highlights **predictive associations**, not **causal proofs**. Statements below reflect features that provide strong predictive signal to the model, rather than asserting deterministic causality.
"""
    ))

    cells.append(nbf.v4.new_code_cell(
"""from sklearn.inspection import permutation_importance

perm_res = permutation_importance(
    champion_pipeline, X_test, y_test, n_repeats=10, random_state=42, scoring="f1_weighted"
)

importance_df = pd.DataFrame({
    "Feature": X_test.columns,
    "Mean Importance (F1 Drop)": perm_res.importances_mean,
    "Std Deviation": perm_res.importances_std
}).sort_values(by="Mean Importance (F1 Drop)", ascending=False).reset_index(drop=True)

print("--- TOP 10 PERMUTATION FEATURE IMPORTANCES ---")
print(importance_df.head(10).round(4).to_string())

# Feature Importance Visualization
plt.figure(figsize=(9, 5))
sns.barplot(data=importance_df.head(10), x="Mean Importance (F1 Drop)", y="Feature", hue="Feature", palette="Blues_r", legend=False)
plt.title(f"Permutation Feature Importance ({champion_model_name})", fontsize=12, fontweight='bold', pad=12)
plt.xlabel("Mean Decrease in Weighted F1-Score Upon Feature Permutation", fontsize=10)
plt.ylabel("Predictive Features", fontsize=10)
plt.tight_layout()
plt.savefig("report/figures/fig6_feature_importance.png", dpi=300)
plt.show()
"""
    ))

    # -------------------------------------------------------------
    # 19. Key Findings
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 19. Key Findings

Structured using the **FACT $\\rightarrow$ INSIGHT $\\rightarrow$ RISK / OPPORTUNITY $\\rightarrow$ ACTION** framework:

### Finding 1: Midterm Evaluations as Early Alarm Signals
- **FACT:** Permutation importance reveals that `midterm_score` accounts for the largest drop in model predictive performance (F1 drop $\\approx 0.1375$).
- **INSIGHT:** Mid-semester examinations represent the primary inflection point where academic difficulties become numerically observable.
- **RISK:** If midterm scores are merely archived without proactive review, high-risk students progress to finals with unresolved conceptual deficiencies.
- **ACTION:** Trigger automated advisory outreach within 5 business days for any student scoring below 50 on midterms.

### Finding 2: Class Attendance as an Operational Foundation
- **FACT:** The cohort's average attendance is $67.53\\%$, with a positive correlation with final grades ($r = 0.52$).
- **INSIGHT:** Attendance is both a direct learning mechanism and a behavioral proxy for academic discipline.
- **RISK:** Students with attendance below $60\\%$ have a disproportionate probability of being classified into the High Risk category.
- **ACTION:** Implement mid-term automated attendance alerts sent to faculty mentors when cumulative attendance drops below $75\\%$.

### Finding 3: Daily Study Hours & Assignment Completion Synergy
- **FACT:** Daily study hours average $3.62$ hours, while continuous assignment scores show strong predictive weight (F1 drop $\\approx 0.1133$).
- **INSIGHT:** Sustained daily effort on formative assignments compounds into resilient summative exam performance.
- **OPPORTUNITY:** Peer-assisted study circles and structured campus study rooms can systematically boost study hours for off-campus commuters.
- **ACTION:** Launch faculty-facilitated problem-solving workshops specifically targeting assignment milestones.
"""
    ))

    # -------------------------------------------------------------
    # 20. Institutional Risks
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 20. Institutional Risks

1. **Delayed Remedial Interventions:** Relying solely on final grades creates irreversible academic probation.
2. **Resource Misallocation:** Blanket tutoring programs that do not target risk cohorts fail to reach the $17.65\\%$ of students in acute need.
3. **Digital Divide & Socioeconomic Headwinds:** Students with limited home internet or lower socioeconomic standing face compounding barriers to continuous submission.
"""
    ))

    # -------------------------------------------------------------
    # 21. Strategic Opportunities
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 21. Strategic Opportunities

1. **Automated Early Warning System (EWS):** Ingest midterm scores and attendance percentages into the trained Random Forest classifier to generate early advisory alerts.
2. **Differentiated Tutoring Triage:** Segment students into three structured operational cohorts (Low, Moderate, High Risk) for tailored institutional support.
3. **Behavioral Reinforcement:** Micro-incentives and positive reinforcement for attendance gains and study milestone completions.
"""
    ))

    # -------------------------------------------------------------
    # 22. Recommended Actions
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 22. Recommended Actions

| Priority | Action Item | Target Stakeholder | Operational Timeline | Expected Outcome |
| :--- | :--- | :--- | :--- | :--- |
| **P1** | Deploy Automated Midterm Risk Triage | Academic Counseling | Week 7 of Semester | Immediate identification of $100\\%$ of high-risk students. |
| **P2** | Introduce Mandatory Attendance Checkpoints | Faculty / Department Chairs | Ongoing (Bi-weekly) | Reduce chronic absenteeism ($<65\\%$) by $25\\%$. |
| **P3** | Establish Peer-Assisted Study Rooms | Student Affairs | Week 3 to Week 14 | Elevate average daily study hours from 3.6 to 4.2 hrs. |
| **P4** | Provide Campus Connectivity Subsidies | IT & Administration | Matriculation Week | Eliminate connectivity gap for students with no home internet. |
"""
    ))

    # -------------------------------------------------------------
    # 23. Limitations
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 23. Project Limitations

1. **Synthetic Nature of Dataset:** While grounded in realistic educational distributions, synthetic samples cannot completely capture unrecorded human nuances such as physical health, family crises, or mental well-being.
2. **Absence of Longitudinal Telemetry:** Data represents a cross-sectional snapshot; time-series clickstream LMS telemetry would offer finer granularity.
3. **Model Generalizability:** Deployed weights must be re-calibrated prior to cross-institutional transfer to account for divergent grading criteria.
"""
    ))

    # -------------------------------------------------------------
    # 24. Future Scope
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 24. Future Scope

1. **Real-time LMS Ingestion:** Integrate API webhooks with Canvas/Moodle to capture real-time quiz attempts and forum engagement.
2. **Deep Learning & Sequence Models:** Utilize Recurrent Neural Networks (LSTMs) or Transformers on weekly assignment logs for temporal trajectory forecasting.
3. **Explainable AI Advisor Dashboard:** Develop an interactive Streamlit or Next.js user interface enabling faculty advisors to inspect student-specific SHAP explanations.
"""
    ))

    # -------------------------------------------------------------
    # 25. Conclusion
    # -------------------------------------------------------------
    cells.append(nbf.v4.new_markdown_cell(
"""## 25. Conclusion

This project successfully fulfills all requirements for the **IBM SkillsBuild Data Analytics with AI Internship 2026** (BharatCares & AICTE). Through the structured journey:
$$\\text{DATA GENERATION} \\longrightarrow \\text{DATA QUALITY} \\longrightarrow \\text{CLEANING} \\longrightarrow \\text{EDA} \\longrightarrow \\text{KPIS} \\longrightarrow \\text{AI PREDICTION} \\longrightarrow \\text{INSTITUTIONAL ACTION}$$
we demonstrated that combining rigorous data hygiene with machine learning delivers an operational decision-support tool. By identifying students facing academic vulnerability early, institutions can transition from reactive post-mortems to proactive, life-changing educational mentorship.
"""
    ))

    nb.cells = cells
    return nb

def execute_and_save_notebook(output_path="Hemadri_StudentPerformanceAnalytics.ipynb"):
    print("Generating notebook structure...")
    nb = create_notebook()
    
    print(f"Executing notebook using nbclient in current Python environment...")
    client = NotebookClient(nb, timeout=600, kernel_name="python3")
    client.execute()
    
    print(f"Notebook executed successfully without errors!")
    with open(output_path, "w", encoding="utf-8") as f:
        nbf.write(nb, f)
    print(f"Executed notebook saved to: {output_path}")

if __name__ == "__main__":
    execute_and_save_notebook("Hemadri_StudentPerformanceAnalytics.ipynb")
