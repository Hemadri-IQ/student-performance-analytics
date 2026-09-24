"""
========================================================================================
AI-POWERED STUDENT PERFORMANCE ANALYTICS & EARLY RISK PREDICTION SYSTEM
========================================================================================
Internship Program : IBM SkillsBuild Data Analytics with AI Internship 2026
Association        : BharatCares in association with AICTE
Lead Developer     : Hemadri
Environment        : Python 3.10+ (Data Science Stack)
Submission File    : Main Executable Source Code File (Capstone Project)
========================================================================================
Description:
End-to-end Python implementation covering data synthesis, automated quality audit,
missing data & duplicate handling, operational KPI calculation, bivariate EDA,
multi-class machine learning classification (Logistic Regression, Random Forest, 
Gradient Boosting), permutation feature importance, and risk stratification.
========================================================================================
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


# ========================================================================================
# 1. SYNTHETIC DATA GENERATION MODULE
# ========================================================================================
def generate_student_dataset(output_path="data/student_performance.csv", random_seed=42):
    """
    Generates a realistic 2,050-student synthetic dataset modeling demographic,
    behavioral, and academic attributes with realistic noise, duplicates, and missingness.
    """
    print("\n" + "="*80)
    print("STEP 1: SYNTHETIC DATASET GENERATION")
    print("="*80)
    np.random.seed(random_seed)
    n_unique = 2020  # Base unique records

    # Unique student IDs
    student_ids = [f"STU{1000 + i:04d}" for i in range(1, n_unique + 1)]

    # Demographics and background
    genders = np.random.choice(["Female", "Male", "Other"], size=n_unique, p=[0.49, 0.49, 0.02])
    ages = np.random.randint(17, 23, size=n_unique)
    extracurricular = np.random.choice(["Yes", "No"], size=n_unique, p=[0.43, 0.57])
    internet_access = np.random.choice(["Yes", "No"], size=n_unique, p=[0.86, 0.14])
    parental_education = np.random.choice(
        ["High School", "Associate", "Bachelor", "Master", "Doctorate"],
        size=n_unique,
        p=[0.25, 0.20, 0.35, 0.15, 0.05]
    )
    family_income = np.random.choice(
        ["Low", "Medium", "High"],
        size=n_unique,
        p=[0.30, 0.55, 0.15]
    )

    # Behavioral metrics
    study_hours = np.round(np.clip(np.random.normal(3.8, 1.4, size=n_unique), 0.5, 9.5), 1)
    sleep_hours = np.round(np.clip(np.random.normal(6.8, 1.1, size=n_unique), 4.0, 10.0), 1)
    attendance = np.round(np.clip(np.random.beta(5, 2, size=n_unique) * 100, 30.0, 100.0), 1)
    class_participation = np.random.choice(["Low", "Medium", "High"], size=n_unique, p=[0.25, 0.50, 0.25])

    # Assessment scores
    previous_score = np.round(np.clip(np.random.normal(68.0, 12.0, size=n_unique), 35.0, 98.0), 1)
    assignment_score = np.round(np.clip(0.4 * previous_score + 0.3 * attendance + np.random.normal(10, 8, n_unique), 20.0, 100.0), 1)
    midterm_score = np.round(np.clip(0.5 * previous_score + 0.25 * (study_hours * 8) + np.random.normal(5, 9, n_unique), 15.0, 99.0), 1)

    # Synthetic final score formula (Domain logic)
    final_score_raw = (
        0.30 * midterm_score +
        0.25 * assignment_score +
        0.20 * previous_score +
        0.15 * attendance +
        1.80 * study_hours +
        np.random.normal(0, 4.5, n_unique)
    )
    final_score = np.round(np.clip(final_score_raw, 10.0, 100.0), 1)

    # Multi-class target derivation
    risk_labels = []
    for score in final_score:
        if score < 50.0:
            risk_labels.append("High Risk")
        elif score < 72.0:
            risk_labels.append("Moderate Risk")
        else:
            risk_labels.append("Low Risk")

    df = pd.DataFrame({
        "student_id": student_ids,
        "gender": genders,
        "age": ages,
        "study_hours_per_day": study_hours,
        "attendance_percentage": attendance,
        "previous_score": previous_score,
        "assignment_score": assignment_score,
        "midterm_score": midterm_score,
        "extracurricular_activity": extracurricular,
        "internet_access": internet_access,
        "sleep_hours": sleep_hours,
        "parental_education": parental_education,
        "family_income_category": family_income,
        "class_participation": class_participation,
        "final_score": final_score,
        "academic_risk": risk_labels
    })

    # Inject realistic duplicates (30 duplicate entries)
    duplicate_indices = np.random.choice(df.index, size=30, replace=False)
    duplicates_df = df.loc[duplicate_indices].copy()
    df = pd.concat([df, duplicates_df], ignore_index=True)

    # Inject missing values (100 total missing entries across key columns)
    num_missing = 100
    for _ in range(num_missing):
        row_idx = np.random.randint(0, len(df))
        col_name = np.random.choice([
            "study_hours_per_day", "sleep_hours", "previous_score",
            "parental_education", "extracurricular_activity"
        ])
        df.loc[row_idx, col_name] = np.nan

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset successfully created and exported to: {output_path}")
    print(f"Total Records: {len(df)} | Columns: {len(df.columns)}")
    return output_path


# ========================================================================================
# 2. MAIN ANALYTICAL & MACHINE LEARNING PIPELINE
# ========================================================================================
def run_analytics_pipeline(data_path="data/student_performance.csv", figures_dir="report/figures"):
    """
    Executes data quality audit, cleaning, KPI calculation, EDA visualization,
    model training/evaluation, and feature importance extraction.
    """
    os.makedirs(figures_dir, exist_ok=True)
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    plt.rcParams['font.sans-serif'] = 'Arial'

    # --- 1. DATA AUDIT & INGESTION ---
    print("\n" + "="*80)
    print("STEP 2: DATA AUDIT & HYGIENE")
    print("="*80)
    if not os.path.exists(data_path):
        generate_student_dataset(data_path)

    df_raw = pd.read_csv(data_path)
    raw_rows, raw_cols = df_raw.shape
    raw_duplicates = int(df_raw.duplicated().sum())
    raw_missing = int(df_raw.isnull().sum().sum())
    missing_by_col = df_raw.isnull().sum()[df_raw.isnull().sum() > 0].to_dict()

    print(f"Raw Dataset Scope : {raw_rows:,} Rows | {raw_cols} Features")
    print(f"Duplicate Count   : {raw_duplicates} records")
    print(f"Missing Values    : {raw_missing} total entries")
    print("Missing Breakdown :", missing_by_col)

    # --- 2. DATA CLEANING & IMPUTATION ---
    print("\n" + "="*80)
    print("STEP 3: DATA CLEANING & PREPROCESSING")
    print("="*80)
    df_clean = df_raw.copy()
    df_clean = df_clean.drop_duplicates().reset_index(drop=True)

    # Impute numerical features using median
    for c in ["study_hours_per_day", "sleep_hours", "previous_score"]:
        df_clean[c] = df_clean[c].fillna(df_clean[c].median())

    # Impute categorical features using mode
    for c in ["parental_education", "extracurricular_activity"]:
        df_clean[c] = df_clean[c].fillna(df_clean[c].mode()[0])

    clean_rows, clean_cols = df_clean.shape
    print(f"Cleaned Dataset   : {clean_rows:,} Unique Rows | {clean_cols} Features")
    print(f"Verification      : {df_clean.duplicated().sum()} duplicates | {df_clean.isnull().sum().sum()} missing values remaining")

    # --- 3. OPERATIONAL KPIS ---
    print("\n" + "="*80)
    print("STEP 4: OPERATIONAL KEY PERFORMANCE INDICATORS (KPIS)")
    print("="*80)
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

    for metric, value in kpis.items():
        print(f"  - {metric:<22}: {value:.2f}")

    risk_counts = df_clean["academic_risk"].value_counts().to_dict()
    risk_pcts = (df_clean["academic_risk"].value_counts(normalize=True) * 100).to_dict()
    print("\nRisk Cohort Breakdown :", risk_counts)
    print("Risk Percentage       :", {k: f"{v:.1f}%" for k, v in risk_pcts.items()})

    # --- 4. EXPLORATORY DATA ANALYSIS (EDA) VISUALIZATIONS ---
    print("\n" + "="*80)
    print("STEP 5: GENERATING EDA PLOTS & VISUALIZATIONS")
    print("="*80)

    # Plot 1: Distributions
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    sns.histplot(df_clean["final_score"], kde=True, color="#1f77b4", ax=axes[0], bins=25)
    axes[0].set_title("Distribution of Final Academic Scores", fontweight='bold')
    axes[0].set_xlabel("Final Score (0-100)")
    axes[0].axvline(kpis["avg_final_score"], color='red', linestyle='--', label=f'Mean: {kpis["avg_final_score"]:.1f}')
    axes[0].legend()

    sns.histplot(df_clean["attendance_percentage"], kde=True, color="#2ca02c", ax=axes[1], bins=25)
    axes[1].set_title("Distribution of Student Attendance Rate", fontweight='bold')
    axes[1].set_xlabel("Attendance Percentage (%)")
    axes[1].axvline(kpis["avg_attendance"], color='red', linestyle='--', label=f'Mean: {kpis["avg_attendance"]:.1f}%')
    axes[1].legend()
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig1_distributions.png"), dpi=300)
    plt.close()

    # Plot 2: Correlation Matrix
    plt.figure(figsize=(9, 7))
    num_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
    sns.heatmap(df_clean[num_cols].corr(), annot=True, fmt=".2f", cmap="Blues", square=True)
    plt.title("Correlation Matrix of Numeric Attributes", fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig3_correlation_matrix.png"), dpi=300)
    plt.close()

    # Plot 3: Academic Risk Distribution
    plt.figure(figsize=(7, 4.5))
    order = ["Low Risk", "Moderate Risk", "High Risk"]
    palette = {"Low Risk": "#2ca02c", "Moderate Risk": "#ff7f0e", "High Risk": "#d62728"}
    ax = sns.countplot(data=df_clean, x="academic_risk", order=order, palette=palette, hue="academic_risk", legend=False)
    plt.title("Academic Risk Distribution (N = 2,023)", fontweight='bold')
    plt.xlabel("Risk Category")
    plt.ylabel("Student Count")
    for p in ax.patches:
        h = p.get_height()
        pct = (h / len(df_clean)) * 100
        ax.annotate(f"{h:,}\n({pct:.1f}%)", (p.get_x() + p.get_width() / 2., h / 2),
                    ha='center', va='center', color='white', fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig4_risk_distribution.png"), dpi=300)
    plt.close()
    print("EDA Visualizations saved to:", figures_dir)

    # --- 5. MACHINE LEARNING MODELING ---
    print("\n" + "="*80)
    print("STEP 6: MACHINE LEARNING MODEL TRAINING & BENCHMARKING")
    print("="*80)
    X = df_clean.drop(columns=["student_id", "final_score", "academic_risk"])
    y = df_clean["academic_risk"]

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

    candidate_models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=150, max_depth=8, random_state=42),
        "Gradient Boosting": GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=4, random_state=42)
    }

    results = {}
    trained_pipelines = {}
    classes = ["High Risk", "Moderate Risk", "Low Risk"]

    for name, clf in candidate_models.items():
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
        print(f"Accuracy: {acc:.4f} | Precision: {prec:.4f} | Recall: {rec:.4f} | F1-Score: {f1:.4f} | ROC-AUC: {roc_auc:.4f}")

    df_metrics = pd.DataFrame(results).T
    print("\nModel Comparison Table:")
    print(df_metrics.round(4).to_string())

    champion_name = df_metrics["F1-Score"].idxmax()
    champion_pipe = trained_pipelines[champion_name]
    print(f"\n* SELECTED CHAMPION MODEL: {champion_name} *")

    # Confusion Matrix for Champion Model
    y_pred_best = champion_pipe.predict(X_test)
    cm = confusion_matrix(y_test, y_pred_best, labels=classes)

    plt.figure(figsize=(6.5, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes, cbar=False)
    plt.title(f"Confusion Matrix: {champion_name} (Test Set)", fontweight='bold')
    plt.xlabel("Predicted Risk Category")
    plt.ylabel("Actual Risk Category")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig5_confusion_matrix.png"), dpi=300)
    plt.close()

    # --- 6. PERMUTATION FEATURE IMPORTANCE ---
    print("\n" + "="*80)
    print("STEP 7: PERMUTATION FEATURE IMPORTANCE EXTRACTION")
    print("="*80)
    perm_imp = permutation_importance(champion_pipe, X_test, y_test, n_repeats=10, random_state=42, scoring="f1_weighted")
    perm_df = pd.DataFrame({
        "Feature": X_test.columns,
        "Importance": perm_imp.importances_mean,
        "Std": perm_imp.importances_std
    }).sort_values(by="Importance", ascending=False)

    print(perm_df.head(10).to_string(index=False))

    plt.figure(figsize=(8, 5))
    sns.barplot(data=perm_df.head(10), x="Importance", y="Feature", palette="Blues_r", hue="Feature", legend=False)
    plt.title(f"Permutation Feature Importance ({champion_name})", fontweight='bold')
    plt.xlabel("Mean Decrease in Weighted F1-Score")
    plt.ylabel("Features")
    plt.tight_layout()
    plt.savefig(os.path.join(figures_dir, "fig6_feature_importance.png"), dpi=300)
    plt.close()

    # Export Metrics JSON
    summary_stats = {
        "author": "Hemadri",
        "clean_rows": clean_rows,
        "kpis": kpis,
        "metrics": {k: {m: round(v, 4) for m, v in vals.items()} for k, vals in results.items()},
        "best_model": champion_name,
        "top_features": perm_df.head(10).to_dict(orient="records")
    }
    with open("report/metrics_summary.json", "w") as f:
        json.dump(summary_stats, f, indent=2)

    print("\n" + "="*80)
    print("EXECUTION COMPLETED SUCCESSFULLY!")
    print(f"Champion Model : {champion_name} ({results[champion_name]['Accuracy']*100:.2f}% Accuracy)")
    print(f"Metrics Output : report/metrics_summary.json")
    print("="*80 + "\n")


if __name__ == "__main__":
    run_analytics_pipeline()
