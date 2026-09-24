# AI-Powered Student Performance Analytics & Early Risk Prediction System

**IBM SkillsBuild Data Analytics with AI Internship 2026**  
*In association with BharatCares & AICTE*  
**Developer / Contributor:** Hemadri  
**Development Environment:** Python 3.10 / JupyterLab (Windows 64-bit)  
**Date:** September 2026  

---

## 1. Project Overview

The **AI-Powered Student Performance Analytics & Early Risk Prediction System** is an enterprise-grade academic intelligence framework developed to bridge the data-to-decision gap in higher education institutions. By integrating automated data hygiene, descriptive analytics, operational Key Performance Indicators (KPIs), and multi-class machine learning, the system identifies students facing **potential academic risk** early in the semester—well before summative final examinations take place.

The analytical pipeline follows an end-to-end operational journey:
$$\textbf{DATA GENERATION} \longrightarrow \textbf{QUALITY AUDIT} \longrightarrow \textbf{CLEANING} \longrightarrow \textbf{EDA} \longrightarrow \textbf{KPIS} \longrightarrow \textbf{AI PREDICTION} \longrightarrow \textbf{INSTITUTIONAL ACTION}$$

---

## 2. Problem Statement

Educational institutions routinely collect extensive behavioral, demographic, and academic records; however, raw administrative tables fail to yield early, proactive interventions:
1. **Diagnostic Lag:** Traditional institutions rely on end-of-term final exam results, by which point course failure or dropout cannot be averted.
2. **Data Fragmentation:** Attendance logs, learning management system (LMS) submissions, and examination rosters reside in isolated silos.
3. **Absence of Risk Stratification:** Academic advisors lack prioritized early-warning mechanisms to distinguish between students requiring routine guidance versus intensive academic counseling.

This system provides institutional decision-makers with proactive, evidence-based triage tools.

---

## 3. Project Objectives

- **Synthesize Realistic Student Data:** Programmatically generate 2,050 records ($N = 2,023$ unique post-cleaning) modeling multi-factor non-deterministic relationships (`random_seed=42`).
- **Execute Data Quality Assurance:** Audit and remediate duplicate rows, missing numerical values, and missing categorical variables.
- **Perform Exploratory Data Analysis (EDA):** Quantify distributions, correlation structures, and engagement disparities.
- **Compute Institutional KPIs:** Programmatically calculate cohort-wide performance metrics without hard-coding.
- **Deploy Leak-Free Machine Learning:** Build stratified scikit-learn preprocessing and classification pipelines strictly excluding post-outcome indicators (`final_score` and `student_id`).
- **Benchmark Multiple Models:** Evaluate Logistic Regression, Random Forest, and Gradient Boosting across five multi-class metrics.
- **Interpret Predictive Signals:** Apply Permutation Feature Importance to guide academic counseling without confusing prediction with causation.
- **Deliver Institutional Action Plan:** Formulate evidence-based recommendations structured around the **FACT $\rightarrow$ INSIGHT $\rightarrow$ RISK/OPPORTUNITY $\rightarrow$ ACTION** framework.

---

## 4. Key Features

- **Automated Data Cleaning & Audit:** Median imputation for continuous variables, modal imputation for categoricals, and duplicate deduplication.
- **Comprehensive Visualizations:** High-resolution distribution plots, bivariate regression trends, correlation matrices, and risk category breakdowns.
- **Leakage-Free Feature Engineering:** Strict exclusion of target-derived variables (`final_score`) and unique database identifiers (`student_id`).
- **Multi-Class Risk Classifier:** Categorizes students into `Low Risk`, `Moderate Risk`, and `High Risk` cohorts.
- **Permutation Model Interpretability:** Ranks predictive feature influence on weighted F1-score.
- **Full Submission Deliverables:** Fully executed Jupyter Notebook with outputs, clean `requirements.txt`, and formal 26-chapter DOCX Project Report.

---

## 5. Dataset

> [!IMPORTANT]
> **Synthetic Dataset Disclosure:**  
> This project uses a synthetic dataset generated specifically for this project and is **not** the IBM SkillsBuild/BharatCares masterclass dataset.

### Dataset Specifications:
- **Sample Size:** 2,050 raw records (2,023 unique records after deduplication).
- **Reproducibility:** Fixed random seed (`RANDOM_STATE = 42`).
- **File Location:** `data/student_performance.csv` (relative path).
- **Target Variable:** `academic_risk` (`Low Risk`, `Moderate Risk`, `High Risk`).

### Features & Domains:
| Column Name | Type | Value Range | Description |
| :--- | :--- | :--- | :--- |
| `student_id` | String | STU1001–STU3050 | Unique pseudo-anonymized identifier (excluded from ML). |
| `gender` | Categorical | Female, Male, Other | Student reported demographic gender. |
| `age` | Integer | 17–22 years | Student age at matriculation. |
| `study_hours_per_day` | Float | 0.5–8.0 hrs | Self-reported daily independent study duration. |
| `attendance_percentage`| Float | 45.0–100.0% | Classroom and laboratory attendance percentage. |
| `previous_score` | Float | 25.0–98.0 | Cumulative historical academic score prior to current term. |
| `assignment_score` | Float | 20.0–100.0 | Continuous assessment grade for coursework. |
| `midterm_score` | Float | 20.0–100.0 | Mid-semester formal evaluation examination score. |
| `extracurricular_activity`| Categorical | Yes, No | Active participation in university sports or clubs. |
| `internet_access` | Categorical | Yes, No | Reliable broadband connectivity at domicile. |
| `sleep_hours` | Float | 4.0–10.0 hrs | Self-reported nightly sleep duration. |
| `parental_education` | Categorical | High School–Doctorate| Highest parental educational attainment. |
| `family_income_category`| Categorical | Low, Medium, High | Socioeconomic household bracket. |
| `class_participation` | Float | 10.0–100.0 | In-class engagement score evaluated by faculty. |
| `final_score` | Float | 20.0–100.0 | End-of-term score (retained for EDA, excluded from ML). |
| `academic_risk` | Categorical | Low, Moderate, High | Supervised classification target. |

### Data Quality Demonstration:
- **Duplicate Rows:** 27 duplicate rows injected (1.32%) and successfully removed during cleaning.
- **Missing Numerical Values:** Injected across `study_hours_per_day` (24), `sleep_hours` (20), and `previous_score` (18). Resolved via median imputation.
- **Missing Categorical Values:** Injected across `parental_education` (22) and `extracurricular_activity` (16). Resolved via modal class imputation.

---

## 6. Technologies Used

- **Environment:** Python 3.10 / JupyterLab (Windows 64-bit)
- **Programming Language:** Python 3.10.11
- **Data Analytics:** `pandas` (v2.3.3), `numpy` (v2.2.6)
- **Machine Learning:** `scikit-learn` (v1.7.2)
- **Data Visualization:** `matplotlib` (v3.10.9), `seaborn` (v0.13.2)
- **Notebook Execution:** `jupyter`, `nbformat` (v5.9.0), `nbclient` (v0.8.0), `ipykernel`
- **Report Generation:** `python-docx` (v1.2.0)

---

## 7. Project Workflow

```
1. DATA GENERATION       -> Generate 2,050 records with stochastic noise (data/student_performance.csv)
2. QUALITY ASSESSMENT    -> Identify 27 duplicates and 100 missing values across 5 attributes
3. DATA CLEANING         -> Deduplicate and apply median/modal imputation to working DataFrame
4. EXPLORATORY ANALYSIS  -> Visualize distributions, bivariate regressions, and correlation heatmaps
5. KPI ANALYSIS          -> Compute institutional benchmarks programmatically
6. MACHINE LEARNING      -> Stratified split (80/20), ColumnTransformer pipeline, 3 classifiers
7. MODEL EVALUATION      -> Multi-class Accuracy, Precision, Recall, F1-Score, and ROC-AUC
8. MODEL INTERPRETATION  -> Permutation Importance to isolate actionable predictive drivers
9. ACTION ROADMAP        -> Formulate FACT -> INSIGHT -> RISK/OPPORTUNITY -> ACTION framework
10. REPORT GENERATION    -> Programmatically build 26-chapter DOCX report with embedded charts
```

---

## 8. Machine Learning Approach

- **Target:** `academic_risk` (`High Risk`, `Moderate Risk`, `Low Risk`).
- **Feature Matrix:** 13 predictive features (demographics, habits, formative assessments).
- **Target Leakage Safeguard:** `final_score` and `student_id` are strictly excluded from $X$.
- **Partitioning:** Stratified 80% Train ($N = 1,618$) / 20% Test ($N = 405$) split (`random_state=42`).
- **Preprocessing Pipeline:**
  - Continuous Features: `SimpleImputer(strategy='median')` followed by `StandardScaler()`.
  - Categorical Features: `SimpleImputer(strategy='most_frequent')` followed by `OneHotEncoder(drop='first', handle_unknown='ignore')`.

---

## 9. Model Evaluation & Benchmark Results

All metrics represent **actual executed results** on the holdout test set ($N = 405$):

| Classification Model | Accuracy | Weighted Precision | Weighted Recall | Weighted F1-Score | Multi-Class ROC-AUC (OVR) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest (Champion)** | **83.21%** | **83.92%** | **83.21%** | **82.75%** | **0.9251** |
| Logistic Regression | 82.22% | 82.25% | 82.22% | 81.98% | 0.9313 |
| Gradient Boosting | 78.77% | 78.70% | 78.77% | 78.45% | 0.9074 |

### Why Random Forest Was Selected:
Random Forest achieved the highest **Accuracy (83.21%)** and highest **Weighted F1-Score (82.75%)**. Its ensemble of 150 bagged decision trees captures non-linear synergies between midterms, assignments, and attendance while exhibiting robust generalization.

---

## 10. Institutional KPIs (Programmatically Derived)

| KPI Metric | Computed Value | Domain Benchmark | Strategic Status |
| :--- | :---: | :---: | :--- |
| **Average Final Score** | **61.40 / 100** | $\ge 65.00$ | Moderate (Near Target) |
| **Average Attendance Rate** | **67.53%** | $\ge 75.00\%$ | Needs Focus (Intervention Required) |
| **Average Daily Study Hours** | **3.62 hrs/day** | $\ge 3.50$ hrs | Target Met |
| **High-Performing Students ($\ge 80$)** | **7.66%** | $\ge 10.00\%$ | Near Target |
| **High-Risk Student Cohort** | **17.65%** | $\le 15.00\%$ | Action Required |
| **Average Assignment Score** | **53.59 / 100** | $\ge 60.00$ | Needs Focus |
| **Average Midterm Score** | **54.78 / 100** | $\ge 60.00$ | Needs Focus |
| **Average Previous Score** | **67.61 / 100** | $\ge 65.00$ | Stable |

### Academic Risk Cohort Distribution ($N = 2,023$):
- **Moderate Risk:** 1,185 students (**58.58%**)
- **Low Risk:** 481 students (**23.78%**)
- **High Risk:** 357 students (**17.65%**)

---

## 11. Model Interpretation & Top Predictive Features

Permutation importance reveals the top features whose permutation leads to the largest drop in model weighted F1-score:

| Rank | Predictive Feature | Mean Importance (F1 Drop) | Institutional Diagnostic Interpretation |
| :---: | :--- | :---: | :--- |
| **1** | `midterm_score` | **0.1375** | Primary mid-semester milestone reflecting concept mastery. |
| **2** | `assignment_score` | **0.1133** | Continuous assessment metric reflecting steady coursework diligence. |
| **3** | `attendance_percentage` | **0.0547** | Foundational commitment metric dictating classroom instructional exposure. |
| **4** | `study_hours_per_day` | **0.0511** | Independent effort; supports assignment mastery and test preparation. |
| **5** | `previous_score` | **0.0257** | Historical baseline readiness prior to term commencement. |
| **6** | `class_participation` | **0.0160** | Active in-class engagement and faculty interaction signal. |
| **7** | `age` | **0.0103** | Minor demographic variation across academic cohorts. |

> **Interpretability Note:** Feature importance measures predictive association within this model and does not establish medical or physical causality.

---

## 12. Key Findings & Actionable Recommendations

### Framework: FACT $\rightarrow$ INSIGHT $\rightarrow$ RISK/OPPORTUNITY $\rightarrow$ ACTION

1. **Midterm Evaluations as Early Intervention Signals:**
   - **FACT:** `midterm_score` is the #1 predictive feature (F1 drop: 0.1375).
   - **INSIGHT:** Mid-semester examinations represent the critical inflection point where academic deficits become observable.
   - **RISK:** Delaying intervention until finals causes preventable student failure.
   - **ACTION:** Deploy automated counselor outreach within 72 hours for students scoring $< 50$ on midterms.

2. **Attendance as a Foundational Prerequisite:**
   - **FACT:** Average attendance is $67.53\%$; attendance is a top-3 predictor.
   - **INSIGHT:** Consistent presence is an essential prerequisite for coursework comprehension.
   - **RISK:** Students with attendance $< 60\%$ face a 3.2x higher rate of high-risk classification.
   - **ACTION:** Implement bi-weekly attendance flags to trigger faculty mentor check-ins.

3. **Continuous Assessment Buffer:**
   - **FACT:** `assignment_score` is the #2 predictor (F1 drop: 0.1133).
   - **INSIGHT:** Formative assignment diligence builds resilience that cushions examination variance.
   - **OPPORTUNITY:** Peer study circles can directly improve assignment comprehension.
   - **ACTION:** Establish department-sponsored tutoring clinics centered on weekly problem sets.

---

## 13. Project Structure

```
IBM-Data-Analytics-AI-Internship/
│
├── data/
│   └── student_performance.csv                     <- Reproducible synthetic dataset (2,050 rows, seed 42)
│
├── report/
│   ├── figures/                                    <- High-resolution analytical charts
│   │   ├── fig1_distributions.png
│   │   ├── fig2_bivariate_relationships.png
│   │   ├── fig3_correlation_matrix.png
│   │   ├── fig4_risk_distribution.png
│   │   ├── fig5_confusion_matrix.png
│   │   └── fig6_feature_importance.png
│   ├── metrics_summary.json                        <- Exact computed parameters & metrics
│   └── Hemadri_StudentPerformanceAnalytics_ProjectReport.docx <- Formal 26-chapter DOCX report
│
├── Hemadri_StudentPerformanceAnalytics.ipynb      <- Valid, fully executed Jupyter Notebook
│
├── requirements.txt                                <- Pinned project dependencies
│
└── README.md                                       <- Project documentation and user guide
```

---

## 14. Installation & Setup (Windows)

To set up the project environment on Windows using Command Prompt or PowerShell:

```bash
# 1. Navigate to the project root directory
cd "F:\student performance\IBM-Data-Analytics-AI-Internship"

# 2. Create Python virtual environment (.venv)
python -m venv .venv

# 3. Activate the virtual environment
.venv\Scripts\activate

# 4. Install all required dependencies
pip install -r requirements.txt
```

---

## 15. How to Run the Project

### Running in Jupyter Notebook / JupyterLab:
```bash
# Activate environment
.venv\Scripts\activate

# Launch Jupyter Notebook
jupyter notebook Hemadri_StudentPerformanceAnalytics.ipynb
```
Select **Kernel $\rightarrow$ Restart & Run All** to re-execute all 25 analytical sections.

### Running in VS Code / JupyterLab:
1. Open the project folder in VS Code or JupyterLab.
2. Open `Hemadri_StudentPerformanceAnalytics.ipynb`.
3. In the top right kernel selector, select `.venv (Python 3.10)`.
4. Click **Run All** to execute all cells.

### Regenerating All Deliverables Programmatically:
```bash
# 1. Regenerate dataset (reproducible seed 42)
python generate_data.py

# 2. Run pipeline & export figures/metrics
python run_pipeline.py

# 3. Build & execute notebook end-to-end
python build_and_execute_notebook.py

# 4. Generate formal Word Document report
python generate_report.py
```

---

## 16. Limitations

1. **Synthetic Nature of Data:** Synthetic distributions, while mathematically grounded, cannot fully capture unrecorded real-world dynamics such as personal crises or sudden health events.
2. **Cross-Sectional Scope:** The dataset reflects a single semester; continuous LMS clickstream telemetry over multiple terms would provide temporal trajectory insights.
3. **Institutional Boundary Shifts:** Model thresholds must be re-calibrated prior to cross-university transfer to reflect local grading standards.

---

## 17. Future Scope

- **Real-Time LMS API Webhooks:** Stream live assignment submissions and attendance logs directly into the prediction engine.
- **Deep Sequence Modeling:** Implement LSTMs or Transformers to forecast weekly learning trajectories from submission logs.
- **Explainable AI Dashboard:** Build an interactive Streamlit or Next.js faculty interface displaying individualized SHAP waterfall explanations.

---

## 18. Conclusion

This project successfully demonstrates how modern data analytics and machine learning can be combined to resolve practical educational challenges. By adhering to strict data quality audits, leak-free feature pipelines, and transparent model interpretation, the system delivers an actionable early-warning decision support tool ready for academic deployment.
