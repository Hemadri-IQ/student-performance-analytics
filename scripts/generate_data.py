"""
Synthetic Dataset Generator for:
IBM SkillsBuild Data Analytics with AI Internship 2026
BharatCares in association with AICTE
Author: Hemadri
"""

import os
import numpy as np
import pandas as pd

def generate_student_dataset(output_path="data/student_performance.csv", random_seed=42):
    np.random.seed(random_seed)
    n_unique = 2020  # Unique base records
    
    # Generate unique student IDs
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
    family_income = np.random.choice(["Low", "Medium", "High"], size=n_unique, p=[0.30, 0.50, 0.20])
    
    # Academic behaviors
    study_hours = np.round(
        np.clip(np.random.gamma(shape=3.2, scale=1.0, size=n_unique) + 0.4, 0.5, 8.0), 1
    )
    sleep_hours = np.round(
        np.clip(np.random.normal(loc=6.9, scale=1.1, size=n_unique), 4.0, 10.0), 1
    )
    
    # Attendance percentage (correlated with study habits + income + random variation)
    att_base = 56.0 + 3.2 * study_hours + np.random.normal(0, 9.5, size=n_unique)
    attendance_percentage = np.round(np.clip(att_base, 45.0, 100.0), 1)
    
    # Previous score (25 - 98)
    prev_base = 42.0 + 3.0 * study_hours + 0.22 * attendance_percentage + np.random.normal(0, 9.0, size=n_unique)
    previous_score = np.round(np.clip(prev_base, 25.0, 98.0), 1)
    
    # Class participation (10 - 100)
    part_base = 15.0 + 0.45 * attendance_percentage + 3.5 * study_hours + np.random.normal(0, 11.0, size=n_unique)
    class_participation = np.round(np.clip(part_base, 10.0, 100.0), 1)
    
    # Assignment score (20 - 100)
    assign_base = 0.35 * previous_score + 3.5 * study_hours + 0.25 * attendance_percentage + np.random.normal(0, 7.5, size=n_unique)
    assignment_score = np.round(np.clip(assign_base, 20.0, 100.0), 1)
    
    # Midterm score (20 - 100)
    mid_base = 0.38 * previous_score + 0.34 * assignment_score + 0.16 * attendance_percentage + np.random.normal(0, 6.8, size=n_unique)
    midterm_score = np.round(np.clip(mid_base, 20.0, 100.0), 1)
    
    # Final score (20 - 100)
    # Realistic synthesis of academic performance across the term
    final_base = (
        0.30 * midterm_score +
        0.25 * assignment_score +
        0.20 * attendance_percentage +
        0.14 * previous_score +
        1.6 * study_hours +
        0.05 * class_participation -
        0.6 * np.maximum(0, 6.0 - sleep_hours) +
        np.random.normal(0, 4.8, size=n_unique)
    )
    final_score = np.round(np.clip(final_base, 20.0, 100.0), 1)
    
    # Academic Risk target calculation (Derived from early-term indicators, WITHOUT final_score!)
    # Early risk index: combination of attendance, study hours, previous score, assignment, midterm
    risk_index = (
        0.34 * midterm_score +
        0.26 * assignment_score +
        0.22 * attendance_percentage +
        0.12 * previous_score +
        1.5 * study_hours +
        0.04 * class_participation +
        np.random.normal(0, 4.0, size=n_unique)
    )
    
    academic_risk = []
    for score in risk_index:
        if score < 52.0:
            academic_risk.append("High Risk")
        elif score < 72.0:
            academic_risk.append("Moderate Risk")
        else:
            academic_risk.append("Low Risk")
            
    df = pd.DataFrame({
        "student_id": student_ids,
        "gender": genders,
        "age": ages,
        "study_hours_per_day": study_hours,
        "attendance_percentage": attendance_percentage,
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
        "academic_risk": academic_risk
    })
    
    # Introduce small realistic amount of duplicate rows (30 duplicates)
    dup_indices = np.random.choice(df.index, size=30, replace=False)
    dup_rows = df.loc[dup_indices].copy()
    df = pd.concat([df, dup_rows], ignore_index=True)
    
    # Introduce small realistic amount of missing values (NaNs)
    # Numerical missing values
    num_missing_study = np.random.choice(df.index, size=24, replace=False)
    df.loc[num_missing_study, "study_hours_per_day"] = np.nan
    
    num_missing_sleep = np.random.choice(df.index, size=20, replace=False)
    df.loc[num_missing_sleep, "sleep_hours"] = np.nan
    
    num_missing_prev = np.random.choice(df.index, size=18, replace=False)
    df.loc[num_missing_prev, "previous_score"] = np.nan
    
    # Categorical missing values
    cat_missing_parent = np.random.choice(df.index, size=22, replace=False)
    df.loc[cat_missing_parent, "parental_education"] = np.nan
    
    cat_missing_extra = np.random.choice(df.index, size=16, replace=False)
    df.loc[cat_missing_extra, "extracurricular_activity"] = np.nan
    
    # Shuffle dataframe
    df = df.sample(frac=1.0, random_state=random_seed).reset_index(drop=True)
    
    # Ensure destination directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset generated successfully at: {output_path}")
    print(f"Total rows: {len(df)}, Total columns: {len(df.columns)}")
    print("Academic Risk Distribution:")
    print(df["academic_risk"].value_counts())
    print("\nMissing values per column:")
    print(df.isnull().sum()[df.isnull().sum() > 0])
    print(f"\nDuplicates in raw dataset: {df.duplicated().sum()}")
    return df

if __name__ == "__main__":
    generate_student_dataset("data/student_performance.csv", 42)
