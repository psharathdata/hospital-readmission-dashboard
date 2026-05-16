# Hospital Patient Readmission Analysis & Prediction

End-to-end healthcare analytics project focused on identifying patients at high risk of hospital readmission using Machine Learning, Data Analysis, and Interactive Dashboards.

---

# Business Problem

Hospital readmissions increase healthcare costs and reduce operational efficiency.  
This project analyzes patient data to identify the major factors contributing to readmissions and predicts patients who are likely to be readmitted within 30 days.

---

# Objectives

- Analyze patient admission and readmission trends
- Identify high-risk patients
- Build predictive Machine Learning models
- Generate actionable healthcare insights
- Create interactive dashboards for hospital management

---

# Tools & Technologies

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- SQL
- Tableau
- Streamlit
- SHAP
- Matplotlib
- Plotly

---

# Dataset

Dataset used: Hospital Readmission Dataset (`diabetic_data.csv`)

The dataset contains:
- Patient demographics
- Admission details
- Medical specialties
- Diagnosis information
- Medication records
- Readmission status

---

# Project Workflow

## 1. Data Cleaning
- Missing value handling
- Duplicate removal
- Feature engineering
- Data preprocessing

## 2. Exploratory Data Analysis (EDA)
- Readmission distribution
- Age-wise patient analysis
- Gender analysis
- Admission type trends
- Medication impact analysis

## 3. Machine Learning
Models Used:
- Logistic Regression
- Random Forest
- XGBoost

## 4. Explainability
- SHAP analysis for feature importance
- High-risk patient identification

## 5. Dashboard Development
- Tableau dashboard
- Streamlit interactive application

---

# Key Insights

- Elderly patients showed higher readmission risk
- Certain admission types had increased readmission rates
- Frequent hospital visits strongly influenced readmission probability
- Specific medications and diagnoses impacted patient outcomes

---

# Machine Learning Performance

| Model | Performance |
|---|---|
| Logistic Regression | Good baseline performance |
| Random Forest | Strong predictive capability |
| XGBoost | Best overall performance |

---

# Features

- Patient readmission prediction
- Risk score generation
- Interactive healthcare dashboard
- SHAP explainability
- Hospital performance insights
- High-risk patient analysis

---

# Dashboard Preview

## Dashboard Includes:
- Total Patients KPI
- Readmission Rate
- High-Risk Patient Count
- Age Group Analysis
- Admission Type Trends
- Risk Distribution
- Interactive Filters

---

# Project Structure

```text
hospital-readmission-analysis/
│
├── data/
│   └── diabetic_data.csv
│
├── notebooks/
│   └── hospital_readmission_analysis.ipynb
│
├── sql/
│   └── hospital_queries.sql
│
├── dashboard/
│   └── tableau_dashboard.png
│
├── outputs/
│   ├── charts/
│   ├── shap_summary.png
│   └── hospital_readmission_with_scores.csv
│
├── app.py
├── requirements.txt
└── README.md
