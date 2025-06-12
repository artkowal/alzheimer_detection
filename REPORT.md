# Alzheimer Detection – Final Report

## 1. Introduction
- **Project goal:** Detecting Alzheimer’s disease based on demographic, clinical, and behavioral data.

## 2. Data
- **Source:** The file `alzheimers_disease_data.csv` with 2149 patients (IDs 4751–6900).
- **Brief variable summary:** Demographics, lifestyle, medical history, laboratory results, cognitive and functional assessments.

## 3. Preprocessing and Feature Engineering
- Conversion of categorical features (`category` type), one-hot encoding, ordinal encoding, standardization.
- New features:
  - `comorbidity_score` – total number of comorbidities.
  - `age_group` – binning age into ranges: 60–69, 70–79, 80–90.
- Handling missing values and outliers.

## 4. Exploratory Data Analysis (EDA)
- Distributions of numeric features: histograms and KDE plots.
- Correlation matrix: mostly weak relationships between features.
- Boxplots and countplots: show differences in `Age`, `MMSE`, `FunctionalAssessment`, etc. between healthy and diseased groups.

## 5. Modeling and Selection
- Pipeline: preprocessing + model.
- Models:
  1. **Logistic Regression** (baseline)
  2. **Random Forest** (with GridSearchCV)
  3. **XGBoost** (with GridSearchCV)
- Cross-validation inside GridSearch, 80/20 split with `stratify=y`.

## 6. Evaluation
- **Logistic Regression:**  
  - Acc ≈ 0.82, AUC ≈ 0.89, F1(AD) ≈ 0.75.
- **Random Forest:**  
  - Acc ≈ 0.93, AUC ≈ 0.94, F1(AD) ≈ 0.90.
- **XGBoost:**  
  - **Acc ≈ 0.94, AUC ≈ 0.944, F1(AD) ≈ 0.92** (best result).

- Confusion matrices and classification reports.

## 7. Feature Importance
- **XGBoost feature_importances_:**  
  1. `MemoryComplaints_1`
  2. `BehavioralProblems_1`
  3. `FunctionalAssessment`, `MMSE`, `ADL`
  4. Others (cholesterol, diet, BMI).
- **SHAP:**  
  - Bar and dot summary plots confirm direction of impact:
    - Low scores on functional and cognitive tests → increased AD risk.
    - Presence of memory complaints/behavioral problems → strong positive effect.

## 8. Conclusions
- Strongest indicators: subjective complaints (memory, behavior) + objective tests (MMSE, ADL, FunctionalAssessment).
- Clinical and biochemical features have smaller but still significant impact.
- Tree-based models (RF, XGB) significantly outperform linear regression.
- XGBoost is the recommended model for deployment.

## 9. Limitations and Improvements
- Data comes from a single source – requires validation on external datasets.
- Outliers and missing data could be handled more robustly.
- Potential for adding feature interactions or temporal variables.
- Consider sequential models (RNNs) for longitudinal data.