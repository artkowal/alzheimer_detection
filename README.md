# 🧠 Alzheimer Detection
[![Python](https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white)](https://www.python.org/)
[![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-0099CC?style=flat-square&logo=xgboost&logoColor=white)](https://xgboost.ai/)
[![Jupyter](https://img.shields.io/badge/Jupyter-F37626?style=flat-square&logo=jupyter&logoColor=white)](https://jupyter.org/)
[![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)](https://numpy.org/)
[![Matplotlib](https://img.shields.io/badge/Matplotlib-11557c?style=flat-square&logo=matplotlib&logoColor=white)](https://matplotlib.org/)
[![Seaborn](https://img.shields.io/badge/Seaborn-43b7ba?style=flat-square&logo=python&logoColor=white)](https://seaborn.pydata.org/)



## Project Overview
"Alzheimer Detection" is a machine learning project aimed at detecting Alzheimer's Disease based on patient data, including demographic details, 
lifestyle factors, medical history, clinical measurements, and cognitive assessments. 
The goal is to build and evaluate predictive models that can support early diagnosis.

## Project Structure

```bash
alzheimer_detection/
├── data/
│ └── alzheimers_disease_data.csv     # raw data
├── notebooks/
│ └── eda.ipynb                       # data exploration notebook
├── src/
│ ├── data_import.py                  # data loading
│ ├── preprocessing.py                # cleaning & preprocessing pipeline
│ ├── feature_engineering.py          # comorbidity_score, age_group
│ ├── model.py                        # baseline: Logistic Regression
│ ├── random_forest_model.py          # RF + GridSearch
│ ├── xgboost_model.py                # XGBoost + GridSearch
│ ├── evaluate_models.py              # model comparison
│ ├── feature_importance.py           # feature_importances_ from XGB
│ └── shap_analysis.py                # SHAP interpretation
├── REPORT.md                         # final report
├── requirements.txt                  # list of dependencies
└── README.md                         # setup & documentation
```

## Setup and Installation
**Clone the repository**
```bash
git clone https://github.com/artkowal/alzheimer_detection.git
cd alzheimer_detection
```
**Install dependencies**
```bash
pip install -r requirements.txt
```