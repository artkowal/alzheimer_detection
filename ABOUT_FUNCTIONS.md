# ABOUT_FUNCTIONS.md

## **Description of Functions and Stages of the alzheimer_detection Project**

---

## 1. **src/data_import.py**

### What does this file do?
- Loads data from a CSV file into a Pandas DataFrame.
- Checks data types and the number of missing values.

### Why do we do this?
- To start the analysis, we need data in the correct format.
- Checking types and missing values helps prepare the data for further processing.

---

## 2. **src/feature_engineering.py**

### What does this file do?
- Creates new features to enrich the dataset:
    - `create_comorbidity_score`: calculates the total number of comorbidities.
    - `create_age_group`: creates age groups based on the age column.

### Why do we do this?
- Such new features can better describe disease risk than raw variables.
- Grouping age simplifies analysis and can reveal nonlinear dependencies.

---

## 3. **src/preprocessing.py**

### What does this file do?
- Converts categorical variables to the `category` type in Pandas.
- Builds a data processing pipeline:
    - encodes categorical features (one-hot, ordinal),
    - scales numeric features,
    - handles new features (e.g., `comorbidity_score`, `age_group`).

### Why do we do this?
- Machine learning models require numeric, well-prepared data.
- Standardization and encoding ensure all features are on a comparable scale.

---

## 4. **src/model.py (train_baseline_model)**

### What does this file do?
- Creates a baseline model (logistic regression) and pipeline.
- Splits the data into training and test sets.
- Trains the model, makes predictions on the test set, and calculates metrics: accuracy, ROC AUC, confusion matrix.
- Plots the ROC curve.

### Why do we do this?
- The baseline model shows how a simple method works—it is a point of reference for more advanced solutions.
- Allows us to assess whether the data contains enough information for prediction.

---

## 5. **src/random_forest_model.py (train_random_forest)**

### What does this file do?
- Builds a pipeline with RandomForestClassifier.
- Performs GridSearchCV to select the best hyperparameters.
- Presents model metrics, confusion matrix, and ROC curve.

### Why do we do this?
- Random Forest often yields better results than simple linear models.
- Hyperparameter tuning allows for optimal model performance.

---

## 6. **src/xgboost_model.py (train_xgboost)**

### What does this file do?
- Creates a pipeline with XGBoost and runs GridSearchCV.
- Selects the best parameters.
- Displays metrics, confusion matrix, and ROC curve.

### Why do we do this?
- XGBoost is an advanced ensemble learning algorithm that often achieves top results on tabular data.
- Cross-validation helps prevent overfitting.

---

## 7. **src/evaluate_models.py**

### What does this file do?
- Compares several models (Logistic Regression, Random Forest, XGBoost) using the same data split.
- For each model, computes: accuracy, ROC AUC, precision, recall, f1-score.
- Plots ROC curve comparison on a single plot.

### Why do we do this?
- Makes it easy to see which model performs best on our data.
- Facilitates the choice of the best solution for deployment.

---

## 8. **src/feature_importance_xgb.py**

### What does this file do?
- Calculates and presents feature importance for XGBoost.
- Displays a table of the most important features and a bar chart.

### Why do we do this?
- Feature importance analysis helps us understand which variables most influence the model’s decisions.
- Can suggest which features to include (or drop) in future analyses.

---

## 9. **src/shap_analysis.py**

### What does this file do?
- Calculates SHAP values for the XGBoost model on the test set.
- Generates bar and dot summary plots showing the impact of features on predictions.

### Why do we do this?
- SHAP allows us to interpret model decisions even with complex algorithms.
- Shows not only importance but also the **direction of the feature’s impact** on risk.

---

## 10. **EDA (Exploratory Data Analysis, e.g., eda.ipynb)**

### Data loading and initial analysis
- Loads the data, shows headers, types, and number of missing values.

### Basic statistics of numeric features
- Computes descriptive statistics for numeric features.
- Draws histograms and density plots.
- **Why?**: Helps understand data distribution, identify outliers, and see general trends.

### Correlations of numeric features
- Plots the correlation matrix (heatmap).
- **Why?**: Shows relationships between variables, helps detect redundant features.

### Boxplots for selected features by diagnosis
- Compares distributions of age, MMSE, and other features in healthy vs. diseased groups.
- **Why?**: Checks if features differ significantly between groups.

### Barplots for binary/categorical variables
- Shows the count of different groups by diagnosis.
- **Why?**: Makes it easier to analyze differences, e.g., the frequency of comorbidities between groups.

---

# **Summary**
Each file and function in the project corresponds to a specific stage of the classic data science process—from loading and preparing data, through feature engineering, exploration, and visualization, to modeling and interpretation of results.  
Each step is designed to maximize prediction quality and enable understanding of **why** the model makes particular decisions.

---
