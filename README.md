# Alzheimer Detection

## Project Overview
"Alzheimer Detection" is a machine learning project aimed at detecting Alzheimer's Disease based on patient data, including demographic details, lifestyle factors, medical history, clinical measurements, and cognitive assessments. The goal is to build and evaluate predictive models that can support early diagnosis.

## Repository Structure

```bash
alzheimer_detection/ 
├── data/ # Raw and processed data files 
├── notebooks/ # Jupyter notebooks for EDA and prototyping 
├── src/ # Source code modules 
│ ├── init.py 
│ └── data_import.py # Script for loading and initial preprocessing 
├── tests/ # Unit tests 
├── requirements.txt # Project dependencies 
└── README.md # Project overview and setup instructions
```

## Branching Strategy
- **main**: Stable branch containing the production-ready code. Only merged changes from `develop` after review and testing.
- **develop**: Active development branch where new features, bug fixes, and experiments are committed. Once ready, merge into `main`.

## Setup and Installation
**Clone the repository**
```bash
git clone https://github.com/artkowal/alzheimer_detection.git
cd alzheimer_detection
```
**Intall dependencies**
```bash
pip install -r requirements.txt
```