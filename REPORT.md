# Alzheimer Detection – Final Report

## 1. Wprowadzenie
- Cel projektu: wykrywanie choroby Alzheimera na podstawie danych demograficznych, klinicznych i behawioralnych.

## 2. Dane
- Źródło: plik `alzheimers_disease_data.csv` z 2149 pacjentami (ID 4751–6900).
- Krótkie podsumowanie zmiennych (demografia, styl życia, historia medyczna, wyniki badań, oceny poznawcze i funkcjonalne).

## 3. Preprocessing i inżynieria cech
- Konwersja kategorii (`category`), one-hot, ordinal, standaryzacja.
- Nowe cechy:
  - `comorbidity_score` – suma chorób współistniejących.
  - `age_group` – binning wieku w przedziały 60–69, 70–79, 80–90.
- Obsługa braków i outlierów.

## 4. Eksploracyjna analiza danych (EDA)
- Rozkłady cech numerycznych: wykresy histogramów + KDE.
- Macierz korelacji: słabe zależności między większością cech.
- Boxploty i countploty: różnice w `Age`, `MMSE`, `FunctionalAssessment` itp. pomiędzy zdrowymi a chorymi.

## 5. Modelowanie i dobór
- Pipeline: preprocessing + model.
- Modele:
  1. **Logistic Regression** (baseline)
  2. **Random Forest** (GridSearchCV)
  3. **XGBoost** (GridSearchCV)
- Walidacja krzyżowa wewnątrz GridSearch, split 80/20 z `stratify=y`.

## 6. Ewaluacja
- **Logistic Regression:**  
  - Acc ≈ 0.82, AUC ≈ 0.89, F1(AD) ≈ 0.75.
- **Random Forest:**  
  - Acc ≈ 0.93, AUC ≈ 0.94, F1(AD) ≈ 0.90.
- **XGBoost:**  
  - **Acc ≈ 0.94, AUC ≈ 0.944, F1(AD) ≈ 0.92** (najlepszy).

  ![ROC Comparison](#)  

- Macierze pomyłek i raporty classification_report.

## 7. Istotność cech
- **XGBoost feature_importances_:**  
  1. `MemoryComplaints_1`  
  2. `BehavioralProblems_1`  
  3. `FunctionalAssessment`, `MMSE`, `ADL`  
  4. Inne (cholesterol, dieta, BMI).
- **SHAP:**  
  - Bar plot i dot summary confirmują kierunek wpływu:
    - Niskie wyniki testów funkcj. i poznawczych → zwiększają ryzyko AD.
    - Obecność memory complaints/behavioral problems → silny dodatni wpływ.

  ![SHAP Summary](#)  

## 8. Wnioski
- Najsilniejsze wskaźniki: subiektywne zgłoszenia (pamięć, zachowanie) + obiektywne testy (MMSE, ADL, FunctionalAssessment).
- Cechy kliniczne i biochemiczne mają mniejszy, ale istotny wpływ.
- Modele drzewiaste (RF, XGB) wyraźnie przewyższają regresję liniową.
- XGBoost to rekomendowany model do wdrożenia.

## 9. Ograniczenia i usprawnienia
- Dane pochodzą z jednego źródła – wymagana walidacja na zewnętrznych zbiorach.
- Outliery i brakujące dane można obsłużyć bardziej zaawansowanie.
- Możliwość dodania interakcji cech czy zmiennych temporalnych.
- Rozważenie modeli sekwencyjnych (RNN) dla longitudinalnych danych.