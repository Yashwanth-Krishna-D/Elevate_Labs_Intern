---

# 🚢 Titanic Survival Prediction – Machine Learning Classification

This project is part of the **Elevate Labs Internship**. It aims to predict passenger survival aboard the Titanic using classic machine learning classification algorithms. The code is modular and includes functions for preprocessing, visualization, and model evaluation.

---

## 📂 Dataset

* **File**: `Titanic-Dataset.csv`
* **Target Feature**: `Survived` (0 = Did not survive, 1 = Survived)

---

## 🔧 Preprocessing Pipeline

The data undergoes several preprocessing steps to ensure quality and model-readiness:

1. **Column Dropping**:

   * Removed: `PassengerId`, `Name`, `Ticket` (non-informative)

2. **Missing Value Handling**:

   * **Numerical**: Imputed using **mean**
   * **Categorical**: Imputed using **most frequent** value

3. **Encoding**:

   * Used `OrdinalEncoder` for categorical variables

4. **Outlier Removal**:

   * Applied **IQR-based filtering** on numerical features

5. **Feature Scaling**:

   * Applied `StandardScaler` for numerical normalization

---

## 📊 Exploratory Data Analysis (EDA)

The following EDA tools are available (optional):

* Box plots for numerical outlier detection
* Histograms with KDE for distribution analysis
* Pair plots with `hue='Survived'`
* Correlation heatmap

---

## 🧠 Models Evaluated

| Model              | Key Parameters                                                                                 |
| ------------------ | ---------------------------------------------------------------------------------------------- |
| Decision Tree      | Criterion: `entropy`, Max Depth: 4                                                             |
| Random Forest      | Criterion: `entropy`, Max Depth: 4, Estimators: 200, Min Samples Split: 4, Min Samples Leaf: 2 |
| MLPClassifier (NN) | Activation: `relu`, Solver: `adam`, Hidden Layers: (10, 5), Max Iter: 500                      |

---

## 📈 Accuracy Results (approx.)

| Model          | Accuracy (%) |
| -------------- | ------------ |
| Decision Tree  | \~79.33      |
| Random Forest  | \~82.00      |
| MLP Classifier | \~82.67      |

---

## 📎 Notes

* Outlier removal noticeably improved performance and generalization.
* Random Forest outperformed single-tree methods in both accuracy and stability.
* MLP Classifier was effective even on a moderately sized dataset, though sensitive to scaling.

---

## 🛠 Requirements

Install the following Python libraries:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

---
