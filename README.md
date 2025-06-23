# Elevate_Labs_Intern
This is a repository created to store works and skills gained through internship

## 🚢 Titanic Survival Prediction – ML Classification

This project aims to predict passenger survival on the Titanic using machine learning models. The dataset is preprocessed, cleaned, and then tested across three models to evaluate performance.

### 📂 Dataset

* **Source**: `Titanic-Dataset.csv`
* **Target Variable**: `Survived` (0 = No, 1 = Yes)

### 🔧 Preprocessing Steps

1. **Column Selection**: Removed irrelevant columns (`PassengerId`, `Name`, `Ticket`)
2. **Missing Value Imputation**:

   * **Numerical**: Mean
   * **Categorical**: Most Frequent
3. **Encoding**: Used `OrdinalEncoder` for categorical features
4. **Outlier Removal**: Applied IQR method to numerical columns
5. **Scaling**: Standardized numerical data using `StandardScaler`

### 🧠 Models Evaluated

1. **Decision Tree Classifier**

   * Criterion: Entropy
   * Max Depth: 4

2. **Random Forest Classifier**

   * Criterion: Entropy
   * Max Depth: 4
   * Estimators: 200
   * Min Samples Split: 4
   * Min Samples Leaf: 2

3. **MLPClassifier (Neural Network)**

   * Activation: ReLU
   * Solver: Adam
   * Hidden Layers: (10, 5)
   * Max Iterations: 500

### 📊 Accuracy Scores

| Model         | Accuracy (%) |
| ------------- | ------------ |
| Decision Tree | \~ 81.03     |
| Random Forest | \~ 80.49     |
| MLPClassifier | \~ 78.73     |


### 🗃 Requirements

```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

### 📎 Notes

* Outlier removal helped improve generalization.
* RandomForest is more robust than a single DecisionTree.
* MLP is suitable for larger dataset and is more generalised
---

