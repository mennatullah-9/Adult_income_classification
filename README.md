# Adult Income Classification — Machine Learning Project

## 📌 Project Overview

This project focuses on predicting whether a person's annual income is **greater than $50K** or **less than or equal to $50K** using the Adult Income dataset.

The project covers the main stages of a machine learning classification workflow:

- Exploratory Data Analysis (EDA)
- Data cleaning and preprocessing
- Handling missing values
- Removing duplicate records
- Encoding categorical features
- Feature scaling
- Train/test splitting with stratification
- Outlier and skewness investigation
- Training multiple classification models
- Hyperparameter tuning using GridSearchCV
- Evaluating class imbalance handling techniques
- Comparing model performance

---

## 📊 Dataset

The dataset contains demographic and employment-related information.

### Features

| Feature | Description |
|---|---|
| `age` | Age of the individual |
| `workclass` | Type of employment |
| `fnlwgt` | Final sampling weight |
| `education` | Education level |
| `education-num` | Numerical representation of education |
| `marital-status` | Marital status |
| `occupation` | Occupation |
| `relationship` | Relationship status |
| `race` | Race |
| `sex` | Sex |
| `capital-gain` | Capital gains |
| `capital-loss` | Capital losses |
| `hours-per-week` | Hours worked per week |
| `native-country` | Country of origin |
| `income` | Target variable |

### Target Distribution

The target variable is imbalanced:

- `<=50K`: approximately **76%**
- `>50K`: approximately **24%**

Therefore, the `>50K` class was treated as the minority class during evaluation.

---

## 🔍 Exploratory Data Analysis

EDA was performed to understand the structure and characteristics of the dataset.

The analysis included:

- Dataset shape and information
- Missing-value analysis
- Duplicate detection
- Target-class distribution
- Unique-value analysis
- Histograms for numerical features
- Box plots for numerical features
- Outlier investigation using the IQR method

EDA helped identify skewed numerical variables and understand the class imbalance before modeling.

---

## 🧹 Data Preprocessing

### 1. Train/Test Split

The data was split into training and testing sets using **stratification** to preserve the target-class distribution.

### 2. Missing Values

Missing values in categorical features were handled using:

```python
SimpleImputer(strategy="most_frequent")
```

The imputer was fitted only on the training data and then applied to the test data.

### 3. Categorical Encoding

Categorical variables were converted into numerical representations using:

```python
OneHotEncoder(handle_unknown="ignore")
```

### 4. Feature Scaling

Numerical features used by scale-sensitive models were standardized using `StandardScaler`.

Scaling was particularly relevant for:

- Logistic Regression
- KNN
- SVM

Tree-based models such as Decision Trees, Random Forest, and XGBoost do not require feature scaling.

### 5. Feature Selection

The `education` column was removed because `education-num` provides a numerical representation of the same educational information.

`fnlwgt` was experimentally evaluated and showed almost no difference in Logistic Regression performance, so its removal did not meaningfully change the results.

### 6. Skewness and Outliers

`capital-gain` and `capital-loss` were highly skewed, so a `log1p` transformation was experimentally tested.

The transformation did not provide a meaningful improvement in the final model performance, so the original representation was retained for the final model.

Outliers were investigated using the IQR method. We did not automatically remove all detected outliers because some extreme values can represent valid observations rather than errors.

---

## 🤖 Models Evaluated

The following classification algorithms were tested:

1. Logistic Regression
2. K-Nearest Neighbors (KNN)
3. Support Vector Machine (SVM)
4. Decision Tree
5. Random Forest
6. XGBoost

---

## 📈 Model Results

The main models were evaluated using:

- Accuracy
- Precision for `>50K`
- Recall for `>50K`
- F1-score for `>50K`

### Baseline / Main Results

| Model | Accuracy | Precision `>50K` | Recall `>50K` | F1 `>50K` |
|---|---:|---:|---:|---:|
| **XGBoost** | **88.08%** | **0.79** | **0.69** | **0.74** |
| Tuned Random Forest | 87.09% | 0.79 | 0.64 | 0.70 |
| Logistic Regression | 85.57% | 0.74 | 0.62 | 0.67 |
| SVM | 85.51% | 0.74 | 0.61 | 0.67 |
| KNN | 84.83% | 0.71 | 0.63 | 0.67 |
| Tuned Decision Tree | 86.45% | 0.80 | 0.59 | 0.68 |

> **Note:** Results are based on the final train/test split used in the project.

---

## 🌳 Decision Tree Tuning

GridSearchCV was used to tune the Decision Tree.

### Best Parameters

```text
criterion = gini
max_depth = 12
min_samples_leaf = 12
```

### Results

```text
Cross-Validation Accuracy = 85.58%
Test Accuracy = 86.45%
```

The tuned tree significantly reduced overfitting compared with the original unpruned tree.

---

## 🌲 Random Forest Tuning

GridSearchCV was also used to optimize the Random Forest.

### Best Parameters

```text
max_depth = None
max_features = sqrt
min_samples_leaf = 2
n_estimators = 50
```

### Results

```text
Baseline Test Accuracy = 84.97%
Tuned Test Accuracy = 87.09%
```

The tuned Random Forest improved generalization compared with the baseline forest.

---

## 🚀 XGBoost Tuning

GridSearchCV was used to search for better XGBoost hyperparameters.

### Best Parameters

```text
subsample = 1.0
n_estimators = 400
max_depth = 8
learning_rate = 0.1
colsample_bytree = 0.7
```

### Tuned XGBoost

```text
Test Accuracy ≈ 87.60%
F1 (>50K) ≈ 0.73
```

The original XGBoost configuration achieved **88.08% test accuracy** and **0.74 F1-score** for the `>50K` class, so the original configuration was retained as the better overall model.

---

## ⚖️ Class Imbalance Experiments

Because the target variable was imbalanced, multiple approaches were tested with XGBoost.

### Results

| Method | Accuracy | Precision `>50K` | Recall `>50K` | F1 `>50K` |
|---|---:|---:|---:|---:|
| **Original XGBoost** | **88.08%** | **0.79** | 0.69 | **0.74** |
| SMOTE | 86.02% | 0.69 | 0.75 | 0.72 |
| `scale_pos_weight` | 84.04% | 0.62 | 0.87 | 0.72 |
| Random Undersampling | 82.74% | 0.60 | 0.88 | 0.71 |

### Conclusion

SMOTE, class weighting, and undersampling increased the recall of the minority class, but they reduced other important metrics.

Since the objective was to achieve a good balance between accuracy, precision, recall, and F1-score, the **original XGBoost model** was selected as the final model.

---

## 🏆 Final Model

### XGBoost

**Test Accuracy: 88.08%**

Classification performance:

```text
              precision    recall    f1-score

<=50K            0.91       0.94       0.92
>50K             0.79       0.69       0.74

Accuracy                              0.88
```

### Confusion Matrix

```text
[[4647  293]
 [ 483 1085]]
```

This means:

- **4647** `<=50K` samples were correctly classified.
- **293** `<=50K` samples were incorrectly classified as `>50K`.
- **483** `>50K` samples were incorrectly classified as `<=50K`.
- **1085** `>50K` samples were correctly classified.

---

## 🔎 Key Findings

- **XGBoost achieved the best overall performance** among the evaluated models.
- Tree-based ensemble methods performed better than the simpler baseline models on this tabular dataset.
- Hyperparameter tuning improved both the Decision Tree and Random Forest.
- The original XGBoost model performed better overall than its tuned and imbalance-handled alternatives.
- Class imbalance was investigated rather than ignored.
- SMOTE, `scale_pos_weight`, and undersampling increased minority-class recall but did not improve the overall balance of metrics.
- Log transformation was tested for highly skewed financial features but did not produce a meaningful improvement.
- Outliers were investigated rather than automatically removed, since extreme values may represent valid observations.

---


## 👥 Team

| Team Member | Role |
|---|---|
| Menna Mahmoud | Developer & Team Lead |
| Shaza Mohamed | Developer |
| Menna Ahmed | Developer |
=======

## 🛠️ Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Imbalanced-learn
- Jupyter Notebook / Google Colab

---

## 📁 Project Structure

```text
Adult-Income-Classification/
│
├── adult.data
├── adult.names
├── notebook.ipynb
├── README.md
└── images/
    ├── target_distribution.png
    ├── confusion_matrix.png
    └── feature_importance.png
```

## 👩‍💻 Project Goal

The goal of this project is to build and compare machine learning classification models capable of predicting whether an individual's annual income is above or below $50K, while applying a complete machine learning workflow from exploratory data analysis to model evaluation and hyperparameter tuning.
