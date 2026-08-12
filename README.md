# Diabetes-Prediction-Model
ML project predicting diabetes risk using Models like Decision Trees, Logistic Regression, and Random Forest, and using SMOTE for class imbalance
# 🩺 Diabetes Prediction Model

> Internship Project | Machine Learning — Binary Classification | Individual Submission

A machine learning project that predicts whether a patient has diabetes based on diagnostic health measurements, using the **Pima Indians Diabetes Dataset**. The project covers the full pipeline — data cleaning, EDA, handling class imbalance, training multiple models, and comparing them using precision/recall/F1 rather than accuracy alone.

---

## 📑 Table of Contents
1. [Introduction](#introduction)
2. [Objectives](#objectives)
3. [Technology Stack](#technology-stack)
4. [Dataset](#dataset)
5. [Project Workflow](#project-workflow)
6. [Handling Class Imbalance](#handling-class-imbalance)
7. [Model Comparison Results](#model-comparison-results)
8. [Key Insights](#key-insights)
9. [Setup & Installation](#setup--installation)
10. [Running the Project](#running-the-project)
11. [Author](#author)

---

## 📖 Introduction

Diabetes is a widespread chronic condition, and early prediction from routine diagnostic measurements can support timely intervention. This project builds and compares several classification models — **Logistic Regression, Decision Tree, and Random Forest** — to predict diabetes outcomes, with a particular focus on **handling class imbalance**, since correctly identifying actual diabetic patients (recall) matters more than raw accuracy in a medical screening context.

---

## 🎯 Objectives

- Clean and prepare a real-world medical dataset (handle disguised missing values)
- Perform exploratory data analysis to understand feature relationships
- Train and evaluate multiple classification models
- Address class imbalance using **class weighting** and **SMOTE**
- Compare models using precision, recall, F1-score, and confusion matrices — not just accuracy
- Identify which features most influence diabetes prediction

---

## 🛠️ Technology Stack

- **Language:** Python 3
- **Environment:** Google Colab / Jupyter Notebook
- **Libraries:** pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn

---

## 📊 Dataset

- **Source:** [Pima Indians Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) (Kaggle)
- **Rows:** 768 | **Features:** 8 diagnostic measurements + target (`Outcome`)
- **Target:** `Outcome` — 1 = diabetic, 0 = non-diabetic

**Note:** Several columns (`Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`) use `0` as a placeholder for missing data, since a real value of 0 is not medically possible for these measurements. These were treated as missing and imputed with the column median.

---

## 🔄 Project Workflow

1. **Data Cleaning** — replaced disguised missing values (zeros) with `NaN`, then imputed using median
2. **EDA** — class balance check, correlation heatmap to understand feature relationships
3. **Train/Test Split** — 80/20 split, **stratified** on the target to preserve class ratio in both sets
4. **Baseline Models** — Logistic Regression, Decision Tree, Random Forest (no imbalance handling)
5. **Imbalance Handling** — retrained models using `class_weight='balanced'` and **SMOTE** (oversampling applied to training data only)
6. **Evaluation** — confusion matrix + classification report (precision, recall, F1) on the untouched test set for every model
7. **Feature Importance** — visualized which features the Decision Tree relied on most

---

## ⚖️ Handling Class Imbalance

The dataset is moderately imbalanced (~65% non-diabetic, ~35% diabetic). In a medical screening context, a **false negative** (telling a diabetic patient they're healthy) is a more costly mistake than a **false positive**, so this project specifically optimizes for **recall on the diabetic class**, not just overall accuracy.

Two techniques were tested and compared:
- **`class_weight='balanced'`** — penalizes misclassifying the minority class more heavily during training
- **SMOTE (Synthetic Minority Over-sampling Technique)** — generates synthetic diabetic-class samples in the training set only, so the model sees a more balanced class distribution

---

## 📈 Model Comparison Results

<!-- PLACEHOLDER: paste your final run's numbers here -->

| Model | Train Accuracy | Test Accuracy | Class 1 Precision | Class 1 Recall | Class 1 F1 |
|---|---|---|---|---|---|
| Logistic Regression (Plain) | 79.32% | 70.13% | 0.59 | 0.50 | 0.54 |
| Decision Tree (Plain) | 81.43% | 75.97% | 0.64 | 0.72 | 0.68 |
| Decision Tree (Balanced) | 83.88% | 72.73% | 0.61 | 0.61 | 0.61 |
| Random Forest (Plain) | 84.53% | 72.08% | 0.63 | 0.50 | 0.56 |
| Decision Tree (SMOTE) | 84.25% | 75.97% | 0.63 | 0.74 | 0.68 |
| Logistic Regression (SMOTE) | 74.38% | 71.43% | 0.58 | 0.67 | 0.62 |
| Decision Tree (SMOTE + Custom Weights) | 82% | 72.08% | 0.57 | 0.83 | 0.68 |
| Random Forest (SMOTE) | 86.5% | 75.32% | 0.62 | 0.78 | 0.69 |

<!-- PLACEHOLDER: optionally paste a confusion matrix / feature importance screenshot below -->

---

## 💡 Key Insights

- **SMOTE + custom class weighting achieved the highest recall (0.83)** on the diabetic class, making it the strongest choice when minimizing missed diagnoses is the top priority — at the cost of lower precision (0.57).
- **`class_weight='balanced'`** alone underperformed plain SMOTE** (recall 0.61 vs 0.72), showing that imbalance-handling techniques don't always help and should be validated empirically rather than assumed.
- **Random Forest (SMOTE) gave the best overall balance**, with the highest F1-score (0.69) and strong recall (0.78), making it the best all-around candidate for a single production model.

---

## ⚙️ Setup & Installation

```bash
git clone https://github.com/<your-username>/Diabetes-Prediction-Model.git
cd Diabetes-Prediction-Model
pip install -r requirements.txt
```

**requirements.txt**
```
pandas
numpy
scikit-learn
imbalanced-learn
matplotlib
seaborn
```

---

## ▶️ Running the Project

1. Open `Diabates_Model.ipynb` in Jupyter Notebook, VS Code, or Google Colab
2. Ensure `diabetes.csv` is in the same directory
3. Run all cells top to bottom (`Runtime → Restart and run all` in Colab)

---

## 👤 Author

**Diptangshu Das**
BCA Student, Heritage Academy, Kolkata
