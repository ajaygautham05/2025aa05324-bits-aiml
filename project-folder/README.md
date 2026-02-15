# Machine Learning Assignment – 2  
## Heart Disease Classification using Machine Learning

---

## a. Problem Statement

The objective of this assignment is to implement, evaluate, and compare multiple machine learning classification models on a real-world healthcare dataset. The task focuses on predicting the presence of heart disease based on clinical attributes. An interactive Streamlit web application is developed to demonstrate model performance and evaluation metrics.

---

## b. Dataset Description (1 Mark)

The dataset used is the Heart Disease Dataset obtained from a public repository (UCI / Kaggle).

- Problem Type: Binary Classification  
- Number of Instances: 1025  
- Number of Features: 13  
- Target Variable: `target`  
  - 1 → Presence of heart disease  
  - 0 → Absence of heart disease  
- Missing Values: None  

The dataset contains patient health attributes such as age, cholesterol levels, blood pressure, and heart rate.

---

## c. Models Used and Evaluation Metrics (6 Marks)

The following six machine learning classification models were implemented on the same dataset and evaluated using standard performance metrics.

### Evaluation Metrics
- Accuracy  
- AUC Score  
- Precision  
- Recall  
- F1 Score  
- Matthews Correlation Coefficient (MCC)  

### Model Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|--------------|---------|-----|----------|--------|----|-----|
| Logistic Regression | 0.870 | 0.939 | 0.851 | 0.905 | 0.877 | 0.941 |
| Decision Tree | 0.860 | 0.937 | 0.836 | 0.905 | 0.869 | 0.723 |
| kNN | 0.815 | 0.916 | 0.822 | 0.816 | 0.819 | 0.630 |
| Naive Bayes | 0.841 | 0.914 | 0.804 | 0.911 | 0.855 | 0.687 |
| Random Forest (Ensemble) | 0.906 | 0.959 | 0.891 | 0.930 | 0.910 | 0.812 |
| XGBoost (Ensemble) | 0.974 | 0.992 | 0.987 | 0.962 | 0.974 | 0.948 |

---

## Observations on Model Performance (3 Marks)

| ML Model Name | Observation about model performance |
|--------------|-------------------------------------|
| Logistic Regression | Provides a strong baseline performance with good interpretability and balanced evaluation metrics. |
| Decision Tree | Simple and interpretable model but prone to overfitting compared to ensemble methods. |
| kNN | Performs well after feature scaling but has higher computational cost during prediction. |
| Naive Bayes | Fast and efficient with reasonable performance despite strong independence assumptions. |
| Random Forest (Ensemble) | Improves performance by reducing overfitting through ensemble learning. |
| XGBoost (Ensemble) | Achieves the best performance due to gradient boosting and regularization techniques. |

---

## Streamlit Application Features

- CSV dataset upload  
- Machine learning model selection  
- Display of evaluation metrics  
- Confusion matrix visualization  
- Classification report  
- Dataset overview  

---

## Repository Structure


