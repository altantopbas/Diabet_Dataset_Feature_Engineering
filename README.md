# 🧪 Diabetes Dataset - Feature Engineering Project

This project focuses on comprehensive **data preprocessing** and **feature engineering** applied to a diabetes dataset. The goal is to prepare the dataset for machine learning models and improve model performance by creating meaningful new features.

## 🔍 Project Objectives

- Identify and handle missing values  
- Detect and treat outliers  
- Separate variable types (numerical, categorical, cardinal)  
- Generate new features (e.g., BMI classification, age groups, interaction terms)  
- Obtain a clean, interpretable dataset suitable for modeling

## 📁 Project Structure

- `diabetes.csv` → Raw dataset used in the project  
- `diabets_feature_engineering.py` → Main script for reading, analyzing, preprocessing data, and feature generation  

## ⚙️ Feature Engineering Techniques Used

- **Missing Value Handling:** Columns containing zeros are treated as missing where appropriate  
- **Outlier Detection:** IQR (Interquartile Range) method used to identify and cap/floor outliers  
- **New Feature Generation:**  
  - Age categorization (`young`, `mature`, `senior`)  
  - BMI classification (`Underweight`, `Normal`, `Overweight`, `Obese`)  
  - Interaction features like `Age * Glucose`  
  - `Insulin Score` category derived from insulin levels

## 🛠️ Libraries Used

- `pandas`  
- `numpy`  
- `seaborn`  
- `matplotlib`
- `sckit-learn`

## 🧠 Notes

This project demonstrates how effective feature engineering can improve machine learning performance. While modeling is not covered in this project, all transformations are designed to support future model development.

## 📌 Source Dataset

The dataset is based on the [Pima Indians Diabetes Database](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database) available on Kaggle.

## ✍️ Author

**Altan Topbaş**  
LinkedIn: [linkedin.com/in/altantopbas](https://www.linkedin.com/in/altantopbas/)  
GitHub: [github.com/altantopbas](https://github.com/altantopbas)  
Kaggle: [kaggle.com/altantopbas](https://www.kaggle.com/altantopbas) 

---

⭐ Feel free to star this repository if you find it helpful!
