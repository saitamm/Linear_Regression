# 📈 Linear Regression

A hands-on implementation of Linear Regression using Python and scikit-learn, demonstrated on a student performance dataset.

---

## 📌 What is Linear Regression?

Linear Regression is a supervised machine learning algorithm that models the relationship between a **dependent variable (y)** and one or more **independent variables (X)** by fitting a straight line:

```
y = mX + b
```

Where:
- `y` = predicted value
- `X` = input features
- `m` = coefficient (slope)
- `b` = intercept

It is commonly used for **predicting continuous numerical values** such as prices, scores, or grades.

---

## 🧪 Dataset Used

A student lifestyle dataset containing behavioral features such as screen time, sleep hours, and attendance, used to predict semester CGPA.

> This dataset is used purely for demonstration purposes. A dedicated CGPA prediction project with full analysis will be available in a separate repository.

---

## 🛠️ Tech Stack

- Python 3.11
- scikit-learn
- pandas
- numpy
- seaborn
- matplotlib

---

## 🚀 How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/saitamm/Linear_Regression.git
   cd Linear_Regression
   ```

2. Install dependencies:
   ```bash
   pip install scikit-learn pandas numpy seaborn matplotlib
   ```

3. Open the notebook:
   ```bash
   jupyter notebook linear_regression.ipynb
   ```

---

## 🔄 Workflow

1. Load and prepare the dataset
2. Create feature matrix **X** and target vector **y**
3. Split into training and testing sets
4. Train the Linear Regression model
5. Evaluate using MSE and R² score
6. Visualize actual vs predicted values

---

## 📊 Key Concepts Covered

- What is Linear Regression and when to use it
- Difference between training and testing data
- How to interpret coefficients and intercept
- Model evaluation metrics: **MSE** and **R²**
- Visualizing predictions with scatter plots

---
