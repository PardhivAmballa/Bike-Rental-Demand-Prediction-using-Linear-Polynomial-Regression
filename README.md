# 🚲 Bike Rental Demand Prediction using Linear & Polynomial Regression

Implementation of **Linear Regression** and **Polynomial Regression** from scratch using the **Normal Equation (Moore–Penrose Pseudoinverse)** to predict hourly bike rental demand from the Bike Sharing Demand dataset.

This project was completed as part of the **Optimization Course Project** at **IIIT Bangalore**. The assignment required implementing regression models manually without using scikit-learn's regression algorithms.

---

## 📌 Project Overview

The objective of this project is to predict the total number of rented bikes (`count`) using weather, seasonal, and time-related features.

Instead of relying on machine learning libraries for regression, all model parameters are computed manually using linear algebra and the **Normal Equation with the Moore–Penrose pseudoinverse**.

The project compares:

- Linear Regression (Baseline)
- Polynomial Regression (Degree 2)
- Polynomial Regression (Degree 3)
- Polynomial Regression (Degree 4)
- Quadratic Regression with Interaction Terms

The models are evaluated using **Mean Squared Error (MSE)** and **R² Score** on an **80/20 train-test split**.

---

## ✨ Features

- Manual implementation of Linear Regression
- Normal Equation solved using the Moore–Penrose pseudoinverse
- Polynomial feature generation without interaction terms
- Quadratic feature generation with interaction terms
- Automatic preprocessing pipeline
- One-hot encoding for categorical variables
- Feature normalization using StandardScaler
- Train/Test split without data leakage
- Model evaluation using MSE and R²

---

## 📂 Repository Structure

```text
.
├── q1_bike_regression.py     # Complete implementation
├── train.csv                 # Bike Sharing Demand dataset
└── README.md
````

---

## ⚙️ Technologies Used

* Python 3
* NumPy
* Pandas
* scikit-learn *(Preprocessing & Evaluation only)*

---

## 🧠 Methodology

### Data Preprocessing

The preprocessing pipeline performs:

* Removal of leakage features (`casual`, `registered`)
* Datetime feature extraction:

  * Hour
  * Day
  * Month
  * Year
  * Weekday
* Standardization of numerical features
* One-Hot Encoding of categorical variables
* 80/20 Train-Test Split

The preprocessing transformations are fit only on the training data to avoid train-test leakage.

### Regression Models

The following models are implemented:

* Linear Regression
* Polynomial Regression (Degree 2)
* Polynomial Regression (Degree 3)
* Polynomial Regression (Degree 4)
* Quadratic Regression with Pairwise Interaction Terms

Regression coefficients are computed using:

```text
θ = (XᵀX)⁺Xᵀy
```

where `(XᵀX)⁺` denotes the **Moore–Penrose pseudoinverse**, allowing stable solutions even when the design matrix is singular due to one-hot encoding and polynomial expansion.

---

## 📊 Evaluation Metrics

Each model is evaluated using:

* Mean Squared Error (MSE)
* Coefficient of Determination (R²)

The implementation prints the performance of every regression model on the test set.

---

## ▶️ Running the Project

### Install Dependencies

```bash
pip install numpy pandas scikit-learn
```

### Run the Program

```bash
python q1_bike_regression.py
```

The program reads `train.csv`, preprocesses the data, trains all regression models, and displays their performance on the test set.

---

## 📈 Results

The experiments show that:

* Polynomial models outperform simple Linear Regression.
* Degree-2 Polynomial Regression generally provides the best balance between accuracy and model complexity.
* Higher-degree polynomial models can improve flexibility but are more prone to overfitting.
* Quadratic models with interaction terms sometimes achieve slightly lower error at the cost of increased computational complexity.

---

## 📚 Dataset

**Bike Sharing Demand Dataset**

The dataset contains:

* Weather conditions
* Temperature
* Humidity
* Wind speed
* Seasonal information
* Holiday/Working day indicators
* Date & time information
* Hourly bike rental count

**Target Variable**

```text
count
```

---

## 👥 Contributors

* Amballa Pardhiv
* P. Sai Pramod
* T. Hemanth Reddy

---

## 📄 License

This repository is intended for **educational and academic purposes**.

```
```
