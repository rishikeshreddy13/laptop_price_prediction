#💻 Laptop Price Predictor using Machine Learning

An intelligent machine learning application that predicts the **fair market price of laptops** based on their hardware specifications. Unlike traditional pricing methods that rely on model numbers, LaptopLens evaluates the complete hardware configuration ("Hardware DNA") to estimate the actual value of a laptop.

The application also analyzes the user's asking price and classifies it as:

- 🟢 Good Deal
- 🟡 Fair Price
- 🔴 Overpriced

Additionally, it provides hardware insights and upgrade suggestions.

---

## 🚀 Features

- Predict laptop prices using Machine Learning
- Hardware-based valuation instead of model-based pricing
- Fair Price confidence range
- Deal Verdict (Good Deal / Fair Price / Overpriced)
- Feature Engineering using hardware specifications
- Automatic PPI (Pixels Per Inch) calculation
- Memory decomposition (SSD/HDD/Hybrid/Flash)
- Upgrade recommendations
- Hardware insights
- Interactive Streamlit dashboard
- Real-time predictions

---

## 🧠 Machine Learning Pipeline

The project uses a **Stacking Ensemble Regressor** consisting of:

- Random Forest Regressor
- XGBoost Regressor
- Extra Trees Regressor

Meta Learner:

- Linear Regression

Target Variable:

- Log Transformation (`np.log`) to reduce price skewness and improve prediction accuracy.

---

## 📊 Feature Engineering

Custom engineered features include:

- Pixels Per Inch (PPI)
- Touchscreen Detection
- IPS Display Detection
- CPU Category Extraction
- Memory Decomposition
    - SSD
    - HDD
    - Hybrid
    - Flash Storage
- Screen Resolution Parsing
- Weight Cleaning
- RAM Cleaning

---

## 🛠 Tech Stack

### Programming Language
- Python 3.x

### Machine Learning
- Scikit-Learn
- XGBoost

### Data Processing
- Pandas
- NumPy

### Visualization
- Matplotlib
- Seaborn

### Deployment
- Streamlit
- Joblib

---

## 📂 Project Structure

```
Laptop-Price-Predictor/
│
├── app.py
├── pipe.joblib
├── df.joblib
├── laptop_data.csv
├── requirements.txt
├── README.md
│
├── notebooks/
│
├── images/
│
└── assets/
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/rishikeshreddy13/laptop_price_prediction.git
```

Move into the project directory

```bash
cd laptop_price_prediction
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the Streamlit application

```bash
streamlit run app.py
```

---

## 📥 Input Features

The application accepts the following specifications:

- Brand
- Laptop Type
- Screen Size
- Screen Resolution
- Touchscreen
- IPS Panel
- CPU Brand
- RAM
- HDD
- SSD
- GPU Brand
- Operating System
- Weight
- Asking Price (Optional)

---

## 📤 Output

The application provides:

- Predicted Fair Market Price
- Confidence Range
- Deal Verdict
- Hardware Summary
- Feature Insights
- Upgrade Suggestions
- Price Comparison Chart

---

## 📈 Workflow

1. Data Cleaning
2. Feature Engineering
3. Data Preprocessing
4. Model Training
5. Stacking Ensemble
6. Model Serialization
7. Streamlit Deployment
8. Real-Time Prediction

---

## 📊 Model Architecture

```
Raw Data
     │
     ▼
Data Cleaning
     │
     ▼
Feature Engineering
(PPI, SSD/HDD, CPU, IPS)
     │
     ▼
ColumnTransformer
     │
     ▼
Stacking Regressor
 ├── Random Forest
 ├── XGBoost
 └── Extra Trees
     │
     ▼
Linear Regression
     │
     ▼
Price Prediction
     │
     ▼
Deal Evaluation
```

---

## 🎯 Objectives

- Predict fair laptop prices accurately
- Handle customized laptop configurations
- Reduce pricing ambiguity in the second-hand market
- Provide buyers with deal recommendations
- Improve transparency in laptop valuation

---

## 📸 Application Preview

The application dashboard includes:

- Hardware configuration sidebar
- Estimated fair price
- Confidence interval
- Deal verdict
- Hardware insights
- Upgrade suggestions
- Price comparison visualization

---

## 🔮 Future Improvements

- Real-time price scraping from e-commerce websites
- Battery health estimation
- Warranty-aware pricing
- Deep Learning models
- Multi-currency support
- Used laptop condition analysis

---

## 📚 Libraries Used

- pandas
- numpy
- scikit-learn
- xgboost
- matplotlib
- seaborn
- streamlit
- joblib

---

## 👨‍💻 Author

**Rishikesh Reddy**

GitHub:
https://github.com/rishikeshreddy13

---

## ⭐ If you like this project

Give the repository a ⭐ on GitHub.
