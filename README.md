# Rossmann Pharmaceuticals Sales Forecasting

This repository contains an end-to-end machine learning project aimed at forecasting daily sales for Rossmann Pharmaceuticals' stores up to six weeks in advance. This project is a highlight of my machine learning portfolio, showcasing skills in exploratory data analysis, machine learning, deep learning, and MLOps.

## Project Overview

### Business Need

Rossmann Pharmaceuticals’ finance team requires an accurate forecasting system to predict daily sales for their stores across various cities. Currently, store managers rely on experience and intuition to make forecasts. By leveraging machine learning, I aim to:

- Provide a data-driven solution for sales prediction.
- Incorporate factors such as promotions, holidays, seasonality, and competition.
- Assist the finance team in better resource allocation and planning.

### Key Features

1. **Exploratory Data Analysis (EDA):** Insights into customer behavior and sales trends.
2. **Machine Learning Modeling:** Predictive analysis using tree-based algorithms and regression models.
3. **Deep Learning Modeling:** LSTM-based time series forecasting.
4. **Model Deployment:** REST API for serving real-time predictions.
5. **MLOps Integration:** Reproducible pipelines, version control, and CI/CD practices.

---

## Data Description

The dataset is sourced from [Kaggle’s Rossmann Store Sales](https://www.kaggle.com/c/rossmann-store-sales). Key features include:

- **Sales:** Target variable representing daily turnover.
- **Store:** Unique identifier for each store.
- **Promotions:** Information about ongoing promotional activities.
- **Competition:** Distance to nearest competitors and their opening periods.
- **Holidays:** State and school holidays.
- **Assortment:** Types of assortments available in stores.

The dataset also includes various temporal features such as weekdays, weekends, and seasonal markers.

---

## Project Workflow

### Task 1: Exploratory Data Analysis (EDA)

- **Objective:** Understand customer purchasing behavior and the impact of features such as promotions, holidays, and competition.
- **Key Deliverables:**
  - Data cleaning pipeline to handle missing values and outliers.
  - Visualization of sales trends, seasonality, and feature correlations.
  - Insights into promotional effectiveness and customer patterns.

### Task 2: Machine Learning Pipeline

- **Preprocessing:**
  - Feature engineering (e.g., extracting temporal features, scaling).
  - Handling categorical and missing data.
- **Modeling:**
  - Tree-based regression models (e.g., Random Forest).
  - Custom loss function selection and hyperparameter tuning.
- **Post-Modeling Analysis:**
  - Feature importance analysis.
  - Confidence interval estimation for predictions.

### Task 3: Deep Learning

- **Objective:** Develop a Long Short-Term Memory (LSTM) model for time series forecasting.
- **Key Steps:**
  - Transforming data for supervised learning.
  - Stationarity testing and autocorrelation analysis.
  - Building and training an LSTM model using TensorFlow or PyTorch.

### Task 4: Model Deployment

- **REST API:** Flask-based API for real-time prediction.
- **Deployment:** Hosting on a cloud platform for accessibility.
- **Features:** Accepts input data, processes it, and returns predictions.

---

## Skills and Technologies

- **Programming:** Python (pandas, NumPy, scikit-learn, TensorFlow/PyTorch, Flask).
- **Visualization:** Matplotlib, Seaborn.
- **Machine Learning:** Feature engineering, regression models, hyperparameter tuning.
- **Deep Learning:** LSTM modeling.
- **MLOps:** Model versioning, CI/CD pipelines, and logging.
- **Tools:** Jupyter Notebook, GitHub, Docker, Google Colab.

---

## Project Structure

```
|-- data/
    |-- raw/
        |-- train.csv             # Original training dataset
        |-- test.csv              # Original test dataset
    |-- processed/
        |-- train_cleaned.csv     # Cleaned and preprocessed training data
        |-- test_cleaned.csv      # Cleaned and preprocessed test data
|-- notebooks/
    |-- 01_exploratory_analysis.ipynb   # EDA and data insights
    |-- 02_ml_pipeline.ipynb           # Machine learning model development
    |-- 03_deep_learning.ipynb         # LSTM-based deep learning model
|-- models/
    |-- serialized_models/
        |-- rf_model.pkl          # Serialized Random Forest model
        |-- lstm_model.h5         # Serialized LSTM model
|-- src/
    |-- preprocessing.py          # Data cleaning and feature engineering scripts
    |-- modeling.py               # Machine learning and deep learning scripts
    |-- deployment.py             # Scripts for model deployment
|-- api/
    |-- app.py                    # Flask app for model serving
    |-- requirements.txt          # Dependencies for the API
|-- README.md                     # Project documentation
|-- LICENSE                       # License information
```

---

## Installation and Usage

1. **Clone the repository:**

   ```bash
   git clone https://github.com/YonInsights/rossmann-sales-forecasting.git
   ```

2. **Install dependencies:**

   ```bash
   pip install -r requirements.txt
   ```

3. **Run the notebooks:**
   Explore the data, build models, and visualize results.

4. **Start the API:**

   ```bash
   cd api
   python app.py
   ```

---

## Results

### Machine Learning Model Performance

- **Metrics:** RMSE, MAE, and R2 Score.
- **Feature Importance:** Key drivers of sales identified.

### Deep Learning Model Performance

- **LSTM:** Captured temporal patterns effectively.

### Deployment

- Predictions available via API endpoints.

---

## Author

**Yonatan Abrham**

- **Email:** [email2yonatan@gmail.com](mailto\:email2yonatan@gmail.com)
- **LinkedIn:** [Yonatan Abrham](https://linkedin.com/in/YonatanAbrham)
- **GitHub:** [YonInsights](https://github.com/YonInsights)

Feel free to connect for collaborations or queries.

---

## Acknowledgements

Heartfelt thanks to 10 Academy for providing an excellent internship opportunity.
Appreciation for the open-source tools and the community that made this project possible.

