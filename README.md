# housing-prices-KaggleComp-learning
End-to-end machine learning pipeline for predicting housing prices using scikit-learn, built for the Kaggle Learn Housing Prices competition.

##  Project Overview
This project focuses on predicting house prices using machine learning techniques on the **Housing Prices Competition for Kaggle Learn Users** dataset.  
The objective was to build a complete end-to-end regression pipeline, covering data preprocessing, model training, evaluation, and Kaggle submission.

---

##  Problem Statement
Given various housing-related features such as location, size, and construction details, the task is to predict the **SalePrice** of houses.

---

##  Technologies Used
- Python  
- pandas, numpy  
- scikit-learn  
- matplotlib  

---

##  Machine Learning Pipeline
All preprocessing and modeling steps were implemented using **scikit-learn Pipelines** to ensure consistency and prevent data leakage.

**Pipeline steps:**
- Missing value handling using `SimpleImputer`
- Numerical features: median imputation
- Categorical features: most-frequent imputation
- Feature scaling using `StandardScaler`
- Categorical feature encoding using `OneHotEncoder`
- Feature selection using Lasso (`SelectFromModel`)
- Model training using `RandomForestRegressor`

---

##  Model Performance
- Validation **R² score**: ~0.84  
- Validation **MAE**: ~17,200
- Validation **RMSE**: ~26308.379
- Kaggle Leaderboard Rank: **1963 / 4647 (~Top 42%)**

The evaluation focused on **generalization performance** rather than leaderboard overfitting.

---

##  Key Learnings
- Importance of handling missing values within machine learning pipelines
- Preventing data leakage using `Pipeline` and `ColumnTransformer`
- Differences between validation metrics and Kaggle leaderboard scores
- Building an end-to-end machine learning workflow from data preprocessing to submission

---

##  Repository Structure
housing-prices-kaggle/
├── housing_prices_model.py
├── submission.csv
└── README.md


---

##  Kaggle Competition
Housing Prices Competition for Kaggle Learn Users
https://www.kaggle.com/competitions/home-data-for-ml-course
---

##  Notes
This project was completed as a learning exercise to strengthen practical understanding of regression modeling and machine learning pipelines using real-world tabular data.

