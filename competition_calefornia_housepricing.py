#This dataset belongs to competition "Housing Prices Competition for Kaggle Learn Users" on Kaggle 
#The goal is to train the model on training data and then predict the prices of houses in test dataset 

import pandas as pd 
import matplotlib.pyplot as plt 
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.linear_model import LassoCV,Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
from sklearn.impute import SimpleImputer

path = "data/train.csv"
df = pd.read_csv(path)

print(df.head())
print(df.columns)
df.info()
print(df.describe())
na_cols= df.isna().sum()

#damn this data is too messy, lets remove columns with too many null values cuz they are too late to save 
useless_columns = [column_name for column_name,null_count in na_cols.items()
            if null_count> 600]
print(len(useless_columns)) #5
df = df.drop(useless_columns,axis=1)

#handling leftover null values 
df.info()
temp=df.fillna({'LotFrontage': df['LotFrontage'].median()}) #replacing the na with median value of that column 
df_new = temp.dropna()
df_new.info()

#seperate target and features 
X = df_new.drop('SalePrice',axis=1)
y = df_new['SalePrice']


#Preprocessing data 
num_col = X.select_dtypes(include='number').columns
obj_col = X.select_dtypes(include='object').columns

num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='median')), #simpleimputer fills the NA in in the databy itself using the rule we choose, here I used median
    ('scaler', StandardScaler())
])

obj_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')), #here the nan values are relace with most_frequent values from the column 
    ('onehot', OneHotEncoder(
        drop='first',
        handle_unknown='ignore',
        sparse_output=False
    ))
])

preprocessor = ColumnTransformer([
    ('num', num_pipeline, num_col),
    ('obj', obj_pipeline, obj_col)
])

#pipeline
pipeline = Pipeline([
    ('prep',preprocessor),
    ('feature selection',SelectFromModel(Lasso())),
    ('model',RandomForestRegressor(
        n_estimators=400,
        n_jobs=-1,
        random_state=42
    ))
])

#train/test split
X_train,X_validate,y_train,y_validate= train_test_split(X,y,test_size=0.2,random_state=42)

pipeline.fit(X_train,y_train)
y_pred = pipeline.predict(X_validate)

print("MSE: ",mean_squared_error(y_validate,y_pred),'\n', #692130828.48
      "R2 score: ",r2_score(y_validate,y_pred),'\n', #0.838
      "MAE: ",mean_absolute_error(y_validate,y_pred)) #17153.67


#The pipeline seems to have worked after changing n_estimator a few times and i was able to achive an r2_score of 0.838 on the first go which is pretty good,
#rather I can conclude this model is completed here, now its time to predict the values on the test dataset and prepare the submission CSV

test_path = "data/test.csv"
test_df = pd.read_csv(test_path)

print(test_df.head())
test_df.info()
print(test_df.isna().sum()) 

test_df = test_df.drop(useless_columns,axis=1)
test_df.info()

#handling null values 
test_df['LotFrontage'] = test_df['LotFrontage'].fillna(df['LotFrontage'].median()) #using the same median as the training data 
#not filing or dropping any other null value containing rows since the competition expects output for every ID, they will be ignored by the pipeline 

#predicting on the test data
test_pred = pipeline.predict(test_df)

#converting the final predictions into the submission csv format
submission = pd.DataFrame({
    "Id": test_df['Id'],
    "SalePrice" : test_pred
})

submission.to_csv("sumission.csv",index= False) #index=False means not to include indexes in the csv 

