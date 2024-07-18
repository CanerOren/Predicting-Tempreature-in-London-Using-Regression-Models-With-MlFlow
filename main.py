import mlflow
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import mlflow.sklearn
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

weather =pd.read_csv("london_weather.csv")

#Data Analysis Cleaning
weather.info()
weather.isna().sum().sort_values()
weather.dtypes
#Converting data column to datetime format
weather['date']=pd.to_datetime(weather['date'],format="%Y%m%d")

#Prepartion for plotting
weather['year']=weather['date'].dt.year
weather['month']=weather['date'].dt.month

weather_metrics=['cloud_cover','sunshine','global_radiation','max_temp','mean_temp','min_temp','precipitation','pressure','snow_depth']
weather_per_month=weather.groupby(['year','month'],as_index=False)[weather_metrics].mean()
#EDA

sns.lineplot(x='year',y="mean_temp",data=weather_per_month,ci=None)
plt.show()
sns.lineplot(x='month',y="mean_temp",data=weather_per_month,ci=None)
plt.show()
sns.heatmap(weather.corr(),annot=True)
plt.show()

#Max and Min Temp Features Removed Because of the Data Leak
features=['month','cloud_cover','sunshine','global_radiation','precipitation','pressure','snow_depth']
target='mean_temp'

#We can't fill target variables missing values so we drop them
weather=weather.dropna(subset=['mean_temp'])

X=weather[features]
y=weather[target]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

imputer=SimpleImputer(strategy='mean')

X_train=imputer.fit_transform(X_train)
X_test= imputer.transform(X_test)

scaler=StandardScaler()

X_train= scaler.fit_transform(X_train)
X_test= scaler.transform(X_test)

mlflow.create_experiment("FirstMLProject")
mlflow.set_experiment("FirstMLProject")
for idx,depth in enumerate([1,2,5,10,20]):
    run_name=f"run_{idx}"
    with mlflow.start_run(run_name=run_name):
        #Instatiating Models
        lin_reg=LinearRegression()
        tree_reg=DecisionTreeRegressor()
        random_reg=RandomForestRegressor()
        #Fitting
        lin_reg.fit(X_train,y_train)
        tree_reg.fit(X_train,y_train)
        random_reg.fit(X_train,y_train)
        #Logging
        mlflow.sklearn.log_model(lin_reg,'linear regression')
        mlflow.sklearn.log_model(tree_reg, 'tree regression')
        mlflow.sklearn.log_model(random_reg, 'randoforest regression')
        #Predicting and Evaluation
        y_pred_lin_reg=lin_reg.predict(X_test)
        lin_reg_mse=mean_squared_error(y_test,y_pred_lin_reg)
        y_pred_tree_reg = tree_reg.predict(X_test)
        tree_reg_mse = mean_squared_error(y_test, y_pred_tree_reg)
        y_pred_random_reg = random_reg.predict(X_test)
        random_reg_mse = mean_squared_error(y_test, y_pred_random_reg)

        mlflow.log_param("max_depth",depth)
        mlflow.log_metric("rmse_lr",lin_reg_mse)
        mlflow.log_metric("rmse_tree", tree_reg_mse)
        mlflow.log_metric("rmse_random", random_reg_mse)

experiment_results=mlflow.search_runs()
print(experiment_results)