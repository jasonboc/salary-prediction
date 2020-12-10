import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import xgboost as xgb
from xgboost import plot_importance
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle


train_feature = pd.read_csv("data/train_features.csv")
train_target = pd.read_csv("data/train_salaries.csv")
test_feature = pd.read_csv("data/test_features.csv")
train_df = pd.merge(train_feature, train_target, on='jobId')
train_df = train_df[train_df.salary > 8.5][0:50000]
train_df["jobType"] = train_df["jobType"].astype('category')
mean = train_df.groupby("jobType")['salary'].mean()
levels = mean.sort_values().index.tolist()
train_df["jobType"].cat.reorder_categories(levels, inplace=True)
train_df.loc[:, train_df.dtypes == 'object'] = train_df.select_dtypes(
    ['object']).apply(lambda x: x.astype('category'))
df_new = train_df.copy()
df_new = pd.get_dummies(
    df_new, columns=["jobType", "degree", "major", "industry", "companyId"])
df_x = df_new.drop(columns=["salary", "jobId"])
df_y = df_new["salary"]
X_train, X_test, y_train, y_test = train_test_split(
    df_x, df_y, test_size=0.33, random_state=0, shuffle=True)
xgbr = xgb.XGBRegressor(n_estimators=400, max_depth=4, learning_rate=0.1)

def model_training(X_train, y_train):
    neg_mse = cross_val_score(xgbr, X_train, y_train,
                              cv=5, scoring='neg_mean_squared_error')
    mean_mse = -1.0*np.mean(neg_mse)
    std_mse = np.std(neg_mse)
    return mean_mse,std_mse

xgbr.fit(df_x,df_y)
feature_important = xgbr.get_booster().get_score(importance_type='weight')
keys = list(feature_important.keys())
values = list(feature_important.values())
feature_importances = pd.DataFrame(values,
                                   keys,
                                   columns=['importance']).sort_values('importance', ascending=False).reset_index()
