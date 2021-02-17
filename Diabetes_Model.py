# Diabetes Model (Pima Indian)

# Import Libraries

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import helper_functions as hf

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, \
    classification_report
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score

warnings.simplefilter(action="ignore")
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Load the dataset
df = pd.read_csv(r"C:\Users\TOSHIBA\Desktop\Diabetes Pima Indian\DiabetesDatasetForModeling.csv")
df.head()


# Models
models = [("LR", LogisticRegression()),
          ("KNN", KNeighborsClassifier()),
          ("CART", DecisionTreeClassifier()),
          ("RF", RandomForestClassifier()),
          ("SVM", SVC(gamma='auto')),
          ('GradientBoosting', GradientBoostingClassifier()),
          ("XGB", GradientBoostingClassifier()),
          ("LightGBM", LGBMClassifier())]


# Define dependent and independent variables
X = df.drop("Outcome",axis=1)
y = df["Outcome"]

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)


# Holdout Method & Cross Val Score
for name,model in models:
    mod = model.fit(X_train,y_train)
    y_pred = mod.predict(X_test)
    acc = accuracy_score(y_test, y_pred) #rmse
    cvscore = cross_val_score(model, X,y, cv = 10).mean()
    print("Holdout Method:",end=" ")
    print(name,acc)
    print("Cross Val Score",end=" ")
    print(name,cvscore)
    print("------------------------------------")


# Random Forest Model Tuning #

rf_params = {"n_estimators": [100, 200, 500, 1000],
             "max_features": [3, 5, 7],
             "min_samples_split": [2, 5, 10, 30],
             "max_depth": [3, 5, 8, None]}

rf_model = RandomForestClassifier(random_state=42)

rf_gscv = GridSearchCV(rf_model, rf_params, cv=10, n_jobs=-1, verbose=2).fit(X, y)
# [Parallel(n_jobs=-1)]: Done 1920 out of 1920 | elapsed: 16.8min finished

rf_gscv.best_params_  # {'max_depth': 8,
                    # 'max_features': 7,
                    # 'min_samples_split': 5,
                    # 'n_estimators': 500}

# Final Model
rf_tuned = RandomForestClassifier(**rf_gscv.best_params_).fit(X,y)

cross_val_score(rf_tuned, X, y, cv=10).mean()  # 0.89

# Feature Importances
hf.plot_feature_importances(rf_tuned, X)


# LightGBM  Model Tuning #

lgbm_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
               "n_estimators": [500, 1000, 1500],
               "max_depth": [3, 5, 8]}

lgbm_model = LGBMClassifier(random_state=42)

rf_gscv = GridSearchCV(lgbm_model, lgbm_params, cv=10, n_jobs=-1, verbose=2).fit(X, y)
# [Parallel(n_jobs=-1)]: Done 450 out of 450 | elapsed:  1.6min finished

lgbm_gscv.best_params_

# Final Model
lgbm_tuned = LGBMClassifier(**rf_gscv.best_params_).fit(X, y)

cross_val_score(lgbm_tuned, X, y, cv=10).mean()   # 0.8997

# Feature Importances
hf.plot_feature_importances(lgbm_tuned, X)


# XGB Model Tuning #

xgb_params = {"learning_rate": [0.01, 0.03, 0.05, 0.1, 0.5],
             "max_depth": [3, 5, 8],
             "n_estimators": [200, 500, 1000]}

xgb_model = XGBClassifier(random_state=42)

xcb_gscv = GridSearchCV(xgb_model, xgb_params, cv=10, n_jobs=-1, verbose=2).fit(X, y)
# [Parallel(n_jobs=-1)]: Done 450 out of 450 | elapsed:  2.0min finished

# Final Model
xgb_tuned = XGBClassifier(**xcb_gscv.best_params_).fit(X, y)

cross_val_score(xgb_tuned, X, y, cv=10).mean()   # 0.8972

# Feature Importances
hf.plot_feature_importances(xgb_tuned, X)
