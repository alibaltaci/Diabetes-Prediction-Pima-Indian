# Diabetes Model (Pima Indian)

# Import Libraries

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns

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



X = df.drop("Outcome",axis=1)
y = df["Outcome"]

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

for name,model in models:
    mod = model.fit(X_train,y_train) #trainleri modele fit etmek
    y_pred = mod.predict(X_test) # tahmin
    acc = accuracy_score(y_test, y_pred) #rmse hesabı
    cvscore = cross_val_score(model, X,y, cv = 10).mean()
    print("Holdout Method:",end=" ")
    print(name,acc) #yazdırılacak kısım
    print("Cross Val Score",end=" ")
    print(name,cvscore)
    print("------------------------------------")