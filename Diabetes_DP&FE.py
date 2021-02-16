# Data Preprocessing and Feature Engineering


# Import Libraries

import warnings
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
import helper_functions as hf

warnings.simplefilter(action="ignore")
pd.pandas.set_option('display.max_columns', None)
pd.set_option('display.float_format', lambda x: '%.3f' % x)


# Load the dataset

df_backup = pd.read_csv(r"C:\Users\TOSHIBA\Desktop\Diabetes Pima Indian\diabetes.csv")
df = df_backup.copy()
df.head()

# OUTLIER ANALYSIS

hf.has_outliers(df,df.columns)

# Threshold
for col in df.columns:
    hf.replace_with_thresholds_with_lambda(df,col)

hf.has_outliers(df,df.columns)


# MISSING VALUES ANALYSIS

df.isnull().values.any() # False --> ?

# It is not possible to have a value of 0 in the some variables in the data set.
# Now let's query the value of 0 in variables.

hf.num_catcher(df,0)

# Variable  Pregnancies : 111
# Variable  Glucose : 5
# Variable  BloodPressure : 35
# Variable  SkinThickness : 227
# Variable  Insulin : 374
# Variable  BMI : 11
# Variable  Outcome : 500

# Missing values may be replaced by 0.

cols_na = ["Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"]

for col in cols_na:
    df.loc[(df[col] == 0), col] = np.NaN

# an other way (with lambda)

# for col in variables_with_na:
#     df[col] = df[col].apply(lambda x: np.NaN if x == 0 else x)

# Checking missing values again

df.isnull().values.any()
df.isnull().sum().sort_values()

hf.num_catcher(df,0)

# Variable  Pregnancies : 111
# Variable  Outcome : 500

# these variables can have values of 0.