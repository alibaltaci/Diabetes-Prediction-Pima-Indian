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