# Exploratory Data Analysis and Data Visualization (Pima Indian)

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


# GENERAL OVERVIEW

df.info()
df.shape
df.columns
df.index
df.describe([0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T

# Missing Values

df.isnull().values.any()


# NUMERICAL VARIABLES ANALYSIS

df.hist(bins=20, figsize=(15, 15), color='b')
plt.show()

# Histograms for numerical variables

hf.hist_for_nums(df, df.columns)


# TARGET ANALYSIS

# Distribution of the target variable

df["Outcome"].value_counts()

sns.countplot(x="Outcome", data=df)
plt.show()

# Look at the mean, meadian and std for each variable groupped by Outcome

for col in df.columns:
    print(df.groupby("Outcome").agg({col: ["mean", "median", "std"]}))



