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

# ggplot
plt.style.use('ggplot')
f, ax = plt.subplots(figsize=(11, 15))
ax.set_facecolor('#fafafa')
ax.set(xlim=(-.05, 300))
#plt.ylabel()
plt.title("Overview Data Set")
ax = sns.boxplot(data = df,
  orient = 'h',
  palette = 'Set2')
plt.show()

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


# FEATURE ENGINEERING

# BMI #
df["NEW_BMIRanges"] = pd.cut(x=df["BMI"], bins=[0, 18.5, 25, 30, 100], labels=["Underweight", "Healthy", "Overweight", "Obese"])
df["NEW_BMIRanges"] = df["NEW_BMIRanges"].astype(str)
df["NEW_BMIRanges"].head()

# Value counts
df[["NEW_BMIRanges"]].value_counts()

# countplot
sns.countplot(x="NEW_BMIRanges", hue="Outcome", data=df)
plt.show()

# AGE #
df["Age"].describe()
df["NEW_AgeRanges"] = pd.cut(x=df["Age"], bins=[15, 25, 65, 81], labels=["Young", "Mid_Aged", "Senior"])
df["NEW_AgeRanges"] = df["NEW_AgeRanges"].astype(str)
df["NEW_AgeRanges"].head()

# Value counts
df[["NEW_AgeRanges"]].value_counts()

# countplot
sns.countplot(x="NEW_AgeRanges", hue="Outcome", data=df)
plt.show()

# GLUCOSE #
df["Glucose"].describe()
df["NEW_GlucoseLevels"] = pd.cut(x=df["Glucose"], bins=[0, 70, 99, 126, 200], labels=["Low", "Normal", "Secret", "High"])
df["NEW_GlucoseLevels"] = df["NEW_GlucoseLevels"].astype(str)
df["NEW_GlucoseLevels"].head()

# Value counts
df[["NEW_GlucoseLevels"]].value_counts()

# countplot
sns.countplot(x="NEW_GlucoseLevels", hue="Outcome", data=df)
plt.show()

# INSULIN #
df.loc[(df["Insulin"] >= 16) & (df["Insulin"] <= 166), "NEW_InsulinDesc"] = 1
df.loc[(df["Insulin"].isnull()), "NEW_InsulinDesc"] = 0

# Value counts
df[["NEW_InsulinDesc"]].value_counts()

# countplot
sns.countplot(x="NEW_InsulinDesc", hue="Outcome", data=df)
plt.show()

# BloodPressure #
df["BloodPressure"].describe()
df.loc[(df["BloodPressure"] > 90), "NEW_HyperBloodPressure"] = 1
df.loc[(df["NEW_HyperBloodPressure"].isnull()), "NEW_HyperBloodPressure"] = 0

# Value counts
df[["NEW_HyperBloodPressure"]].value_counts()

# countplot
sns.countplot(x="NEW_InsulinDesc", hue="Outcome", data=df)
plt.show()

