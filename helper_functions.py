# Helper Functions for Diabetes Prediction - Pima Indian

# Histograms for numerical variables
def hist_for_nums(data, numeric_cols):
    """
    :param data:
    :param numeric_cols:
    :return:
    """
    import matplotlib.pyplot as plt
    col_counter = 0
    data = data.copy()
    for col in numeric_cols:
        data[col].hist(bins=20)
        plt.xlabel(col)
        plt.title(col)
        plt.show()
        col_counter += 1
    print(col_counter, "variables have been plotted")


def num_catcher(dataframe, number):
    """

    :param dataframe:
    :param cols:
    :param number:
    :return:
    """

    for i in dataframe.columns:
        zero = list(dataframe[i]).count(number)
        if zero == 0:
            pass
        else:
            print("Variable ", i, ":", zero)


# Find correlations for numeric variables
def find_correlation(dataframe, num_cols, corr_limit=0.60):
    """
    :param dataframe:
    :param num_cols:
    :param corr_limit:
    :return:
    """
    high_correlations = []
    low_correlations = []
    for col in num_cols:
        if col == "TARGET":
            pass
        else:
            correlation = dataframe[[col, "TARGET"]].corr().loc[col, "TARGET"]
            print(col, correlation)
            if abs(correlation) > corr_limit:
                high_correlations.append(col + ": " + str(correlation))
            else:
                low_correlations.append(col + ": " + str(correlation))
    return low_correlations, high_correlations


# Outlier Thresholds
def outlier_thresholds(dataframe, variable):
    """
    :param dataframe:
    :param variable:
    :return:
    """
    quartile1 = dataframe[variable].quantile(0.05)
    quartile3 = dataframe[variable].quantile(0.95)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


# Function to report variables with outliers and return the names of the variables with outliers with a list
def has_outliers(dataframe, num_col_names, plot=False):
    """

    :param dataframe:
    :param num_col_names:
    :param plot:
    :return:
    """
    variable_names = []
    for col in num_col_names:
        low_limit, up_limit = outlier_thresholds(dataframe, col)
        if dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].any(axis=None):
            number_of_outliers = dataframe[(dataframe[col] > up_limit) | (dataframe[col] < low_limit)].shape[0]
            print(col, ":", number_of_outliers)
            variable_names.append(col)
            if plot:
                sns.boxplot(x=dataframe[col])
                plt.show()
    return variable_names


# Function to reassign up/low limits to the ones above/below up/low limits by using apply and lambda method
def replace_with_thresholds_with_lambda(dataframe, variable):
    """
    :param dataframe:
    :param variable:
    :return:
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe[variable] = dataframe[variable].apply(lambda x: up_limit if x > up_limit else (low_limit if x < low_limit else x))


# One Hot Encoding to categorical variables.
def one_hot_encoder(dataframe, categorical_cols, nan_as_category=True):
    """
    :param dataframe:
    :param categorical_cols:
    :param nan_as_category:
    :return:
    """
    original_columns = list(dataframe.columns)
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, dummy_na=False, drop_first=True)
    new_columns = [c for c in dataframe.columns if c not in original_columns]
    return dataframe, new_columns























