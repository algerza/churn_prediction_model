from dateutil import parser
import pandas as pd

# def convert_dates_to_iso(df):
#     """
#     A function to convert all date columns in a DataFrame to ISO 8601 format.
    
#     Parameters:
#     df: a pandas DataFrame
    
#     Returns:
#     df: the updated DataFrame with date columns in ISO 8601 format
#     """
#     # Loop through all columns in the DataFrame
#     for col in df.columns:
#         # Check if the column data type is object
#         if df[col].dtype == 'object':
#             # Try to parse the string date values in the column and convert to ISO 8601 format
#             try:
#                 df[col] = df[col].apply(lambda x: parser.parse(x).strftime('%Y-%m-%d') if isinstance(x, str) else x)
#             except ValueError:
#                 # If there's an error parsing the string date values, skip the column and move to the next one
#                 pass
#     # Return the updated DataFrame
#     return df

def remove_constant_columns(df):
    """
    Removes columns that have the same value for every row.

    Args:
    df: pandas DataFrame object.

    Returns:
    pandas DataFrame object with constant columns removed.
    """
    # Get a boolean mask of columns that have the same value for every row
    constant_cols = (df.nunique() == 1)

    # Get the names of the constant columns
    constant_col_names = constant_cols[constant_cols == True].index.tolist()

    # Drop the constant columns
    df = df.drop(constant_col_names, axis=1)

    return df


def remove_sparse_columns(df):
    """
    Remove columns that are at least 99% blank or null values.

    Args:
    df (pandas.DataFrame): input dataframe.

    Returns:
    pandas.DataFrame: updated dataframe with sparse columns removed.
    """
    threshold = len(df) * 0.01
    return df.dropna(thresh=threshold, axis=1)



def remove_null_rows(df):
    """
    Removes rows with null values for columns that are at least 99% filled in.

    Args:
    df: pandas DataFrame object.

    Returns:
    pandas DataFrame object with null rows removed.
    """
    # Get the percentage of non-null values for each column
    non_null_pct = df.count() / len(df)

    # Get the columns where at least 99% of values are non-null
    non_null_cols = non_null_pct[non_null_pct >= 0.95].index

    # Drop the rows with null values in those columns
    df = df.dropna(subset=non_null_cols)

    return df


def flag_outliers(df):
    """
    This function takes a pandas DataFrame as input and flags outliers in numerical columns.
    """
    # Create a new DataFrame to store the outlier flags
    outlier_flags = pd.DataFrame(index=df.index)

    # Loop through each numerical column in the input DataFrame
    for col in df.select_dtypes(include=np.number):
        # Skip outlier detection for binary columns
        if df[col].nunique() == 2:
            continue
        
        # Calculate the mean and standard deviation of the column
        col_mean = df[col].mean()
        col_std = df[col].std()

        # Calculate the 1st and 99th percentiles of the column
        col_1st_perc = np.percentile(df[col], 1)
        col_99th_perc = np.percentile(df[col], 99)

        # Calculate the lower and upper bounds for outlier detection
        lower_bound = col_mean - 3 * col_std
        upper_bound = col_mean + 3 * col_std

        # Determine if there are any outliers in the column
        has_outliers = ((df[col] < col_1st_perc) | (df[col] > col_99th_perc) | 
                        (df[col] < lower_bound) | (df[col] > upper_bound)).any()

        # If there are outliers, create a new column in the outlier flags DataFrame
        # that flags whether or not each value in the input column is an outlier
        if has_outliers:
            outlier_flags[f"{col}_outlier"] = (
                (df[col] < col_1st_perc) | (df[col] > col_99th_perc) | 
                (df[col] < lower_bound) | (df[col] > upper_bound)
            ).astype(int)

    # Join the outlier flags DataFrame with the input DataFrame
    df = pd.concat([df, outlier_flags], axis=1)

    # Return the new DataFrame with outlier flags
    return df


def check_for_duplicates(df):
    """Checks for duplicate values in a given dataframe."""
    if df.duplicated().any().any():
        # if any duplicated values exist, print a message indicating their presence
        print("- There are duplicated values in the dataframe.")
    else:
        # if no duplicated values exist, print a message indicating their absence
        print("- There are no duplicated values in the dataframe.")


def check_for_missing_values(df):
    """Checks for missing values in a given dataframe."""
    if df.isnull().any().any():
        # create a list of columns with missing values
        null_cols = df.columns[df.isnull().any()].tolist()
        for col in null_cols:
            # for each column with missing values, print a message indicating their presence
            print(f"- There are missing values in the dataframe. Column '{col}' has missing values.")
    else:
        # if no missing values exist, print a message indicating their absence
        print("- There are no missing values in the dataframe.")


def check_for_imbalanced_dataset(df):
    """Checks for class imbalance in a given dataframe."""
    # count the number of samples in each class
    class_counts = df['Churn'].value_counts()

    # calculate the imbalance ratio
    imbalance_ratio = class_counts.min() / class_counts.sum()

    if imbalance_ratio < 0.3:
        # if the imbalance ratio is below a threshold, print a message indicating class imbalance
        print(f"- The dataframe is imbalanced, with an imbalance ratio of {round(imbalance_ratio, 2)}.")
    else:
        # if the imbalance ratio is above a threshold, print a message indicating class balance
        print(f"- The dataframe is balanced, with an imbalance ratio of {round(imbalance_ratio, 2)}.")


