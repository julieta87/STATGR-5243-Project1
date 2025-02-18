import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer, KNNImputer
from scipy import stats


# Define file paths for all datasets 
file_paths = [
    "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/2017_yellow/yellow_tripdata_2017-01.csv",
    "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/2017_yellow/yellow_tripdata_2017-02.csv",
    "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/2017_yellow/yellow_tripdata_2017-03.csv",
    "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/2017_yellow/yellow_tripdata_2017-04.csv",
    "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/2017_yellow/yellow_tripdata_2017-05.csv",
    "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/2017_yellow/yellow_tripdata_2017-06.csv",
    "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/2017_yellow/yellow_tripdata_2017-07.csv",
    "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/2017_yellow/yellow_tripdata_2017-08.csv",
    "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/2017_yellow/yellow_tripdata_2017-09.csv",
    "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/2017_yellow/yellow_tripdata_2017-10.csv",
    "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/2017_yellow/yellow_tripdata_2017-11.csv",
    "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/2017_yellow/yellow_tripdata_2017-12.csv"
]

# Load all datasets into dataframes
dataframes = [pd.read_csv(file) for file in file_paths]

# Merge datasets into one dataframe
merged_df = pd.concat(dataframes, ignore_index=True)

# Save the merged dataframe 
merged_df.to_csv("/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/Merged_Yellow_2017.csv", index=False)

# Display a preview of the merged dataset
print(merged_df.head()) 



# Standardize column names (lowercase, remove spaces)
merged_df.columns = merged_df.columns.str.lower().str.replace(" ", "_")

# Convert datetime columns to proper format before handling missing values and outliers
merged_df['tpep_pickup_datetime'] = pd.to_datetime(merged_df['tpep_pickup_datetime'], errors='coerce')
merged_df['tpep_dropoff_datetime'] = pd.to_datetime(merged_df['tpep_dropoff_datetime'], errors='coerce')

# Handle inconsistent data types
# Ensure categorical columns are properly formatted
merged_df['store_and_fwd_flag'] = merged_df['store_and_fwd_flag'].astype(str).str.upper()

# Remove duplicates
merged_df.drop_duplicates(inplace=True)

# Handle missing values 
# Drop columns with too many missing values (if missing > 80%)
missing_threshold = 0.8 * len(merged_df)
merged_df = merged_df.dropna(axis=1, thresh=missing_threshold)

# Define different imputation strategies
imputers = {
    "mean": SimpleImputer(strategy="mean"),
    "median": SimpleImputer(strategy="median"),
    "most_frequent": SimpleImputer(strategy="most_frequent"),
    "knn": KNNImputer(n_neighbors=5)
}

# Apply imputation methods and evaluate
imputed_dfs = {}
for method, imputer in imputers.items():
    df_imputed = merged_df.copy()
    numeric_cols = df_imputed.select_dtypes(include=[np.number]).columns
    df_imputed[numeric_cols] = imputer.fit_transform(df_imputed[numeric_cols])
    imputed_dfs[method] = df_imputed

# Choose the best imputation strategy (based on minimal variance change)
best_method = min(imputed_dfs, key=lambda method: imputed_dfs[method][numeric_cols].var().sum())
df = imputed_dfs[best_method]
print(f"Best imputation method chosen: {best_method}")

# Handle outliers after handling missing values 
outlier_methods = {
    "zscore": lambda x: np.abs(stats.zscore(x)) < 3,
    "iqr": lambda x: (x > (x.quantile(0.25) - 1.5 * (x.quantile(0.75) - x.quantile(0.25)))) & \
                     (x < (x.quantile(0.75) + 1.5 * (x.quantile(0.75) - x.quantile(0.25))))
}

outlier_evaluations = {}
for method, func in outlier_methods.items():
    df_filtered = df.copy()
    for col in df_filtered.select_dtypes(include=[np.number]).columns:
        df_filtered = df_filtered[func(df_filtered[col])]
    outlier_evaluations[method] = df_filtered

# Choose the best outlier removal method (based on retaining the most data)
best_outlier_method = max(outlier_evaluations, key=lambda method: len(outlier_evaluations[method]))
df = outlier_evaluations[best_outlier_method]
print(f"Best outlier handling method chosen: {best_outlier_method}")