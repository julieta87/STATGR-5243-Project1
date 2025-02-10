#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Feb  9 18:09:28 2025

@author: ivy
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler



# load the Paths 
file_paths = ["/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/Merged_Green_2024.csv",
            "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/Merged_Green_2023.csv",
            "/Users/ivy/Documents/Semester 2/5243_Applied Data Science/5243_HW/5243_HW1/TLC_Trip_Record_Data/Merged_Green_2022.csv"
            ]


# Load all CSV files into a list of DataFrames
df_list = [pd.read_csv(f, low_memory=False) for f in file_paths]

# Add a new column indicating the expected year based on the file name
for i, year in enumerate([2024, 2023, 2022]):
    df_list[i]['expected_year'] = year

# Merge all DataFrames
df_combined = pd.concat(df_list, ignore_index=True)

# Convert the pickup and dropoff datetime columns to datetime format
df_combined['lpep_pickup_datetime'] = pd.to_datetime(df_combined['lpep_pickup_datetime'], errors='coerce')
df_combined['lpep_dropoff_datetime'] = pd.to_datetime(df_combined['lpep_dropoff_datetime'], errors='coerce')


# Fix inconsistent dates based on the expected year
# (Example: look through the dataset in Jan 2023)
def fix_dates(row):
    if row['lpep_pickup_datetime'].year != row['expected_year']:
        row['lpep_pickup_datetime'] = pd.NaT
    if row['lpep_dropoff_datetime'].year != row['expected_year']:
        row['lpep_dropoff_datetime'] = pd.NaT
    return row

df_combined = df_combined.apply(fix_dates, axis=1)

# Drop rows with invalid or NaT dates
df_combined.dropna(subset=['lpep_pickup_datetime', 'lpep_dropoff_datetime'], inplace=True)



# Extract date and hour for aggregation
df_combined['date'] = df_combined['lpep_pickup_datetime'].dt.date
df_combined['hour'] = df_combined['lpep_pickup_datetime'].dt.hour

# Aggregate by date and hour, summing up relevant numerical columns
aggregated_df = df_combined.groupby(['date', 'hour']).agg({
    'passenger_count': 'sum',
    'trip_distance': 'sum',
    'fare_amount': 'sum',
    'tip_amount': 'sum',
    'total_amount': 'sum'
}).reset_index()

# Save the aggregated data to a CSV file 
aggregated_df.to_csv("Aggregated_Taxi_Data_2022_2024.csv", index=False)

# Display the aggregated data
print(aggregated_df)



# Detect and rectify data inconsistencies
# Check data types to identify problematic columns
print(df_combined.info())

# Drop columns with no meaningful information
df_combined.drop(columns=['ehail_fee'], inplace=True)


# Ensure only numerical columns are processed for imputation and cleaning
numerical_columns = ['passenger_count', 'trip_distance', 'fare_amount', 'tip_amount', 'total_amount']
non_numeric_columns = [col for col in df_combined.columns if col not in numerical_columns]

# Convert numerical columns to the correct data type (python default)
for col in numerical_columns:
    df_combined[col] = pd.to_numeric(df_combined[col], errors='coerce')

# Handle missing values in numerical columns with median imputation 
# (sth maybe we can change and refine later)
for col in numerical_columns:
    df_combined[col].fillna(df_combined[col].median(), inplace=True)

# Drop duplicates
df_combined.drop_duplicates(inplace=True)

# Handle outliers using IQR
# (sth maybe we can change and refine later)
Q1 = df_combined[numerical_columns].quantile(0.25)
Q3 = df_combined[numerical_columns].quantile(0.75)
IQR = Q3 - Q1

# Filter out outliers for numerical columns
df_combined = df_combined[~((df_combined[numerical_columns] < (Q1 - 1.5 * IQR)) |
                            (df_combined[numerical_columns] > (Q3 + 1.5 * IQR))).any(axis=1)]

# Display cleaned data
print(df_combined.info())

# Save the cleaned data to a CSV file
df_combined.to_csv("Cleaned_Taxi_Data_2022_2024.csv", index=False)


# some basic descriptions and visualizations
# 1. Generate Summary Statistics
print("Summary Statistics:\n", df_combined.describe())

# 2. Visualize Distributions
# Plot distribution for numerical columns
for col in ['trip_distance', 'fare_amount', 'tip_amount', 'total_amount']:
    plt.figure(figsize=(8, 5))
    sns.histplot(df_combined[col], bins=50, kde=True, color="blue")
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.show()

# 3. Visualize Relationships
# Scatter plot for trip distance vs fare amount
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_combined, x='trip_distance', y='fare_amount', alpha=0.5)
plt.title("Trip Distance vs Fare Amount")
plt.xlabel("Trip Distance")
plt.ylabel("Fare Amount")
plt.show()

# Scatter plot for trip distance vs total amount
plt.figure(figsize=(8, 5))
sns.scatterplot(data=df_combined, x='trip_distance', y='total_amount', alpha=0.5, color="green")
plt.title("Trip Distance vs Total Amount")
plt.xlabel("Trip Distance")
plt.ylabel("Total Amount")
plt.show()

# 4. Correlation Matrix and Heatmap
# Compute correlation matrix for numerical columns
correlation_matrix = df_combined[['passenger_count', 'trip_distance', 'fare_amount', 'tip_amount', 'total_amount']].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm", cbar=True)
plt.title("Correlation Matrix Heatmap")
plt.show()

# 5. Identify Patterns and Anomalies
# Box plot to identify patterns in fare amount by hour of day
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_combined, x='hour', y='fare_amount')
plt.title("Fare Amount by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Fare Amount")
plt.show()

# Box plot for total amount by hour
plt.figure(figsize=(10, 6))
sns.boxplot(data=df_combined, x='hour', y='total_amount', palette="viridis")
plt.title("Total Amount by Hour of Day")
plt.xlabel("Hour of Day")
plt.ylabel("Total Amount")
plt.show()





# Time Series Modeling 
# (but there exists high_variance, and the data average in different year vary significantly)
# (actually including 2022,2023 is better than just include 2024 for prediction)
# Aggregate data by date for time series analysis
aggregated_daily = df_combined.groupby('date').agg({
    'passenger_count': 'sum',
    'trip_distance': 'sum',
    'fare_amount': 'sum',
    'total_amount': 'sum'
}).reset_index()

# Convert date to datetime and set as index
aggregated_daily['date'] = pd.to_datetime(aggregated_daily['date'])
aggregated_daily.set_index('date', inplace=True)

# Select a column to model 
# (e.g., I used total_amount)
time_series_data = aggregated_daily['total_amount']

# Scale the data
scaler = MinMaxScaler()
time_series_scaled = scaler.fit_transform(time_series_data.values.reshape(-1, 1))


# LSTM model 
# (i feel that my results exists overfitting)
# Prepare sequences for LSTM
def create_sequences(data, seq_length):
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i + seq_length])
        y.append(data[i + seq_length])
    return np.array(X), np.array(y)

seq_length = 30  # Use the past 30 days to predict the next day (i just randomly set the seq_length to be 30)
X, y = create_sequences(time_series_scaled, seq_length)

# Split into train and test sets
train_size = int(len(X) * 0.8)
X_train, X_test = X[:train_size], X[train_size:]
y_train, y_test = y[:train_size], y[train_size:]

# Build the refined LSTM model
model = Sequential([
    LSTM(100, return_sequences=True, input_shape=(seq_length, 1)),
    Dropout(0.2),
    LSTM(100, return_sequences=True),
    Dropout(0.2),
    LSTM(100),
    Dropout(0.2),
    Dense(1)
])

# Compile the model
model.compile(optimizer='adam', loss='mean_squared_error')

# Train the model
model.fit(X_train, y_train, epochs=50, batch_size=64, verbose=1, validation_split=0.1)

# Make predictions
predicted = model.predict(X_test)
predicted = scaler.inverse_transform(predicted)
y_test_rescaled = scaler.inverse_transform(y_test)

# Plot actual vs predicted values
plt.figure(figsize=(10, 5))
plt.plot(y_test_rescaled, label="Actual")
plt.plot(predicted, label="Predicted")
plt.legend(loc="upper left")
plt.title("Refined LSTM Model Forecast vs Actual (2022-2024)")
plt.show()

# Evaluate the model
mse_lstm = mean_squared_error(y_test_rescaled, predicted)
mae_lstm = mean_absolute_error(y_test_rescaled, predicted)
print(f"Mean Squared Error (Refined LSTM): {mse_lstm}")
print(f"Mean Absolute Error (Refined LSTM): {mae_lstm}")

# Save the predictions
y_test_df = pd.DataFrame({'Actual': y_test_rescaled.flatten(), 'Predicted': predicted.flatten()})
y_test_df.to_csv("Refined_LSTM_Forecast_2022_2024.csv", index=False)

