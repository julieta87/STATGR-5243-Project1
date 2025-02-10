import pandas as pd

df_2015_01 = pd.read_parquet('yellow_tripdata_2015-01.parquet')
df_2015_01.to_csv('yellow_tripdata_2015-01.csv')
df_2015_02 = pd.read_parquet('yellow_tripdata_2015-02.parquet')
df_2015_02.to_csv('yellow_tripdata_2015-02.csv')
df_2015_03 = pd.read_parquet('yellow_tripdata_2015-03.parquet')
df_2015_03.to_csv('yellow_tripdata_2015-03.csv')
df_2015_04 = pd.read_parquet('yellow_tripdata_2015-04.parquet')
df_2015_04.to_csv('yellow_tripdata_2015-04.csv')
df_2015_05 = pd.read_parquet('yellow_tripdata_2015-05.parquet')
df_2015_05.to_csv('yellow_tripdata_2015-05.csv')
df_2015_06 = pd.read_parquet('yellow_tripdata_2015-06.parquet')
df_2015_06.to_csv('yellow_tripdata_2015-06.csv')
df_2015_07 = pd.read_parquet('yellow_tripdata_2015-07.parquet')
df_2015_07.to_csv('yellow_tripdata_2015-07.csv')
df_2015_08 = pd.read_parquet('yellow_tripdata_2015-08.parquet')
df_2015_08.to_csv('yellow_tripdata_2015-08.csv')
df_2015_09 = pd.read_parquet('yellow_tripdata_2015-09.parquet')
df_2015_09.to_csv('yellow_tripdata_2015-09.csv')
df_2015_10 = pd.read_parquet('yellow_tripdata_2015-10.parquet')
df_2015_10.to_csv('yellow_tripdata_2015-10.csv')
df_2015_11 = pd.read_parquet('yellow_tripdata_2015-11.parquet')
df_2015_11.to_csv('yellow_tripdata_2015-11.csv')
df_2015_12 = pd.read_parquet('yellow_tripdata_2015-12.parquet')
df_2015_12.to_csv('yellow_tripdata_2015-12.csv')

dfs = [df_2015_01, df_2015_02, df_2015_03, df_2015_04, df_2015_05, df_2015_06,
       df_2015_07, df_2015_08, df_2015_09, df_2015_10, df_2015_11, df_2015_12]

combined_data = pd.concat(dfs, ignore_index=True)
print(combined_data.head())
