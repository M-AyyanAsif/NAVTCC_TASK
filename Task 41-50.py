
import io
import statistics
import numpy as np
import pandas as pd
from scipy import stats

# =============================================================================
# TASK 41
# =============================================================================
"""
TASK 41: Pandas (Exercise on Pandas Basics / Tutorial)
Reference Link: https://www.w3schools.com/python/pandas_tutorial.asp

"""
print("=" * 70)
print("TASK 41: Pandas - Basics Tutorial Exercises")
print("=" * 70)

# Sample dataset
dataset = {
    'Student': ['Ali', 'Ayesha', 'Bilal', 'Fatima', 'Usman'],
    'Course': ['AI', 'Data Science', 'AI', 'Machine Learning', 'AI'],
    'Score': [92, 85, 78, 96, 88]
}

df_basic = pd.DataFrame(dataset)
print("Basic Pandas DataFrame:")
print(df_basic)
print(f"\nDataFrame Dimensions: {df_basic.shape[0]} rows x {df_basic.shape[1]} columns")


# =============================================================================
# TASK 42
# =============================================================================
"""
TASK 42: Pandas (Exercise on Pandas Getting Started & Installation)
Reference Link: https://www.w3schools.com/python/pandas_getting_started.asp

"""
print("\n" + "=" * 70)
print("TASK 42: Pandas - Installation & Getting Started")
print("=" * 70)

print(f"Pandas Version: {pd.__version__}")

# Simple car dataset
cars_data = {
    'cars': ["Toyota Corolla", "Honda Civic", "Suzuki Swift"],
    'passings': [3, 7, 2]
}
cars_df = pd.DataFrame(cars_data)
print("\nFirst Test DataFrame:")
print(cars_df)


# =============================================================================
# TASK 43
# =============================================================================
"""
TASK 43: Pandas (Exercise on Pandas Series Data)
Reference Link: https://www.w3schools.com/python/pandas_series.asp

"""
print("\n" + "=" * 70)
print("TASK 43: Pandas - Series Data Exercises")
print("=" * 70)

# 1. Series from list with default numeric index
temperatures = [31.5, 33.2, 30.0, 29.8]
s_default = pd.Series(temperatures)
print("1. Series with default index:")
print(s_default)

# 2. Series with custom labels
s_labeled = pd.Series(temperatures, index=["Day 1", "Day 2", "Day 3", "Day 4"])
print("\n2. Series with custom index labels:")
print(s_labeled)
print(f"   Value at 'Day 2': {s_labeled['Day 2']} °C")

# 3. Series from dictionary
calories_dict = {"Mon": 2100, "Tue": 1950, "Wed": 2200}
s_dict = pd.Series(calories_dict)
print("\n3. Series created from Dictionary:")
print(s_dict)


# =============================================================================
# TASK 44
# =============================================================================
"""
TASK 44: Pandas (Exercise on Pandas Data Frame)
Reference Link: https://www.w3schools.com/python/pandas_dataframes.asp

"""
print("\n" + "=" * 70)
print("TASK 44: Pandas - DataFrame Exercises")
print("=" * 70)

workout_data = {
    "calories": [420, 380, 390],
    "duration_mins": [50, 40, 45]
}

# 1. Standard index DataFrame
df_workout = pd.DataFrame(workout_data)
print("1. Workout DataFrame:")
print(df_workout)

# 2. Locating rows using .loc
print(f"\n2. Row at index 0 (Series):\n{df_workout.loc[0]}")
print(f"\n   Rows at index 0 and 1 (DataFrame):\n{df_workout.loc[[0, 1]]}")

# 3. Named index DataFrame
df_named = pd.DataFrame(workout_data, index=["day1", "day2", "day3"])
print(f"\n3. DataFrame with Named Indexes:\n{df_named}")
print(f"\n   Selecting 'day2':\n{df_named.loc['day2']}")


# =============================================================================
# TASK 45
# =============================================================================
"""
TASK 45: Pandas (Exercise on Pandas Open CSV Files)
Reference Link: https://www.w3schools.com/python/pandas_csv.asp

"""
print("\n" + "=" * 70)
print("TASK 45: Pandas - CSV File Ingestion Exercises")
print("=" * 70)

# Self-contained CSV simulation using io.StringIO
csv_buffer = io.StringIO("""Duration,Pulse,Maxpulse,Calories
60,110,130,409.1
60,117,145,479.0
60,103,135,340.0
45,109,175,282.4
45,117,148,406.0
60,102,127,300.0
""")

df_csv = pd.read_csv(csv_buffer)
print("DataFrame loaded from CSV stream:")
print(df_csv.to_string())

print(f"\nCurrent Pandas pd.options.display.max_rows = {pd.options.display.max_rows}")


# =============================================================================
# TASK 46
# =============================================================================
"""
TASK 46: Pandas (Exercise on Pandas Data Analyzing)
Reference Link: https://www.w3schools.com/python/pandas_analyzing.asp

"""
print("\n" + "=" * 70)
print("TASK 46: Pandas - Data Analyzing Exercises")
print("=" * 70)

# Creating representative dataset
np.random.seed(42)
records = {
    'Age': np.random.randint(18, 50, size=15),
    'Experience_Years': np.random.randint(1, 20, size=15),
    'Salary_k': np.random.randint(45, 180, size=15)
}
df_analysis = pd.DataFrame(records)

print("1. First 4 rows (head):")
print(df_analysis.head(4))

print("\n2. Last 3 rows (tail):")
print(df_analysis.tail(3))

print("\n3. Data Structure & Types (info):")
df_analysis.info()

print("\n4. Descriptive Statistics (describe):")
print(df_analysis.describe())


# =============================================================================
# TASK 47
# =============================================================================
"""
TASK 47: Pandas (Exercise on Pandas Data Cleaning Techniques)
Reference Link: https://www.w3schools.com/python/pandas_cleaning.asp

"""
print("\n" + "=" * 70)
print("TASK 47: Pandas - Data Cleaning Exercises")
print("=" * 70)

# Dirty dataset with nulls, format issues, and duplicates
dirty_data = {
    'Date': ['2026-10-01', '2026-10-02', '20261003', '2026-10-04', '2026-10-04'],
    'Duration': [60, 60, 450, 45, 45],  # 450 is a typo for 45
    'Pulse': [110, np.nan, 103, 109, 109],  # NaN value
    'Calories': [409.1, 479.0, 340.0, 282.4, 282.4]  # duplicate row
}
df_dirty = pd.DataFrame(dirty_data)
print("Raw Dirty Dataset:")
print(df_dirty)

# 1. Fill missing values with median
median_pulse = df_dirty['Pulse'].median()
df_dirty['Pulse'] = df_dirty['Pulse'].fillna(median_pulse)

# 2. Fix date formats
df_dirty['Date'] = pd.to_datetime(df_dirty['Date'], format='mixed')

# 3. Fix wrong value (Duration > 120 replaced with 45)
df_dirty.loc[df_dirty['Duration'] > 120, 'Duration'] = 45

# 4. Remove duplicate rows
df_cleaned = df_dirty.drop_duplicates()

print("\nCleaned Dataset:")
print(df_cleaned)


# =============================================================================
# TASK 48
# =============================================================================
"""
TASK 48: Pandas (Exercise on Pandas Data Correlation)
Reference Link: https://www.w3schools.com/python/pandas_correlations.asp

"""
print("\n" + "=" * 70)
print("TASK 48: Pandas - Correlation Exercises")
print("=" * 70)

study_data = {
    'Hours_Studied': [2, 3, 5, 7, 8, 10, 12],
    'Practice_Tests': [1, 2, 3, 4, 5, 7, 8],
    'Exam_Score': [55, 62, 70, 81, 85, 93, 98],
    'TV_Hours': [5, 4, 3, 2, 2, 1, 0]
}
df_study = pd.DataFrame(study_data)
corr_matrix = df_study.corr()

print("Study & Exam Performance Correlation Matrix:")
print(corr_matrix.round(3))

print("\nInterpretation:")
print(f"- Hours_Studied vs Exam_Score : {corr_matrix.loc['Hours_Studied', 'Exam_Score']:.2f} (Strong Positive)")
print(f"- TV_Hours vs Exam_Score      : {corr_matrix.loc['TV_Hours', 'Exam_Score']:.2f} (Strong Negative)")


# =============================================================================
# TASK 49
# =============================================================================
"""
TASK 49: Stats (Perform Mean, Median, and Mode)
Reference Link: https://www.w3schools.com/python/python_ml_mean_median_mode.asp

"""
print("\n" + "=" * 70)
print("TASK 49: Statistics - Mean, Median, and Mode")
print("=" * 70)

speed_records = [99, 86, 87, 88, 111, 86, 103, 87, 94, 78, 77, 85, 86]
print(f"Dataset: {speed_records}")

# Mean
mean_val = np.mean(speed_records)
# Median
median_val = np.median(speed_records)
# Mode using statistics module
mode_val = statistics.mode(speed_records)
# Mode using scipy
scipy_mode = stats.mode(speed_records, keepdims=True)

print(f"Mean (Average)           : {mean_val:.2f}")
print(f"Median (Middle value)    : {median_val}")
print(f"Mode (Most common value) : {mode_val} (scipy count: {scipy_mode.count[0]})")


# =============================================================================
# TASK 50
# =============================================================================
"""
TASK 50: Stats (Perform Standard Deviation & Variance)
Reference Link: https://www.w3schools.com/python/python_ml_standard_deviation.asp

"""
print("\n" + "=" * 70)
print("TASK 50: Statistics - Standard Deviation and Variance")
print("=" * 70)

# Dataset A: Consistent speeds
speeds_A = [86, 87, 88, 86, 87, 85, 86]
# Dataset B: Wide dispersion
speeds_B = [32, 111, 138, 28, 59, 77, 97]

std_A = np.std(speeds_A)
var_A = np.var(speeds_A)

std_B = np.std(speeds_B)
var_B = np.var(speeds_B)

print(f"Dataset A: {speeds_A}")
print(f"   Mean: {np.mean(speeds_A):.2f} | Variance: {var_A:.2f} | Standard Deviation: {std_A:.2f}")

print(f"\nDataset B: {speeds_B}")
print(f"   Mean: {np.mean(speeds_B):.2f} | Variance: {var_B:.2f} | Standard Deviation: {std_B:.2f}")

print("\nInsight: Low standard deviation indicates data points cluster close to the mean;")
print("high standard deviation indicates values are dispersed across a wider range.")
print("=" * 70)

