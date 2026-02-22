import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# ===============================
# 1. Load CSV Dataset
# ===============================
# Make sure "marksheet (1).csv" is in the SAME folder as this file
df = pd.read_csv("marksheet (1).csv")

print("Dataset Loaded Successfully!")
print("\nColumns in dataset:")
print(df.columns)

print("\nFirst 5 rows:")
print(df.head())

# ===============================
# 2. Select Random Variable
# ===============================
# Change column name if needed (example: 'Marks', 'marks', 'Score')
X = df.iloc[:, 0]   # first column automatically taken

print("\nSelected Random Variable (Marks):")
print(df)

# ===============================
# 3. Type of Random Variable
# ===============================
print("\nType of Random Variable: Discrete Random Variable")

# ===============================
# 4. Probability Distribution (PMF)
# ===============================
pmf = X.value_counts(normalize=True).sort_index()

print("\nProbability Distribution (PMF):")
print(pmf)

# ===============================
# 5. Expectation, Variance, Std Dev
# ===============================
mean = np.mean(X)
variance = np.var(X)
std_dev = np.std(X)

print("\nStatistical Measures:")
print("Mean (Expectation):", mean)
print("Variance:", variance)
print("Standard Deviation:", std_dev)

# ===============================
# 6. Histogram Plot
# ===============================
plt.hist(X, bins=10)
plt.xlabel("Marks")
plt.ylabel("Number of Students")
plt.title("Histogram of Student Exam Marks")
plt.show()