
import pandas as pd

# Load the student dataset
data = pd.read_csv("data/student-mat.csv", sep=";")

# Show basic information
print("\n--- DATASET SIZE ---")
print("Rows and columns:", data.shape)

print("\n--- FIRST 5 STUDENTS ---")
print(data.head())

print("\n--- COLUMN NAMES ---")
print(data.columns.tolist())

print("\n--- MISSING VALUES ---")
print(data.isnull().sum().sum())

print("\n--- FINAL GRADE (G3) SUMMARY ---")
print(data["G3"].describe())

print("\n--- FINAL GRADE DISTRIBUTION ---")
print(data["G3"].value_counts().sort_index())
