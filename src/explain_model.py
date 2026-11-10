
import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt

# Load the trained model
saved = joblib.load("models/student_model.pkl")
model = saved["model"]
features = saved["features"]

# Load student data
data = pd.read_csv("data/student-mat.csv", sep=";")
X = data[features]

# Select 50 students for explanation
sample = X.sample(n=50, random_state=42)

# Calculate SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(sample)

# Display feature importance
print("\n--- SHAP FEATURE IMPORTANCE ---")

importance = pd.Series(
    abs(shap_values).mean(axis=0),
    index=features
).sort_values(ascending=False)

print(importance)

# Create an explanation plot
shap.summary_plot(shap_values, sample, show=False)
plt.tight_layout()
plt.savefig("models/shap_summary.png", dpi=150, bbox_inches="tight")
plt.close()

print("\nSHAP explanation completed!")
print("Plot saved to models/shap_summary.png")
