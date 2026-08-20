import pickle
import pandas as pd
import matplotlib.pyplot as plt

# Load regression artifacts
with open("Regression_model.pkl", "rb") as f:
    reg_artifacts = pickle.load(f)

model = reg_artifacts["model"]
feature_names = reg_artifacts["feature_order"]

# Calculate and sort feature importances
importances = model.feature_importances_
importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importances
}).sort_values(by="Importance", ascending=False)

# Display top 10 most influential features
print(importance_df.head(10))

# Plot top 10 features
plt.figure(figsize=(10, 5))
plt.barh(importance_df["Feature"][:10][::-1], importance_df["Importance"][:10][::-1])
plt.xlabel("Importance Score")
plt.title("Top 10 Feature Importances (Random Forest Regressor)")
plt.tight_layout()
plt.show()