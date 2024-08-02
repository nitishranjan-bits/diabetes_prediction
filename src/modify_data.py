import pandas as pd

# Load the original dataset
df = pd.read_csv("data/diabetes.csv")

# Modify the dataset (e.g., remove some rows)
df_modified = df.sample(frac=0.8, random_state=42)

# Save the modified dataset
df_modified.to_csv("data/diabetes_modified.csv", index=False)

print("Dataset modified and saved as diabetes_modified.csv")
