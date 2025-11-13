import pandas as pd
import os

os.makedirs("results", exist_ok=True)

# Recreate your last best result manually
best_params = {
    "units": 64,
    "dropout_rate": 0.5,
    "learning_rate": 0.001,
    "val_mae": 46.231304
}

results_df = pd.DataFrame([best_params])
results_df.to_csv("results/hyperparameter_tuning_results.csv", index=False)
print("Results saved successfully!")
