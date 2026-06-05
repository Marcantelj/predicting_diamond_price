# Predicting the price of dimonds based on ["carat","cut","color","clarity","depth","table", "width", "length", "height"]

import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error,r2_score
from sklearn.model_selection import train_test_split

df = pd.read_csv('Diamond Prediction/diamonds.csv')
df.rename(columns={"x" : "width", "y" : "length", "z":"height"}, inplace=True)

# Remapping all of are qualitative date into quantitative data for our model
cut_mapping = {"Fair" : 0, "Good": 1, "Very Good": 2, "Premium":3, "ideal": 4}
color_mapping = {"J":0, "I": 1, "H": 2, "G":3, "F": 4, "E":5, "D":6}
clarity_mapping = {"I1":0, "SI2": 1, "SI1": 2, "VS2":3, "VS1": 4, "VVS2":5, "VVS1":6, "IF":7}

# Applying our mapped data to the df.
df['cut'] = df['cut'].map(cut_mapping)
df['color'] = df['color'].map(color_mapping)
df['clarity'] = df['clarity'].map(clarity_mapping)

# 1. Separate features and target using clean, standard naming
FEATURES = ["carat","cut","color","clarity","depth","table", "width", "length", "height"]
X = df[FEATURES]
y = df["price"]

# 2. Split data with explicit stratify/shuffle intent if needed
X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.25, random_state=1
    )

# Estimators for RandomForestRegressor (n_estimator=estimator_range)
estimator_range = [10, 50, 100, 150, 200, 300, 500, 1000]

# Value placeholders for calculating optimal_tree, lowest_mae, and optimal_time.
optimal_tree = 0
lowest_mae = 0
optimal_time = 0

mae_results = []
time_results = []

for n in estimator_range:
    # start our timer
    start_time = time.time()

    # 3. Initialize the model with optimized hyper-parameters   
    # running all n_estimators in the estimator_range
    # n_jobs=-1 uses all CPU cores to speed up Random Forest training
    rf_model = RandomForestRegressor(n_estimators=n, n_jobs=-1, random_state=1)

    # 4. Train and predict model
    rf_model.fit(X_train, y_train)
    val_predictions = rf_model.predict(X_val)

    # Collect our end time and find out our elapsed_time
    end_time = time.time()
    elapsed_time = end_time - start_time

    # 5. Evaluate mean_absolute_error and print strings.
    val_mae = mean_absolute_error(y_val, val_predictions)
    print(f"Trees: {n}, Validation MAE for Random Forest Model: {val_mae:,.2f}, Execution Time: {elapsed_time:,.4f} seconds")
    mae_results.append(val_mae)
    time_results.append(elapsed_time)

    # calculating optimal_tree, lowest_mae, and optimal_time.
    if val_mae < lowest_mae or lowest_mae == 0:
        optimal_tree = n
        lowest_mae = val_mae
        optimal_time = elapsed_time

print(f"The optimal amount of n_estimators: {optimal_tree}, our lowest mean absolute error {lowest_mae:,.2f}, Execution Time: {optimal_time:,.4f} seconds.")
score = r2_score(y_val,val_predictions)
print(f'R^2 Score of {round(score * 100,2)}%.')

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Left Plot: Mean Absolute Error
ax1.plot(estimator_range, mae_results, marker='o', color='#1f77b4', linewidth=2)
ax1.set_title('Validation MAE vs. Number of Trees', fontsize=12, fontweight='bold')
ax1.set_xlabel('n_estimators (Number of Trees)')
ax1.set_ylabel('Mean Absolute Error (MAE)')
ax1.grid(True, linestyle='--', alpha=0.7)

# Right Plot: Training/Prediction Time
ax2.plot(estimator_range, time_results, marker='s', color='#ff7f0e', linewidth=2)
ax2.set_title('Execution Time vs. Number of Trees', fontsize=12, fontweight='bold')
ax2.set_xlabel('n_estimators (Number of Trees)')
ax2.set_ylabel('Time (Seconds)')
ax2.grid(True, linestyle='--', alpha=0.7)

# Save the plots to your working directory
plt.tight_layout()
plt.savefig('rf_optimization_plots.png', dpi=300)
print("Plots successfully saved as 'rf_optimization_plots.png'")