# predicting_diamond_price

Diamond Price Prediction & Random Forest Optimization
This repository features a machine learning pipeline designed to predict the price of diamonds using a Random Forest Regressor. The project focuses on data preprocessing, ordinal categorical encoding, and hyperparameter tuning—specifically optimizing the number of estimators (n_estimators) to strike the perfect balance between model accuracy (Mean Absolute Error), computational efficiency (Training Time), and R^2 Score to measure the proportion of variance in diamond prices that our model can successfully predict.

💎 Features & Dataset
The model predicts a diamond's price based on its physical and qualitative attributes. The dataset is sourced from diamonds.csv and includes the following features:

Carat: The weight of the diamond.

Cut, Color, Clarity: Qualitative measurements of the diamond's quality.

Depth & Table: Physical proportions (percentages).

Length, Width, Height: Dimensions in mm (renamed from x, y, z for clarity).

🚀 Workflow & Methodology
1. Data Preprocessing & Feature Engineering

Column Renaming: Standardized the dimension columns from x, y, and z to length, width, and height to improve code readability.

Categorical Encoding: Converted qualitative features into quantitative ordinal values based on industry standard grading scales:

Python
cut_mapping = {"Fair": 0, "Good": 1, "Very Good": 2, "Premium": 3, "Ideal": 4}
color_mapping = {"J": 0, "I": 1, "H": 2, "G": 3, "F": 4, "E": 5, "D": 6}
clarity_mapping = {"I1": 0, "SI2": 1, "SI1": 2, "VS2": 3, "VS1": 4, "VVS2": 5, "VVS1": 6, "IF": 7}

2. Model Training Setup

Data Splitting: The dataset is split into 75% training data (to train the model) and 25% testing data (to evaluate predictive performance).

Hardware Acceleration: Initialized the RandomForestRegressor with n_jobs=-1 to utilize all available CPU cores, significantly speeding up the iterative training process.

3. Hyperparameter Optimization Loop

To find the most efficient model, a tracking loop iterates through a defined estimator_range (different numbers of trees). For each iteration, the script:

Records the execution start time.

Trains the model on X_train and predicts on X_test.

Calculates the elapsed training/prediction time.

Evaluates performance using Mean Absolute Error (MAE).

Captures and updates the placeholder variables tracking:

optimal_tree: The number of estimators yielding the lowest error.

lowest_mae: The best (lowest) MAE score achieved.

optimal_time: The time taken by the optimal model configuration.

R^2 Score: How much our model explains the variance of diamond pricing.

📊 Evaluation & Visualization
After the optimization loop completes, the script generates a dual-plot visualization to analyze the trade-offs of scaling the model:

Left Graph (Accuracy Trend): Plots Mean Absolute Error against the number of estimators to show where performance gains begin to diminish.

Right Graph (Efficiency Trend): Plots Training/Prediction Time against the number of estimators to visualize the computational cost of adding more trees.

Key Takeaway: This dual-plot allows us to identify the "elbow point" where adding more trees no longer significantly reduces MAE but continues to increase computational overhead.

Results:
    Trees: 10, Validation MAE for Random Forest Model: 280.29, Execution Time: 0.3001 seconds
    Trees: 50, Validation MAE for Random Forest Model: 265.94, Execution Time: 1.3782 seconds
    Trees: 100, Validation MAE for Random Forest Model: 264.57, Execution Time: 2.5430 seconds
    Trees: 150, Validation MAE for Random Forest Model: 264.07, Execution Time: 3.7779 seconds
    Trees: 200, Validation MAE for Random Forest Model: 263.62, Execution Time: 5.1087 seconds
    Trees: 300, Validation MAE for Random Forest Model: 263.36, Execution Time: 7.3073 seconds
    Trees: 500, Validation MAE for Random Forest Model: 263.15, Execution Time: 11.9356 seconds
    Trees: 1000, Validation MAE for Random Forest Model: 263.11, Execution Time: 23.2679 seconds

The optimal amount of n_estimators: 1000, our lowest mean absolute error 263.11, Execution Time: 23.2679 seconds.
R^2 Score of 98.12%.


🛠️ Technologies Used
Python 3
Pandas (Data manipulation and cleaning)
Scikit-Learn (Train/test splitting, Random Forest Regressor, MAE evaluation, and R^2)
Matplotlib (Data visualization)


