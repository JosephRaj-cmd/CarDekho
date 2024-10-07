import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, RandomizedSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
import time

try:
    # Load dataset
    data = pd.read_csv(r"C:\Users\Joseph\Desktop\cardheko cleaned data\car_dekho_cleaned_dataset.csv", low_memory=False)
    print(f"Dataset loaded successfully with {data.shape[0]} rows and {data.shape[1]} columns.")

    # Load preprocessing steps
    label_encoders = joblib.load(r"C:\Users\Joseph\Desktop\cardheko cleaned data\label_encoders.pkl")
    scalers = joblib.load(r"C:\Users\Joseph\Desktop\cardheko cleaned data\scalers.pkl")

    # Feature Engineering
    data['car_age'] = 2024 - data['modelYear']
    brand_popularity = data.groupby('oem')['price'].mean().to_dict()
    data['brand_popularity'] = data['oem'].map(brand_popularity)
    data['mileage_normalized'] = data['mileage'] / data['car_age']

    # Define features and target
    features = ['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName', 'City', 'mileage', 'Seats', 'car_age', 'brand_popularity', 'mileage_normalized']
    X = data[features]
    y = data['price']
    print("Features and target defined.")

    # Split dataset
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print("Data split into training and testing sets.")

    # Initialize and train the model with RandomizedSearchCV for hyperparameter tuning
    rf_model = RandomForestRegressor(random_state=42)

    param_dist = {
        'n_estimators': [100, 150],  # Reduced number of estimators
        'max_depth': [10, 20, 30],   # Reduced depth
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True]
    }

    print("Starting RandomizedSearchCV for hyperparameter tuning...")
    start_time = time.time()
    rf_random = RandomizedSearchCV(rf_model, param_distributions=param_dist, n_iter=10, cv=3, scoring='neg_mean_squared_error', random_state=42, n_jobs=-1)  # Reduced iterations and parallel jobs
    rf_random.fit(X_train, y_train)
    end_time = time.time()
    print("RandomizedSearchCV completed.")

    # Best model
    best_rf_model = rf_random.best_estimator_
    print("Best model obtained from RandomizedSearchCV.")

    # Cross-Validation
    rf_cv_scores = cross_val_score(best_rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
    print(f'Random Forest CV Mean MSE: {-rf_cv_scores.mean()}')

    # Model Prediction
    print("Making predictions on the test set...")
    start_predict_time = time.time()
    y_pred_rf = best_rf_model.predict(X_test)
    end_predict_time = time.time()
    print("Predictions completed.")

    # Model Evaluation
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    r2_rf = r2_score(y_test, y_pred_rf)

    print(f'Random Forest - MSE: {mse_rf}, MAE: {mae_rf}, R²: {r2_rf}')
    print(f'Training Time: {end_time - start_time} seconds')
    print(f'Prediction Time: {end_predict_time - start_predict_time} seconds')

    # Evaluate on older cars
    older_cars = data[data['car_age'] > 10]  # Assuming older cars are those older than 10 years
    X_older = older_cars[features]
    y_older = older_cars['price']

    print("Making predictions on older cars...")
    y_pred_older = best_rf_model.predict(X_older)
    mse_older = mean_squared_error(y_older, y_pred_older)
    mae_older = mean_absolute_error(y_older, y_pred_older)
    r2_older = r2_score(y_older, y_pred_older)

    print(f'Older Cars - MSE: {mse_older}, MAE: {mae_older}, R²: {r2_older}')

    # Plotting Actual vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred_rf)
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title('Random Forest: Actual vs Predicted Prices')
    plt.show()

    # Save the trained model
    joblib.dump(best_rf_model, r'C:\Users\Joseph\Desktop\cardheko cleaned data\car_price_prediction_model.pkl')
    print("Model training complete. Model saved as 'car_price_prediction_model.pkl'.")

except Exception as e:
    print(f"An error occurred: {e}")
