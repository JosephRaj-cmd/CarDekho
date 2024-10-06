import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset with low_memory=False
df = pd.read_csv("C:\\Users\\Joseph\\Desktop\\cardheko cleaned data\\car_dekho_cleaned_dataset.csv", low_memory=False)

# Convert all columns to numeric where possible
for col in df.columns:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Drop rows with NaN target values
df = df.dropna(subset=['price'])

# Fill NaN values in features with column means
df = df.fillna(df.mean())

# Define features and target
X = df[['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName', 'City', 'mileage', 'Seats']]
y = df['price']

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Function to evaluate model performance
def evaluate_model(model, X_test, y_test, y_pred, model_name):
    mse = mean_squared_error(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    print(f'{model_name} - MSE: {mse}, MAE: {mae}, R²: {r2}')

    # Plotting Actual vs Predicted
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=y_test, y=y_pred)
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
    plt.xlabel('Actual Prices')
    plt.ylabel('Predicted Prices')
    plt.title(f'{model_name}: Actual vs Predicted Prices')
    plt.show()

# 1. Linear Regression Model
print("Training Linear Regression Model...")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)
evaluate_model(lr_model, X_test, y_test, y_pred_lr, 'Linear Regression')

# 2. Ridge Regression Hyperparameter Tuning with GridSearchCV
print("Tuning Ridge Regression...")
ridge = Ridge()
ridge_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
ridge_grid = GridSearchCV(ridge, ridge_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
ridge_grid.fit(X_train, y_train)
print(f'Best Ridge Alpha: {ridge_grid.best_params_}')

# 3. Lasso Regression Hyperparameter Tuning with GridSearchCV
print("Tuning Lasso Regression...")
lasso = Lasso()
lasso_params = {'alpha': [0.01, 0.1, 1, 10, 100]}
lasso_grid = GridSearchCV(lasso, lasso_params, cv=5, scoring='neg_mean_squared_error', n_jobs=-1)
lasso_grid.fit(X_train, y_train)
print(f'Best Lasso Alpha: {lasso_grid.best_params_}')

# 4. Gradient Boosting Model
print("Training Gradient Boosting Model...")
gbr_model = GradientBoostingRegressor(random_state=42)
gbr_model.fit(X_train, y_train)
y_pred_gbr = gbr_model.predict(X_test)
evaluate_model(gbr_model, X_test, y_test, y_pred_gbr, 'Gradient Boosting')

# Hyperparameter Tuning using RandomizedSearchCV for Gradient Boosting
print("Tuning Gradient Boosting Hyperparameters...")
gbr_params = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    'learning_rate': [0.05, 0.1],
    'subsample': [0.8, 1.0]
}
gbr_random = RandomizedSearchCV(
    gbr_model,
    gbr_params,
    n_iter=4,
    cv=3,
    scoring='neg_mean_squared_error',
    random_state=42,
    n_jobs=-1
)
gbr_random.fit(X_train, y_train)
print(f'Best Gradient Boosting Params: {gbr_random.best_params_}')

# 5. Decision Tree Hyperparameter Tuning with GridSearchCV
print("Tuning Decision Tree...")
dt_model = DecisionTreeRegressor(random_state=42)
dt_params = {
    'max_depth': [3, 5, 7],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2]
}
dt_grid = GridSearchCV(dt_model, dt_params, cv=3, scoring='neg_mean_squared_error', n_jobs=-1)
dt_grid.fit(X_train, y_train)  # Ensure this line is executed to fit the model
print(f'Best Decision Tree Params: {dt_grid.best_params_}')

# Cross-Validation for Decision Tree
print("Performing Cross-Validation for Decision Tree...")
dt_cv_scores = cross_val_score(dt_grid.best_estimator_, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f'Decision Tree CV Mean MSE: {-dt_cv_scores.mean()}')

# Model Prediction for Decision Tree
y_pred_dt = dt_grid.best_estimator_.predict(X_test)  # Predict using the best fitted model
evaluate_model(dt_grid.best_estimator_, X_test, y_test, y_pred_dt, 'Decision Tree')

# 6. Random Forest Model
print("Training Random Forest Model...")
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Cross-Validation
print("Performing Cross-Validation for Random Forest...")
rf_cv_scores = cross_val_score(rf_model, X_train, y_train, cv=5, scoring='neg_mean_squared_error')
print(f'Random Forest CV Mean MSE: {-rf_cv_scores.mean()}')

# Model Prediction
y_pred_rf = rf_model.predict(X_test)

# Model Evaluation
mse_rf = mean_squared_error(y_test, y_pred_rf)
mae_rf = mean_absolute_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f'Random Forest - MSE: {mse_rf}, MAE: {mae_rf}, R²: {r2_rf}')

# Plotting Actual vs Predicted for Random Forest
plt.figure(figsize=(10, 6))
sns.scatterplot(x=y_test, y=y_pred_rf)
plt.xlabel('Actual Prices')
plt.ylabel('Predicted Prices')
plt.title('Random Forest: Actual vs Predicted Prices')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red')
plt.show()

# Hyperparameter Tuning using RandomizedSearchCV for Random Forest
print("Tuning Random Forest Hyperparameters...")
rf_params = {
    'n_estimators': [100, 200],  # Reduced for quicker tuning
    'max_depth': [10, 20],
    'min_samples_split': [2, 5],
    'min_samples_leaf': [1, 2, 4],
    'bootstrap': [True, False]
}
rf_random = RandomizedSearchCV(
    rf_model,
    rf_params,
    n_iter=5,  # Reduced for quicker tuning
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=42,
    verbose=1
)
rf_random.fit(X_train, y_train)

print(f'Best Random Forest Params: {rf_random.best_params_}')

# Summary and Comparison Table
model_results = {
    'Model': ['Linear Regression', 'Gradient Boosting', 'Decision Tree', 'Random Forest'],
    'MSE': [
        mean_squared_error(y_test, lr_model.predict(X_test)),
        mean_squared_error(y_test, gbr_model.predict(X_test)),
        mean_squared_error(y_test, y_pred_dt),
        mse_rf
    ],
    'MAE': [
        mean_absolute_error(y_test, lr_model.predict(X_test)),
        mean_absolute_error(y_test, gbr_model.predict(X_test)),
        mean_absolute_error(y_test, y_pred_dt),
        mae_rf
    ],
    'R²': [
        r2_score(y_test, lr_model.predict(X_test)),
        r2_score(y_test, gbr_model.predict(X_test)),
        r2_score(y_test, y_pred_dt),
        r2_rf
    ]
}

# Create a DataFrame for comparison
comparison_df = pd.DataFrame(model_results)

# Display the Model Comparison Table
print("Model Comparison Table:")
print(comparison_df)

# Identify the best model based on the highest R² and the lowest MSE/MAE
best_model_idx = comparison_df['R²'].idxmax()
best_model_name = comparison_df.loc[best_model_idx, 'Model']
best_model_mse = comparison_df.loc[best_model_idx, 'MSE']
best_model_mae = comparison_df.loc[best_model_idx, 'MAE']
best_model_r2 = comparison_df.loc[best_model_idx, 'R²']

# Print the summary of the best model
print("\nBest Model Summary:")
print(f"Best Model: {best_model_name}")
print(f"MSE: {best_model_mse}")
print(f"MAE: {best_model_mae}")
print(f"R²: {best_model_r2}")
