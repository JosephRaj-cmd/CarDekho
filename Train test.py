import pandas as pd

df = pd.read_csv("C:\\Users\\Joseph\\Desktop\\cardheko cleaned data\\car_dekho_cleaned_dataset.csv")


#Train-Test Split For Cleaned Dataset

from sklearn.model_selection import train_test_split

# Assuming df is the preprocessed DataFrame
X = df[['ft', 'bt', 'km', 'transmission', 'ownerNo', 'oem', 'model', 'modelYear', 'variantName','City','mileage', 'Seats']]
y = df['price']

# One-hot encode categorical variables
X = pd.get_dummies(X, drop_first=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the lengths of the train and test datasets
print("X_train length:", len(X_train))
print("X_test length:", len(X_test))
print("y_train length:", len(y_train))
print("y_test length:", len(y_test))


