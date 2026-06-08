import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# 1. Create a sample house price dataset
data = {
    'Square_Feet': [1000, 1200, 1500, 1800, 2000, 2200, 2500, 2800, 3000, 3500],
    'Bedrooms':    [2, 3, 3, 4, 4, 5, 4, 5, 5, 6],
    'Bathrooms':   [1, 2, 2, 2, 3, 3, 3, 4, 4, 5],
    'Price':       [200000, 250000, 300000, 350000, 400000,
                    450000, 500000, 550000, 600000, 700000]
}

# Convert data into a Pandas DataFrame
df = pd.DataFrame(data)

# 2. Define Features (X) and Target variable (y)
A = df[['Square_Feet', 'Bedrooms', 'Bathrooms']]
B = df['Price']

# 3. Split data into training (80%) and testing (20%) sets
A_train, A_test, B_train, B_test = train_test_split(
    A, B, test_size=0.2, random_state=42
)

# 4. Create and train the Linear Regression model
model = LinearRegression()
model.fit(A_train, B_train)

# 5. Make predictions on the test set
B_pred = model.predict(A_test)

# 6. Evaluate the model's accuracy
mse = mean_squared_error(B_test, B_pred)
r2 = r2_score(B_test, B_pred)

print("Model Performance")
print("-----------------")
print(f"Mean Squared Error: {mse:,.2f}")
print(f"R2 Score:           {r2:.4f}")

# 7. Predict the price for a brand new house
new_house = pd.DataFrame({
    'Square_Feet': [2400],
    'Bedrooms': [4],
    'Bathrooms': [3]
})

# Make the prediction
predicted_price = model.predict(new_house)

print("\nNew House Details")
print("-----------------")
print(new_house.to_string(index=False)) # index=False hides the row number for a cleaner look

print(f"\nPredicted House Price: ${predicted_price[0]:,.2f}")
