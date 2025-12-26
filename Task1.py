import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# =========================
# LOAD DATA
# =========================
train_path = "./FILES/train.csv"
test_path = "./FILES/test.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

print("Training data loaded:")
print(train_df.head())

# =========================
# SELECT FEATURES
# =========================
features = [
    "GrLivArea",     # living area
    "BedroomAbvGr",  # bedrooms
    "FullBath"       # bathrooms
]

target = "SalePrice"

# =========================
# HANDLE MISSING VALUES
# =========================
for col in features:
    train_df[col] = train_df[col].fillna(train_df[col].median())
    test_df[col] = test_df[col].fillna(train_df[col].median())

# Drop rows where target is missing
train_df = train_df.dropna(subset=[target])

X = train_df[features]
y = train_df[target]

# =========================
# TRAIN TEST SPLIT
# =========================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================
# TRAIN MODEL
# =========================
model = LinearRegression()
model.fit(X_train, y_train)

# =========================
# PREDICTION
# =========================
y_pred = model.predict(X_test)

# =========================
# MODEL EVALUATION
# =========================
print("\nModel Performance:")
print("MAE :", mean_absolute_error(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))
print("R² Score:", r2_score(y_test, y_pred))

# =========================
# SAVE TRAIN PREDICTIONS
# =========================
results = X_test.copy()
results["ActualPrice"] = y_test
results["PredictedPrice"] = y_pred

results.to_csv("house_price_predictions.csv", index=False)
print("\nSaved: house_price_predictions.csv")

# =========================
# PREDICT TEST DATA
# =========================
test_features = test_df[features]
test_predictions = model.predict(test_features)

submission = pd.DataFrame({
    "Id": test_df["Id"],
    "SalePrice": test_predictions
})

submission.to_csv("submission.csv", index=False)
print("Saved: submission.csv")
