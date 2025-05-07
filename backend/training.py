import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping, ModelCheckpoint
import pickle
import os

# Set random seeds for reproducibility
np.random.seed(42)
import tensorflow as tf
tf.random.set_seed(42)

# Load and preprocess data
print("Loading and preprocessing data...")
ipl = pd.read_csv("ipl_data.csv")
df = ipl[["bat_team", "bowl_team", "overs", "runs", "wickets", "total"]].copy()
df.columns = ["batting_team", "bowling_team", "overs", "runs", "wickets", "total"]

# Exploratory Data Analysis
print("\nData Overview:")
print(df.head())
print("\nData Summary:")
print(df.describe())

# Check for any missing values
print("\nMissing Values:")
print(df.isnull().sum())

# Show team distribution
plt.figure(figsize=(12, 6))
sns.countplot(y=df['batting_team'], order=df['batting_team'].value_counts().index)
plt.title('Distribution of Batting Teams')
plt.tight_layout()
plt.savefig('batting_teams_distribution.png')

# Visualize relationship between features and target
fig, axs = plt.subplots(1, 3, figsize=(18, 6))
sns.scatterplot(x='overs', y='total', data=df, ax=axs[0])
axs[0].set_title('Overs vs Total Score')

sns.scatterplot(x='runs', y='total', data=df, ax=axs[1])
axs[1].set_title('Runs vs Total Score')

sns.scatterplot(x='wickets', y='total', data=df, ax=axs[2])
axs[2].set_title('Wickets vs Total Score')
plt.tight_layout()
plt.savefig('feature_relationships.png')

# Correlation heatmap
plt.figure(figsize=(10, 8))
numeric_df = df.select_dtypes(include=[np.number])
correlation = numeric_df.corr()
mask = np.triu(correlation)
sns.heatmap(correlation, annot=True, cmap='coolwarm', mask=mask)
plt.title('Correlation Between Numeric Features')
plt.tight_layout()
plt.savefig('correlation_heatmap.png')

# Label encode teams
print("\nEncoding categorical features...")
label_encoders = {}
for col in ["batting_team", "bowling_team"]:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col])
    label_encoders[col] = le
    
    # Print mapping for reference
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print(f"\n{col} Encoding:")
    for team, code in mapping.items():
        print(f"  {team} -> {code}")

# Define features and target
X = df.drop("total", axis=1)
y = df["total"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"\nTraining set size: {X_train.shape[0]} samples")
print(f"Testing set size: {X_test.shape[0]} samples")

# Scale features
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the MLP model with slight improvements
print("\nBuilding and training the model...")
model = Sequential([
    Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
    Dropout(0.2),  # Add dropout for regularization
    Dense(64, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

# Compile model with additional metrics
model.compile(
    optimizer='adam', 
    loss='mse', 
    metrics=['mae', 'mape']  # Add Mean Absolute Error and Mean Absolute Percentage Error
)

# Model summary
model.summary()

# Add callbacks for better training
callbacks = [
    EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True),
    ModelCheckpoint('best_ipl_model.h5', monitor='val_loss', save_best_only=True)
]

# Train the model and store history
history = model.fit(
    X_train_scaled, 
    y_train, 
    epochs=150, 
    validation_split=0.2, 
    callbacks=callbacks,
    verbose=1
)

# Plot training history
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Loss (MSE)')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['mae'], label='Training MAE')
plt.plot(history.history['val_mae'], label='Validation MAE')
plt.title('Model MAE Over Epochs')
plt.xlabel('Epochs')
plt.ylabel('Mean Absolute Error')
plt.legend()

plt.tight_layout()
plt.savefig('training_history.png')

# Evaluate the model
print("\nEvaluating model performance...")
# Load the best model (usually has better performance than the final epoch)
if os.path.exists('best_ipl_model.h5'):
    model = load_model('best_ipl_model.h5')
    print("Loaded best model from checkpoint.")

# Get predictions
y_pred_train = model.predict(X_train_scaled).flatten()
y_pred_test = model.predict(X_test_scaled).flatten()

# Calculate metrics
train_mse = mean_squared_error(y_train, y_pred_train)
test_mse = mean_squared_error(y_test, y_pred_test)
train_rmse = np.sqrt(train_mse)
test_rmse = np.sqrt(test_mse)
train_mae = mean_absolute_error(y_train, y_pred_train)
test_mae = mean_absolute_error(y_test, y_pred_test)
train_r2 = r2_score(y_train, y_pred_train)
test_r2 = r2_score(y_test, y_pred_test)

# Print metrics
print(f"\nTraining MSE: {train_mse:.2f}")
print(f"Testing MSE: {test_mse:.2f}")
print(f"Training RMSE: {train_rmse:.2f} runs")
print(f"Testing RMSE: {test_rmse:.2f} runs")
print(f"Training MAE: {train_mae:.2f} runs")
print(f"Testing MAE: {test_mae:.2f} runs")
print(f"Training R²: {train_r2:.4f}")
print(f"Testing R²: {test_r2:.4f}")

# Visualize predictions vs actual
plt.figure(figsize=(12, 10))

plt.subplot(2, 1, 1)
plt.scatter(y_train, y_pred_train, alpha=0.5)
plt.plot([y_train.min(), y_train.max()], [y_train.min(), y_train.max()], 'r--')
plt.xlabel('Actual Total')
plt.ylabel('Predicted Total')
plt.title('Training: Actual vs Predicted Total Scores')

plt.subplot(2, 1, 2)
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
plt.xlabel('Actual Total')
plt.ylabel('Predicted Total')
plt.title('Testing: Actual vs Predicted Total Scores')

plt.tight_layout()
plt.savefig('prediction_performance.png')

# Analyze prediction errors
train_errors = y_pred_train - y_train
test_errors = y_pred_test - y_test

plt.figure(figsize=(12, 10))

plt.subplot(2, 2, 1)
plt.hist(train_errors, bins=30, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Training Error Distribution')
plt.xlabel('Prediction Error (runs)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 2)
plt.hist(test_errors, bins=30, alpha=0.7)
plt.axvline(x=0, color='r', linestyle='--')
plt.title('Testing Error Distribution')
plt.xlabel('Prediction Error (runs)')
plt.ylabel('Frequency')

plt.subplot(2, 2, 3)
plt.scatter(y_train, train_errors, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Training: Actual vs Error')
plt.xlabel('Actual Total')
plt.ylabel('Prediction Error (runs)')

plt.subplot(2, 2, 4)
plt.scatter(y_test, test_errors, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('Testing: Actual vs Error')
plt.xlabel('Actual Total')
plt.ylabel('Prediction Error (runs)')

plt.tight_layout()
plt.savefig('error_analysis.png')

# Save model and preprocessors
print("\nSaving model and preprocessors...")
model.save("ipl_mlp_model.h5")
with open("label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)
with open("scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("\nModel and preprocessors saved!")

# Function to make a prediction for a new match
def predict_score(batting_team, bowling_team, current_overs, current_runs, current_wickets):
    # Transform team names using the label encoders
    try:
        batting_team_encoded = label_encoders['batting_team'].transform([batting_team])[0]
        bowling_team_encoded = label_encoders['bowling_team'].transform([bowling_team])[0]
    except ValueError:
        print(f"Error: Team names must be from the training data.")
        return None
    
    # Create input features
    X_new = np.array([[batting_team_encoded, bowling_team_encoded, 
                       current_overs, current_runs, current_wickets]])
    
    # Scale features
    X_new_scaled = scaler.transform(X_new)
    
    # Make prediction
    predicted_score = model.predict(X_new_scaled)[0][0]
    
    return predicted_score

# Example prediction
print("\nExample Prediction:")
# Get the first team name from the encoders to use in the example
first_batting_team = label_encoders['batting_team'].classes_[0]
first_bowling_team = label_encoders['bowling_team'].classes_[1]

predicted_score = predict_score(
    batting_team=first_batting_team,
    bowling_team=first_bowling_team,
    current_overs=10.0,
    current_runs=80,
    current_wickets=2
)

print(f"Predicted total for {first_batting_team} against {first_bowling_team} " 
      f"(current: 10.0 overs, 80 runs, 2 wickets): {predicted_score:.1f} runs")

# Create another example with different match situation
predicted_score2 = predict_score(
    batting_team=first_batting_team,
    bowling_team=first_bowling_team,
    current_overs=15.0,
    current_runs=120,
    current_wickets=4
)

print(f"Predicted total for {first_batting_team} against {first_bowling_team} " 
      f"(current: 15.0 overs, 120 runs, 4 wickets): {predicted_score2:.1f} runs")

print("\nDone! Check the saved PNG files for visualizations.")