import pandas as pd
import numpy as np
from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import joblib
import os

# 1. Load Dataset
print("Loading Wine dataset...")
data = load_wine()
df = pd.DataFrame(data.data, columns=data.feature_names)
df['cultivar'] = data.target

# 2. Feature Selection
# Selected features based on requirements:
# alcohol, flavanoids, color_intensity, hue, od280/od315_of_diluted_wines, proline
selected_features = [
    'alcohol',
    'flavanoids',
    'color_intensity',
    'hue',
    'od280/od315_of_diluted_wines',
    'proline'
]

X = df[selected_features]
y = df['cultivar']

print(f"Selected Features: {selected_features}")
print(f"Target Classes: {data.target_names}")

# 3. Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# 4. Pipeline Construction (Scaling + Modeling)
# Using StandardScaler for scaling as required
# Using RandomForestClassifier as the chosen algorithm
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', RandomForestClassifier(n_estimators=100, random_state=42))
])

# 5. Model Training
print("Training Random Forest Classifier...")
pipeline.fit(X_train, y_train)

# 6. Evaluation
print("Evaluating model...")
y_pred = pipeline.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=data.target_names)
conf_matrix = confusion_matrix(y_test, y_pred)

print(f"\nModel Accuracy: {accuracy:.4f}")
print("\nClassification Report:\n")
print(report)
print("\nConfusion Matrix:\n")
print(conf_matrix)

# 7. Save Model
model_path = 'model/wine_cultivar_model.pkl'
if not os.path.exists('model'):
    os.makedirs('model')

print(f"\nSaving model to {model_path}...")
joblib.dump(pipeline, model_path)
print("Model saved successfully.")

# Verification of saved file
if os.path.exists(model_path):
    print(f"Verification: File found at {model_path}")
else:
    print("Verification: File NOT found!")
