https://chatgpt.com/share/681e5f52-bb08-8013-b025-7620faf72c7f




# -----------------------------------------------
# 📦 1. Import Libraries (Tools we need)
# -----------------------------------------------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix

# -----------------------------------------------
# 📁 2. Load the Dataset
# -----------------------------------------------
df = pd.read_csv("heart_disease_uci(1).csv")

# -----------------------------------------------
# 🧹 3. Data Cleaning
# -----------------------------------------------

# Remove duplicate rows
df.drop_duplicates(inplace=True)

# Fill missing values
mean_cols = ['trestbps', 'chol', 'thalch', 'oldpeak', 'ca']  # Numeric columns
mode_cols = ['fbs', 'restecg', 'exang', 'slope', 'thal']     # Categorical columns

for col in mean_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mean())

for col in mode_cols:
    if col in df.columns:
        df[col] = df[col].fillna(df[col].mode()[0])

# -----------------------------------------------
# 🔗 4. Data Integration
# -----------------------------------------------

# Add a fake hospital_id column to simulate merged data
df["hospital_id"] = np.random.randint(100, 105, size=len(df))

# -----------------------------------------------
# 🔄 5. Data Transformation
# -----------------------------------------------

# Convert categorical columns to numbers
label_encoder = LabelEncoder()
categorical_cols = ['fbs', 'restecg', 'exang', 'slope', 'thal']
for col in categorical_cols:
    if col in df.columns:
        df[col] = label_encoder.fit_transform(df[col])

# Scale numeric columns to a common range
scaler = StandardScaler()
numeric_cols = ['age', 'trestbps', 'chol', 'thalch', 'oldpeak', 'ca']
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# -----------------------------------------------
# ⚠️ 6. Error Detection and Correction (optional)
# -----------------------------------------------

# Visualizing outliers using boxplots (uncomment to view)
# for col in numeric_cols:
#     plt.figure(figsize=(6, 4))
#     sns.boxplot(data=df[col])
#     plt.title(f"Boxplot of {col}")
#     plt.show()

# -----------------------------------------------
# 🧠 7. Model Building
# -----------------------------------------------

# Set target and features
target_col = 'num'
X = df.drop(target_col, axis=1)
y = df[target_col]

# Split into training and test sets (80/20)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# -----------------------------------------------
# 🧪 8. Make Predictions
# -----------------------------------------------

y_pred = model.predict(X_test)

# -----------------------------------------------
# 🎯 9. Evaluate Model
# -----------------------------------------------

# Print accuracy score
accuracy = accuracy_score(y_test, y_pred)
print("✅ Model Accuracy:", accuracy)

# Show confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
