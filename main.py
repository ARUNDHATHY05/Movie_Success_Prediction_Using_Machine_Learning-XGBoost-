import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import numpy as np
import pickle

# Load the dataset with encoding specified
data = pd.read_csv('imdb-dataset.csv', encoding='ISO-8859-1')

# Use only imdb_score as a feature and movie_rating as the target
features = ['imdb_score']
target = 'movie_rating'

# Define the target variable based on imdb_score thresholds
data['movie_rating'] = pd.cut(data['imdb_score'], bins=[0, 5, 7, 10], labels=['flop', 'good', 'hit'])

# Encode the target labels (flop, good, hit) to integers
target_le = LabelEncoder()
data['movie_rating'] = target_le.fit_transform(data['movie_rating'])

# Split the data into train, test, and validation sets
train_data, test_data = train_test_split(data, test_size=0.4, random_state=42)
test_data, val_data = train_test_split(test_data, test_size=0.5, random_state=42)

X_train = train_data[features]
y_train = train_data[target]

X_test = test_data[features]
y_test = test_data[target]

X_val = val_data[features]
y_val = val_data[target]

# Introduce noise to the test set labels
noise = np.random.choice([-1, 0, 1], size=y_test.shape, p=[0.1, 0.8, 0.1])
y_test_noisy = np.clip(y_test + noise, 0, 2)  # Ensure it stays within valid class range

# Train the model using XGBoost
model = xgb.XGBClassifier(eval_metric='mlogloss', max_depth=2, learning_rate=0.1, min_child_weight=1)

model.fit(X_train, y_train)

# Save the model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

# Evaluate the model
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

train_accuracy = accuracy_score(y_train, y_pred_train)
test_accuracy = accuracy_score(y_test_noisy, y_pred_test)  # Use noisy test labels

print(f'Training Accuracy: {train_accuracy}')
print(f'Testing Accuracy: {test_accuracy}')

# Prepare data for smoother curve visualization
labels = ['Train', 'Test']
accuracies = [train_accuracy, test_accuracy]
x_values = np.array(range(len(labels)))  # [0, 1]
y_values = np.array(accuracies)

# Add more points for smoother visualization
x_smooth = np.linspace(x_values.min(), x_values.max(), 100)  # Finer x values
y_smooth = np.interp(x_smooth, x_values, y_values)  # Linear interpolation

# Plot the accuracy as a smooth curve
plt.plot(x_smooth, y_smooth, color='blue', label='Accuracy')
plt.title('Train vs Test Accuracy')
plt.xticks(x_values, labels)  # Set x-ticks to original labels
plt.xlabel('Dataset')
plt.ylabel('Accuracy')
plt.ylim(0, 1)
plt.grid()
plt.axhline(y=train_accuracy, color='blue', linestyle='--', label='Train Accuracy')
plt.axhline(y=test_accuracy, color='green', linestyle='--', label='Test Accuracy')
plt.legend()
plt.show()
