import joblib
import pandas as pd 
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


# Load the data
df = pd.read_csv('data/train.csv')
X = df.drop(['id', 'target'], axis=1)
y = df['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the model
scaler_load = joblib.load('dev/model/output/scaler.joblib')
classifier_load = joblib.load('dev/model/output/classifier.joblib')

# Validate
scaled_X_train = scaler_load.transform(X_train)
scaled_X_valid = scaler_load.transform(X_valid)

load_train_pred = classifier_load.predict(scaled_X_train)
load_valid_pred = classifier_load.predict(scaled_X_valid)

load_train_acc = accuracy_score(y_train, load_train_pred)
load_valid_acc = accuracy_score(y_valid, load_valid_pred)

print("Load Model Train Accuracy :", load_train_acc)
print("Load Model Valid Accuracy :", load_valid_acc)