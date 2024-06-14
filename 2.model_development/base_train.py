import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load the data
df = pd.read_csv('data/train.csv')
X = df.drop(['id', 'target'], axis=1)
y = df['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Scale the data
scaler = StandardScaler()
scaled_X_train = scaler.fit_transform(X_train)
scaled_X_valid = scaler.transform(X_valid)  # valid(test) data should be scaled with the same scaler as train data

# Train the model
classifier = SVC()  # default kernel='rbf'
classifier.fit(scaled_X_train, y_train)

# Predict
train_pred = classifier.predict(scaled_X_train)
valid_pred = classifier.predict(scaled_X_valid)

# Evaluate
train_acc = accuracy_score(y_train, train_pred)
valid_acc = accuracy_score(y_valid, valid_pred)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)

# Save the model
joblib.dump(scaler, 'dev/model/output/scaler.joblib')
joblib.dump(classifier, 'dev/model/output/classifier.joblib')

