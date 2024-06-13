import datetime
import pandas as pd 
import joblib
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline


# Connect to the database
db_connect = psycopg2.connect(host='172.19.75.88', database='kaggle', user='kcw', password='sk1346')

# Load the data
df = pd.read_sql('SELECT * FROM train', db_connect)
X = df.drop(['id', 'target'], axis=1)
y = df['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Create a pipeline
model_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', SVC())
])
model_pipeline.fit(X_train, y_train)

# Predict
train_pred = model_pipeline.predict(X_train)
valid_pred = model_pipeline.predict(X_valid)

# Evaluate
train_acc = accuracy_score(y_train, train_pred)
valid_acc = accuracy_score(y_valid, valid_pred)

print("Train Accuracy :", train_acc)
print("Valid Accuracy :", valid_acc)

# Save the model
joblib.dump(model_pipeline, 'dev/model/output/model_pipeline.joblib')

# Save the data for validation after training
df.to_csv(f'data/train{datetime.datetime.now().strftime("%Y%m%d")}.csv', index=False)