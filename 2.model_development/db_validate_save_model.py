import datetime
import pandas as pd 
import joblib
import psycopg2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Reproduce the data
df = pd.read_csv(f'data/train{datetime.datetime.now().strftime("%Y%m%d")}.csv')
X = df.drop(['id', 'target'], axis=1)
y = df['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Load the model
model_pipeline_load = joblib.load('dev/model/output/model_pipeline.joblib')

# Validate
load_train_pred = model_pipeline_load.predict(X_train)
load_valid_pred = model_pipeline_load.predict(X_valid)

load_train_acc = accuracy_score(y_train, load_train_pred)
load_valid_acc = accuracy_score(y_valid, load_valid_pred)

print("Load Train Accuracy :", load_train_acc)
print("Load Valid Accuracy :", load_valid_acc)
