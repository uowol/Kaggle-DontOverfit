import joblib 
import pandas as pd
from sklearn.model_selection import train_test_split 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline 
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score


# Load the data
df = pd.read_csv('data/train.csv')
X = df.drop(['id', 'target'], axis=1)
y = df['target']
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
model_pipeline = Pipeline([('scaler', StandardScaler()), ('svc', SVC())])   # ('model_name', model_instance)
model_pipeline.fit(X_train, y_train)    

# 학습이 완료된 파이프라인은 바로 예측을 하거나 각 단계별로 진행해볼 수 있습니다.
# Example
# print(model_pipeline[0].transform(X_train[:1]))

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