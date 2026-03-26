import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# dataset load
data = pd.read_csv("dataset.csv")

# text columns convert to numbers
data = pd.get_dummies(data)

# features and label
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# model
model = RandomForestClassifier(n_estimators=50)

# train
model.fit(X_train, y_train)

# prediction
pred = model.predict(X_test)

# accuracy
print("NIDS Model Accuracy:", accuracy_score(y_test, pred))