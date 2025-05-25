# Tests number of features to find the best one

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

df = pd.read_csv('./../dataset/processed/all_features_cropped.csv')
X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

base_rf = RandomForestClassifier(n_estimators=200, random_state=42)
base_rf.fit(X_train, y_train)
importances = pd.Series(base_rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

results = []
for K in [5, 10, 15, 20, 30, 50]:
    topK = importances.iloc[:K].index
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train[topK], y_train)
    y_pred = rf.predict(X_test[topK])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    results.append({'K': K, 'accuracy': acc, 'macro_f1': f1})

res_df = pd.DataFrame(results).set_index('K')
print(res_df)