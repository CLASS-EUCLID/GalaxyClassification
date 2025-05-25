import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

# 1) Load the full feature set
df = pd.read_csv('./../dataset/processed/all_features_cropped.csv')
X = df.drop(columns=['label'])
y = df['label']

# 2) Split once (so each experiment is comparable)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# 3) Train RF on all features to get importances
base_rf = RandomForestClassifier(n_estimators=200, random_state=42)
base_rf.fit(X_train, y_train)
importances = pd.Series(base_rf.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

# 4) Try different top-K subsets
results = []
for K in [5, 10, 15, 20, 30, 50]:
    topK = importances.iloc[:K].index
    rf = RandomForestClassifier(n_estimators=200, random_state=42)
    rf.fit(X_train[topK], y_train)
    y_pred = rf.predict(X_test[topK])
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='macro')
    results.append({'K': K, 'accuracy': acc, 'macro_f1': f1})

# 5) Show a table of results
res_df = pd.DataFrame(results).set_index('K')
print(res_df)
