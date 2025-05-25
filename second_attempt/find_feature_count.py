# Tests number of features to find the best one

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

try:
    df = pd.read_csv('./../dataset/processed/all_features_cropped.csv')
except Exception as e:
    print('Could not read feature file:', e)
    df = pd.DataFrame()

if 'label' not in df.columns:
    print('Warning: no label column found in data')

# separate X and y
X = df.drop(columns=['label'], errors='ignore')
y = df.get('label')  # might be None if missing

# split into train/test
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
except Exception as e:
    print('Error splitting data:', e)
    X_train = X_test = y_train = y_test = pd.DataFrame()

# baseline random forest to get feature importance
print('Training base RandomForest to get importances...')
rf_base = RandomForestClassifier(n_estimators=100, random_state=42)
try:
    rf_base.fit(X_train, y_train)
except Exception as e:
    print('Training failed:', e)

# get importances
importances = pd.Series(rf_base.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)

# test several top-K sets
results = []
for K in [5, 10, 15, 20, 30, 50]:
    # pick top K features
    top_feats = importances.iloc[:K].index.tolist()
    print(f'Testing top {K} features: {top_feats[:3]}...')

    # new model
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    try:
        rf.fit(X_train[top_feats], y_train)
        y_pred = rf.predict(X_test[top_feats])
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='macro')
    except Exception as e:
        print(f'Error with K={K}:', e)
        acc = f1 = None

    results.append({
        'K': K,
        'accuracy': acc,
        'macro_f1': f1
    })

# show results in a DataFrame
res_df = pd.DataFrame(results).set_index('K')
print('\nResults for different feature counts:')
print(res_df)

# save results
try:
    res_df.to_csv('./../dataset/processed/feature_count_results.csv')
    print('Saved results to feature_count_results.csv')
except Exception as e:
    print('Could not save results:', e)

print('Done testing feature counts.')
