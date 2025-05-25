# Trains the Forest using ALL Features
# This leads to suboptimal results

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

# File with all features
FEATURE_CSV = './../dataset/processed/all_features_cropped.csv'
TEST_SIZE = 0.2
RND_STATE = 42

# load data
df = pd.DataFrame()
try:
    df = pd.read_csv(FEATURE_CSV)
except Exception as e:
    print('Could not read feature file:', e)

# check label column
if 'label' not in df.columns:
    print('Error: No label column!')

# split into X and y
X = df.drop(columns=['label'], errors='ignore')
y = df['label'] if 'label' in df.columns else pd.Series()

# train/test split
try:
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, stratify=y, random_state=RND_STATE
    )
except Exception as e:
    print('Error during train/test split:', e)
    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    y_train = pd.Series()
    y_test = pd.Series()

# train the model
print('Training RandomForest on all features...')
rf = RandomForestClassifier(
    n_estimators=200,
    random_state=RND_STATE,
    class_weight='balanced'
)
try:
    rf.fit(X_train, y_train)
except Exception as e:
    print('Training failed:', e)

# predict and evaluate
y_pred = None
if len(X_test) > 0:
    try:
        y_pred = rf.predict(X_test)
    except Exception as e:
        print('Prediction failed:', e)

# print report
if y_pred is not None and len(y_pred) > 0:
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))

    # confusion matrix
    try:
        cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
        disp = ConfusionMatrixDisplay(cm, display_labels=rf.classes_)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
        plt.title('Confusion Matrix (All Features)')
        plt.tight_layout()
        try:
            plt.show()
        except:
            plt.savefig('./conf_matrix_all_features.png')
    except Exception as e:
        print('Could not plot confusion matrix:', e)
else:
    print('No predictions to report.')

print('Done.')
