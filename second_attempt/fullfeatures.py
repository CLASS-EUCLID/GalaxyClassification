# Trains the Forest using ALL Features
# This leads to suboptimal results

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

FEATURE_CSV = './../dataset/processed/all_features_cropped.csv'
TEST_SIZE   = 0.2
RND_STATE   = 42

df = pd.read_csv(FEATURE_CSV)

X = df.drop(columns=['label'])
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    stratify=y,
    random_state=RND_STATE
)

rf = RandomForestClassifier(n_estimators=200, random_state=RND_STATE, class_weight='balanced')
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

print("Classification Report:\n")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred, labels=rf.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=rf.classes_)
fig, ax = plt.subplots(figsize=(6,5))
disp.plot(ax=ax, cmap='Blues', xticks_rotation=45)
plt.title("Confusion Matrix (All Features)")
plt.tight_layout()
plt.show()