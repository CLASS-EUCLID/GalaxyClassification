import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# 1) Load train and test data
train_df = pd.read_csv('dataset/processed/train.csv')
test_df = pd.read_csv('dataset/processed/test.csv')

# 2) Prepare features and target
X_train = train_df.drop(columns=['filename', 'class'])  # drop non-feature columns
y_train = train_df['class']

X_test = test_df.drop(columns=['filename', 'class'])
y_test = test_df['class']

# 3) Initialize and train the model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 4) Predict on test data
y_pred = model.predict(X_test)

# 5) Evaluate
print("Classification Report:")
print(classification_report(y_test, y_pred))

# 6) Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=model.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=model.classes_)
disp.plot(cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.show()
