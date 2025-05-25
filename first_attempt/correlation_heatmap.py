import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load your training dataset
train_df = pd.read_csv('./../dataset/processed/train.csv')

# Select only numerical columns for correlation (excluding the target if categorical)
numerical_cols = train_df.select_dtypes(include=['float64', 'int64']).columns

# Calculate correlation matrix
corr = train_df[numerical_cols].corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Heatmap of Numerical Features')
plt.show()
