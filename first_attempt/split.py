from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv('dataset/processed/augmented_galaxy_features.csv')
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['class'])

train_df.to_csv('dataset/processed/train.csv', index=False)
test_df.to_csv('dataset/processed/test.csv', index=False)
