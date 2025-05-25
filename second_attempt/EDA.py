# Statistics for the data

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

TRAIN_CSV = './../dataset/processed/train.csv'
TEST_CSV  = './../dataset/processed/test.csv'
OUT_DIR   = './../dataset/processed/eda_outputs'

os.makedirs(OUT_DIR, exist_ok=True)

def eda_report(df, name):
    print(f"\n\n=== EDA for {name} set ===\n")

    miss = df.isnull().sum()
    pct  = 100 * miss / len(df)
    missing = pd.DataFrame({'count': miss, 'percent': pct})
    print("Missing values per column:\n", missing, "\n")
    missing.to_csv(os.path.join(OUT_DIR, f'{name}_missing.csv'))

    desc_num = df.describe()
    print("Numeric descriptive stats:\n", desc_num, "\n")
    desc_num.to_csv(os.path.join(OUT_DIR, f'{name}_describe_numeric.csv'))

    print("Categorical value counts:")
    for col in df.select_dtypes(include=['object','category']).columns:
        vc = df[col].value_counts()
        print(f"\n{col}:\n", vc)
        vc.to_csv(os.path.join(OUT_DIR, f'{name}_vc_{col}.csv'))

    num_cols = df.select_dtypes(include=[np.number]).columns.drop('Unnamed: 0', errors='ignore')
    for col in num_cols:
        plt.figure(figsize=(4,3))
        sns.histplot(df[col], bins=30, kde=True)
        plt.title(f'{name} distribution: {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{name}_hist_{col}.png'))
        plt.close()

    for col in df.select_dtypes(include=['object','category']).columns:
        plt.figure(figsize=(4,3))
        sns.countplot(y=col, data=df, order=df[col].value_counts().index)
        plt.title(f'{name} countplot: {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{name}_count_{col}.png'))
        plt.close()

    for col in num_cols:
        plt.figure(figsize=(4,3))
        sns.boxplot(x=df[col])
        plt.title(f'{name} boxplot: {col}')
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{name}_box_{col}.png'))
        plt.close()

    corr = df[num_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=False, cmap='coolwarm', square=True)
    plt.title(f'{name} correlation heatmap')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f'{name}_corr_heatmap.png'))
    plt.close()

    top_cols = df[num_cols].var().sort_values(ascending=False).head(5).index
    for col in top_cols:
        plt.figure(figsize=(5,4))
        sns.violinplot(x='label', y=col, data=df, order=sorted(df['label'].unique()))
        plt.title(f'{name} violin: {col} vs label')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(OUT_DIR, f'{name}_violin_{col}.png'))
        plt.close()

    print(f"EDA for {name} complete; outputs in {OUT_DIR}/\n")

train_df = pd.read_csv(TRAIN_CSV)
test_df  = pd.read_csv(TEST_CSV)

eda_report(train_df, 'train')
eda_report(test_df, 'test')