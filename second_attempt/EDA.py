# Statistics for the data

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# rmbr to change if dir change...
TRAIN_CSV = './../dataset/processed/train.csv'
TEST_CSV  = './../dataset/processed/test.csv'
OUT_DIR   = './../dataset/processed/eda_outputs'

# output
if not os.path.exists(OUT_DIR):
    try:
        os.makedirs(OUT_DIR)
    except Exception as e:
        print('Could not create output:', e)

def eda_report(df, name):
    print(f"\n--- EDA for {name} ---")
    # missing values
    miss = df.isnull().sum()
    pct = 100 * miss / len(df)
    miss_df = pd.DataFrame({'missing_count': miss, 'missing_pct': pct})
    print(miss_df)
    miss_df.to_csv(os.path.join(OUT_DIR, f'{name}_missing.csv'))

    # descriptive stats
    try:
        desc = df.describe(include='all')
        print(desc)
        desc.to_csv(os.path.join(OUT_DIR, f'{name}_describe.csv'))
    except Exception as e:
        print('Describe failed:', e)

    # plots for numeric columns
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in num_cols:
        plt.figure()
        try:
            sns.histplot(df[col].dropna(), kde=True)
        except:
            plt.hist(df[col].dropna())
        plt.title(f'{col} dist')
        plt.savefig(os.path.join(OUT_DIR, f'{name}_{col}_hist.png'))
        plt.close()

    # countplots for categorical
    cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    for col in cat_cols:
        plt.figure()
        try:
            sns.countplot(y=col, data=df)
        except:
            df[col].value_counts().plot(kind='barh')
        plt.title(f'{col} counts')
        plt.savefig(os.path.join(OUT_DIR, f'{name}_{col}_count.png'))
        plt.close()

    # boxplots and violin for top 3 numeric by variance
    if num_cols:
        var_sorted = df[num_cols].var().sort_values(ascending=False)
        top3 = var_sorted.head(3).index.tolist()
        for col in top3:
            plt.figure()
            try:
                sns.boxplot(x=df[col])
            except:
                pass
            plt.title(f'{col} boxplot')
            plt.savefig(os.path.join(OUT_DIR, f'{name}_{col}_box.png'))
            plt.close()

            plt.figure()
            if 'label' in df.columns:
                sns.violinplot(x='label', y=col, data=df)
            else:
                sns.violinplot(y=col, data=df)
            plt.title(f'{col} violin')
            plt.savefig(os.path.join(OUT_DIR, f'{name}_{col}_violin.png'))
            plt.close()

# load data
try:
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)
except Exception as err:
    print('Error loading CSVs:', err)
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()

eda_report(train_df, 'train')
eda_report(test_df, 'test')
