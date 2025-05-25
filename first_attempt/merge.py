import pandas as pd

# 1) Load mapping
corr_df = pd.read_csv('dataset/raw/id_mapping.csv', dtype={'objid': str})

# 2) Load features
features_df = pd.read_csv('dataset/raw/classification.csv', dtype={'dr7objid': str})

# 3) Merge on objid
merged_df = corr_df.merge(
    features_df,
    left_on='objid',
    right_on='dr7objid',
    how='inner'
)

# 4) Build jpg filename
merged_df['jpg_name'] = merged_df['asset_id'].astype(str) + '.jpg'

# 5) Inspect columns (optional, for debugging)
print("features_df.columns:", features_df.columns.tolist())
print("merged_df.columns:", merged_df.columns.tolist())

# 6) Select only the columns that exist
feature_cols = [
    col for col in features_df.columns
    if col != 'dr7objid' and col in merged_df.columns
]

final_columns = ['jpg_name'] + feature_cols
merged_df = merged_df[final_columns]

# 7) Save
merged_df.to_csv('dataset/processed/image_features.csv', index=False)
