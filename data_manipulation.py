import os
import pandas as pd

CLASS_CSV = 'dataset/raw/classification.csv'
MAPPING_CSV = 'dataset/raw/id_mapping.csv'
IMAGE_DIR = 'dataset/raw/images'
OUTPUT_CSV = 'dataset/processed/image_labels.csv'

df_class = pd.read_csv(CLASS_CSV, usecols=['dr7objid', 'gz2_class'])
df_map = pd.read_csv(MAPPING_CSV, usecols=['objid', 'asset_id'])

df = df_map.merge(df_class, left_on='objid', right_on='dr7objid', how='inner')

df['filename'] = df['asset_id'].astype(str) + '.jpg'

def image_exists(fname):
    return os.path.isfile(os.path.join(IMAGE_DIR, fname))

df['exists'] = df['filename'].apply(image_exists)
df = df[df['exists']].copy()

df['image_path'] = df['filename'].apply(lambda f: os.path.join(IMAGE_DIR, f))

df_out = df[['image_path', 'gz2_class']].rename(columns={'gz2_class': 'label'})

print(f"Saving {len(df_out)} entries to {OUTPUT_CSV}")
df_out.to_csv(OUTPUT_CSV, index=False)

if __name__ == '__main__':
    pass
