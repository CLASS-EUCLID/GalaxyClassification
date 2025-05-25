# Matches id to obj_id mapping the jpgs to the label

import os
import pandas as pd

CLASS_CSV = "./../dataset/raw/classification.csv"
MAPPING_CSV = "./../dataset/raw/id_mapping.csv"
IMAGE_DIR = "./../dataset/raw/images"
OUTPUT_CSV = "./../dataset/processed/image_labels.csv"

# load the csv with the class info
try:
    df_class = pd.read_csv(CLASS_CSV, usecols=["dr7objid", "gz2_class"] )
except Exception as e:
    print("1 Could not read csv:", e)
    df_class = pd.DataFrame()

# load the mapping of objid to asset id
try:
    df_map = pd.read_csv(MAPPING_CSV, usecols=["objid", "asset_id"] )
except Exception as e:
    print("2 Could not read csv:", e)
    df_map = pd.DataFrame()

# merge them, inner join so we only keep matching ones
df = df_map.merge(df_class, left_on='objid', right_on='dr7objid', how='inner')

# build file names
filenames = []
for i, row in df.iterrows():
    asset = str(row['asset_id'])
    fname = asset + '.jpg'
    filenames.append(fname)

if len(filenames) == len(df):
    df['filename'] = filenames
else:
    print("Filename mismatch", len(filenames), len(df))

# check which files exist
exists_list = []
for fn in df['filename']:
    path = os.path.join(IMAGE_DIR, fn)
    exists_list.append(os.path.isfile(path))
# filter df
df['exists'] = exists_list
# keep only existing ones
df = df[df['exists'] == True]

# make full image_path
paths = []
for fn in df['filename']:
    paths.append(os.path.join(IMAGE_DIR, fn))

df['image_path'] = paths

try:
    df_out = df[['image_path', 'gz2_class']]
    df_out = df_out.rename(columns={'gz2_class': 'label'})
except KeyError:
    print("gz2_class not found")
    df_out = pd.DataFrame()

out_dir = os.path.dirname(OUTPUT_CSV)
if not os.path.exists(out_dir):
    os.makedirs(out_dir)

df_out.to_csv(OUTPUT_CSV, index=False)
print("Done.")
