# Eliptical galaxies were dominating by number so we add effects to the other ones to equalize the dataset without overfitting hopefully

import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

mapping_csv = './../dataset/processed/grouped_classifications.csv'
# img_dir left blank because CSV has full paths
img_dir = ''
out_dir = './../dataset/processed/images'
out_csv = './../dataset/processed/augmented.csv'
TARGET = 5826  # aim to have this many per class

# prepare output folder
if os.path.exists(out_dir):
    try:
        shutil.rmtree(out_dir)
    except Exception as e:
        print('Warning: could not clear folder', e)
os.makedirs(out_dir, exist_ok=True)

# load the classification CSV
df = pd.DataFrame()
try:
    df = pd.read_csv(mapping_csv)
except Exception as e:
    print('Error reading mapping CSV:', e)

records = []  # to store (image_path, galaxy_class)

print('Copying original images...')
for idx, row in tqdm(df.iterrows(), total=len(df)):
    src = row.image_path  # full path
    fname = os.path.basename(src)
    dst = os.path.join(out_dir, fname)
    try:
        img = Image.open(src)
        img.save(dst)
        records.append({'image_path': fname, 'galaxy_class': row.galaxy_class})
    except Exception as e:
        print(f"Skipped {src}: {e}")
        continue

# define augmentation functions
def random_flip(img):
    if np.random.rand() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.rand() < 0.3:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def random_rotate(img, angle=30):
    a = np.random.uniform(-angle, angle)
    return img.rotate(a)

def random_shear(img, shear=0.2):
    w, h = img.size
    s = np.random.uniform(-shear, shear)
    # simple horizontal shear
    return img.transform((w, h), Image.AFFINE, (1, s, 0, 0, 1, 0))

def random_brightness(img, var=0.2):
    enhancer = ImageEnhance.Brightness(img)
    f = 1.0 + np.random.uniform(-var, var)
    return enhancer.enhance(f)

# count per class
totals = df.galaxy_class.value_counts().to_dict()
print('\nAugmenting classes to target...')
for cls, cnt in totals.items():
    need = TARGET - cnt
    if need <= 0:
        continue
    print(f"{cls}: current {cnt}, need {need}")
    files = df[df.galaxy_class == cls].image_path.apply(os.path.basename).tolist()
    i = 0
    while i < need:
        for f in files:
            if i >= need:
                break
            p = os.path.join(out_dir, f)
            try:
                img = Image.open(p)
            except:
                print('Could not open', p)
                continue
            # apply random transformations
            new = random_flip(img)
            new = random_rotate(new)
            new = random_shear(new)
            new = random_brightness(new)

            new_name = f.replace('.jpg', '') + f'_aug{i}.jpg'
            save_path = os.path.join(out_dir, new_name)
            try:
                new.save(save_path)
                records.append({'image_path': new_name, 'galaxy_class': cls})
                i += 1
            except Exception as e:
                print('Save failed:', e)
                continue

# save augmented CSV
out_df = pd.DataFrame(records)
print(f"Saving augmented CSV with {len(out_df)} rows to {out_csv}")
out_dir2 = os.path.dirname(out_csv)
if not os.path.exists(out_dir2):
    os.makedirs(out_dir2, exist_ok=True)
try:
    out_df.to_csv(out_csv, index=False)
except Exception as e:
    print('CSV save error:', e)

print('All done.')

