# Eliptical galaxies were dominating by number so we add effects to the other ones to equalize the dataset without overfitting hopefully

import os
import shutil
import pandas as pd
import numpy as np
from PIL import Image, ImageEnhance
from tqdm import tqdm

# --- PARAMETERS ---
mapping_csv = 'dataset/processed/grouped_classifications.csv'
img_dir     = ''  # full paths are already in CSV
out_dir     = 'dataset/processed/images'
out_csv     = 'dataset/processed/augmented.csv'
TARGET      = 5826  # target per-class count

# Reset output directory
if os.path.exists(out_dir):
    shutil.rmtree(out_dir)
os.makedirs(out_dir, exist_ok=True)

# Load CSV
df = pd.read_csv(mapping_csv)

# Track all output records
records = []

# --- Copy original images to flat output directory ---
print("Copying original images...")
for _, r in tqdm(df.iterrows(), total=len(df)):
    src = r.image_path  # full path in CSV
    filename = os.path.basename(src)
    dst = os.path.join(out_dir, filename)

    try:
        img = Image.open(src)
        img.save(dst)
        records.append({'image_path': filename, 'galaxy_class': r.galaxy_class})
    except Exception as e:
        print(f"Error copying {src}: {e}")
        continue

# --- Define augmentations ---
def random_flip(img):
    if np.random.rand() < 0.5:
        img = img.transpose(Image.FLIP_LEFT_RIGHT)
    if np.random.rand() < 0.2:
        img = img.transpose(Image.FLIP_TOP_BOTTOM)
    return img

def random_rotate(img, max_angle=30):
    ang = np.random.uniform(-max_angle, max_angle)
    return img.rotate(ang, resample=Image.BILINEAR, expand=False)

def random_shear(img, max_shear=0.2):
    w, h = img.size
    sh = np.random.uniform(-max_shear, max_shear)
    a, b, c = 1, sh, 0
    d, e, f = 0, 1, 0
    return img.transform((w, h), Image.AFFINE, (a, b, c, d, e, f), resample=Image.BILINEAR)

def random_brightness(img, var=0.2):
    enhancer = ImageEnhance.Brightness(img)
    factor = 1.0 + np.random.uniform(-var, var)
    return enhancer.enhance(factor)

# --- Perform augmentations ---
print("\nAugmenting minority classes...")
class_counts = df.galaxy_class.value_counts().to_dict()

for cls, count in class_counts.items():
    needed = max(0, TARGET - count)
    if needed == 0:
        continue

    print(f"Augmenting {cls}: need {needed}")
    class_files = df[df.galaxy_class == cls].image_path.apply(os.path.basename).tolist()
    i = 0

    while i < needed:
        for fname in class_files:
            if i >= needed:
                break

            try:
                img = Image.open(os.path.join(out_dir, fname))
            except:
                continue

            # Apply augmentations
            aug = random_flip(img)
            aug = random_rotate(aug)
            aug = random_shear(aug)
            aug = random_brightness(aug)

            new_name = f"{os.path.splitext(fname)[0]}_aug{i}.jpg"
            out_path = os.path.join(out_dir, new_name)
            aug.save(out_path)

            records.append({'image_path': new_name, 'galaxy_class': cls})
            i += 1

# --- Save new mapping ---
out_df = pd.DataFrame(records)
out_df.to_csv(out_csv, index=False)

print("\n Done! Final dataset size:", len(out_df))
