import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import rotate, gaussian_filter, sobel
from tqdm import tqdm

def concentration_index(img):
    h, w = img.shape
    y, x = np.indices((h, w))
    r = np.sqrt((x - w // 2) ** 2 + (y - h // 2) ** 2).ravel()
    I = img.ravel()
    order = np.argsort(r)
    cumI = np.cumsum(I[order])
    total = cumI[-1]
    r20 = r[order][np.searchsorted(cumI, 0.2 * total)]
    r80 = r[order][np.searchsorted(cumI, 0.8 * total)]
    return 5 * np.log10((r80 + 1e-8) / (r20 + 1e-8))

def ellipticity(img):
    y, x = np.indices(img.shape)
    total = img.sum()
    if total == 0:
        return 0
    x_mean = (x * img).sum() / total
    y_mean = (y * img).sum() / total
    x_diff = x - x_mean
    y_diff = y - y_mean
    Ixx = (img * x_diff ** 2).sum() / total
    Iyy = (img * y_diff ** 2).sum() / total
    Ixy = (img * x_diff * y_diff).sum() / total
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy ** 2
    if trace ** 2 < 4 * det:
        return 0
    lambda1 = trace / 2 + np.sqrt((trace / 2) ** 2 - det)
    lambda2 = trace / 2 - np.sqrt((trace / 2) ** 2 - det)
    if lambda1 == 0:
        return 0
    return 1 - (lambda2 / lambda1)

def gini_coefficient(img):
    arr = img.ravel()
    arr = arr[arr > 0]
    if arr.size == 0:
        return 0
    sorted_arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return (2 * (index * sorted_arr).sum()) / (n * sorted_arr.sum()) - (n + 1) / n

def m20_moment(img):
    h, w = img.shape
    y, x = np.indices(img.shape)
    total_intensity = img.sum()
    if total_intensity == 0:
        return 0
    x_centroid = (x * img).sum() / total_intensity
    y_centroid = (y * img).sum() / total_intensity
    M_tot = ((img) * ((x - x_centroid) ** 2 + (y - y_centroid) ** 2)).sum()
    sorted_pixels = np.sort(img.ravel())[::-1]
    cumulative_sum = np.cumsum(sorted_pixels)
    twenty_percent = 0.2 * total_intensity
    try:
        idx_20 = np.where(cumulative_sum >= twenty_percent)[0][0]
    except IndexError:
        return 0
    brightest_pixels = sorted_pixels[: idx_20 + 1]
    # approximate moment of brightest pixels
    coords = np.column_stack(np.unravel_index(np.argsort(img.ravel())[::-1][:idx_20 + 1], img.shape))
    M20 = 0
    for y_pix, x_pix in coords:
        intensity = img[y_pix, x_pix]
        M20 += intensity * ((x_pix - x_centroid) ** 2 + (y_pix - y_centroid) ** 2)
    if M_tot == 0:
        return 0
    return np.log10(M20 / M_tot + 1e-8)

def asymmetry(img):
    rotated = rotate(img, 180, reshape=False)
    diff = np.abs(img - rotated)
    return diff.sum() / (img.sum() + 1e-8)

def smoothness(img):
    blurred = gaussian_filter(img, sigma=1)
    diff = np.abs(img - blurred)
    return diff.sum() / (img.sum() + 1e-8)

def edge_density(img):
    dx = sobel(img, axis=0)
    dy = sobel(img, axis=1)
    mag = np.hypot(dx, dy)
    threshold = np.percentile(mag, 75)
    edges = mag > threshold
    return edges.sum() / img.size

# Load your mapping csv
df_map = pd.read_csv('dataset/processed/augmented.csv')

records = []
print("Starting feature extraction...")
for _, row in tqdm(df_map.iterrows(), total=len(df_map)):
    fname = row['image_path']
    label = row['galaxy_class']
    img_path = os.path.join('dataset/processed/images', fname)

    try:
        img = Image.open(img_path).convert('L')
        img = np.array(img)
    except Exception as e:
        print(f"Failed to open {fname}: {e}")
        continue

    feats = {
        'filename': fname,
        'class': label,
        'C': concentration_index(img),
        'ellipticity': ellipticity(img),
        'gini': gini_coefficient(img),
        'M20': m20_moment(img),
        'A': asymmetry(img),
        'S': smoothness(img),
        'mean_intensity': img.mean(),
        'edge_density': edge_density(img)
    }
    records.append(feats)

out_path = 'dataset/processed/augmented_galaxy_features.csv'
pd.DataFrame(records).to_csv(out_path, index=False)
print(f"Feature extraction complete. Extracted features for {len(records)} images saved to {out_path}")
