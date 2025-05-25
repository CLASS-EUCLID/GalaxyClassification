import os
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import rotate, gaussian_filter, sobel

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
    # approximate ellipticity by ratio of major/minor axis of intensity moments
    y, x = np.indices(img.shape)
    total = img.sum()
    x_mean = (x * img).sum() / total
    y_mean = (y * img).sum() / total
    x_diff = x - x_mean
    y_diff = y - y_mean
    Ixx = (img * x_diff**2).sum() / total
    Iyy = (img * y_diff**2).sum() / total
    Ixy = (img * x_diff * y_diff).sum() / total
    # covariance matrix eigenvalues (major, minor axis)
    trace = Ixx + Iyy
    det = Ixx * Iyy - Ixy ** 2
    lambda1 = trace / 2 + np.sqrt((trace / 2) ** 2 - det)
    lambda2 = trace / 2 - np.sqrt((trace / 2) ** 2 - det)
    if lambda1 == 0:
        return 0
    return 1 - (lambda2 / lambda1)

def gini_coefficient(img):
    arr = img.ravel()
    if arr.size == 0:
        return 0
    arr = arr[arr > 0]  # consider only positive intensity pixels
    if arr.size == 0:
        return 0
    sorted_arr = np.sort(arr)
    n = arr.size
    index = np.arange(1, n + 1)
    return (2 * (index * sorted_arr).sum()) / (n * sorted_arr.sum()) - (n + 1) / n

def m20_moment(img):
    # calculate second-order moment M20
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
    M20 = ((brightest_pixels) * ((x.ravel() - x_centroid) ** 2 + (y.ravel() - y_centroid) ** 2)[:len(brightest_pixels)]).sum()
    if M_tot == 0:
        return 0
    return np.log10(M20 / M_tot + 1e-8)

def asymmetry(img):
    # asymmetry = normalized difference with 180 deg rotated image
    rotated = rotate(img, 180, reshape=False)
    diff = np.abs(img - rotated)
    return diff.sum() / (img.sum() + 1e-8)

def smoothness(img):
    # smoothness = sum of absolute differences between img and blurred img, normalized
    blurred = gaussian_filter(img, sigma=1)
    diff = np.abs(img - blurred)
    return diff.sum() / (img.sum() + 1e-8)

def edge_density(img):
    # edge density = fraction of edge pixels using Sobel filter
    dx = sobel(img, axis=0)
    dy = sobel(img, axis=1)
    mag = np.hypot(dx, dy)
    threshold = np.percentile(mag, 75)
    edges = mag > threshold
    return edges.sum() / img.size



# 1) Load your mapping of images → classes
df_map = pd.read_csv('dataset/processed/augmented.csv')

# --- Feature functions (same as before) ---
def concentration_index(img):
    h, w = img.shape
    y,x = np.indices((h,w))
    r = np.sqrt((x - w//2)**2 + (y - h//2)**2).ravel()
    I = img.ravel()
    order = np.argsort(r)
    cumI = np.cumsum(I[order])
    total = cumI[-1]
    r20 = r[order][np.searchsorted(cumI, 0.2*total)]
    r80 = r[order][np.searchsorted(cumI, 0.8*total)]
    return 5*np.log10(r80/r20 + 1e-8)

# … define ellipticity(img), gini_coefficient(img), m20_moment(img),
#    asymmetry(img), smoothness(img), mean_intensity(img), edge_density(img)
#    exactly as in the earlier script.

# 2) Loop and build feature rows
records = []
for _, row in df_map.iterrows():
    fname = row['image_path']
    label = row['galaxy_class']
    img = np.array(Image.open(os.path.join('dataset/processed/images', fname)).convert('L'))

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

# 3) Save to CSV
pd.DataFrame(records).to_csv('dataset/processed/augmented_galaxy_features.csv', index=False)
print("Extracted features for", len(records), "images")
