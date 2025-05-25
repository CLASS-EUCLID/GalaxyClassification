import os
import cv2
import numpy as np
from tqdm import tqdm

IMAGE_DIR  = './../dataset/processed/images'
CROP_DIR   = './../dataset/processed/cropped'
MIN_AREA   = 500    # ignore tiny contours
PAD_FACTOR = 1.2    # enlarge crop by 20%

os.makedirs(CROP_DIR, exist_ok=True)

def crop_central_galaxy(path):
    # 1) Load & blur to reduce noise
    img = cv2.imread(path)
    if img is None: 
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    # 2) Threshold → binary mask
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # 3) Find contours
    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 4) Pick the largest contour near image center
    H,W = gray.shape
    cx, cy = W//2, H//2
    best = None
    best_dist = 1e9
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA: 
            continue
        # contour centroid
        M = cv2.moments(cnt)
        if M['m00']==0: 
            continue
        x0, y0 = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        dist = (x0-cx)**2 + (y0-cy)**2
        if dist < best_dist:
            best_dist, best = dist, cnt

    if best is None:
        return False

    # 5) Get bounding box and pad
    x,y,w,h = cv2.boundingRect(best)
    pad = int(max(w,h)*(PAD_FACTOR-1)/2)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    # 6) Crop & save
    crop = img[y0:y1, x0:x1]
    out_path = os.path.join(CROP_DIR, os.path.basename(path))
    cv2.imwrite(out_path, crop)
    return True

# Process all images
print("Cropping central galaxy from images…")
for fname in tqdm(os.listdir(IMAGE_DIR)):
    full = os.path.join(IMAGE_DIR, fname)
    crop_central_galaxy(full)

print("Done! Cropped images in:", CROP_DIR)
