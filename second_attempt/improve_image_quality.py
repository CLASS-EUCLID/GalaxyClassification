# Crops the images so that only the galaxy in the center remains
# Not workign for 100% of the photos
# Has trouble with images that contain two or mmore galaxies in very close proximity

import os
import cv2
import numpy as np
from tqdm import tqdm

IMAGE_DIR  = './../dataset/processed/images'
CROP_DIR   = './../dataset/processed/cropped'
MIN_AREA   = 500    
PAD_FACTOR = 1.2    

os.makedirs(CROP_DIR, exist_ok=True)

def crop_central_galaxy(path):

    img = cv2.imread(path)
    if img is None: 
        return False
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5,5), 0)

    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    contours, _ = cv2.findContours(th, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    H,W = gray.shape
    cx, cy = W//2, H//2
    best = None
    best_dist = 1e9
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < MIN_AREA: 
            continue

        M = cv2.moments(cnt)
        if M['m00']==0: 
            continue
        x0, y0 = int(M['m10']/M['m00']), int(M['m01']/M['m00'])
        dist = (x0-cx)**2 + (y0-cy)**2
        if dist < best_dist:
            best_dist, best = dist, cnt

    if best is None:
        return False

    x,y,w,h = cv2.boundingRect(best)
    pad = int(max(w,h)*(PAD_FACTOR-1)/2)
    x0 = max(0, x - pad)
    y0 = max(0, y - pad)
    x1 = min(W, x + w + pad)
    y1 = min(H, y + h + pad)

    crop = img[y0:y1, x0:x1]
    out_path = os.path.join(CROP_DIR, os.path.basename(path))
    cv2.imwrite(out_path, crop)
    return True

print("Cropping central galaxy from images…")
for fname in tqdm(os.listdir(IMAGE_DIR)):
    full = os.path.join(IMAGE_DIR, fname)
    crop_central_galaxy(full)

print("Done! Cropped images in:", CROP_DIR)