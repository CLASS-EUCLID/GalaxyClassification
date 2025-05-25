# Extracts all features
# Trains a model on all of them
# Chooses best 15
# Trains a model on the top 15

import os
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm
from scipy.ndimage import rotate, gaussian_filter, sobel
from skimage.feature import hog
import tensorflow as tf
from tensorflow.keras.applications import ResNet50, resnet50
from tensorflow.keras.preprocessing.image import img_to_array
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

IMAGE_DIR   = './../dataset/processed/cropped'
MAP_CSV     = './../dataset/processed/augmented.csv'     
OUT_FEATURE = './../dataset/processed/all_features_cropped.csv'
TEST_SIZE   = 0.2
RND_STATE   = 42

df_map = pd.read_csv(MAP_CSV)

cnn_model = ResNet50(weights='imagenet', include_top=False, pooling='avg',
                     input_shape=(224,224,3))

def concentration_index(img):
    h,w = img.shape
    y,x = np.indices((h,w))
    r = np.sqrt((x-w/2)**2 + (y-h/2)**2).ravel()
    I = img.ravel()
    order = np.argsort(r)
    cumI = np.cumsum(I[order])
    tot = cumI[-1] + 1e-8
    r20 = r[order][np.searchsorted(cumI, 0.2*tot)]
    r80 = r[order][np.searchsorted(cumI, 0.8*tot)]
    return 5*np.log10((r80+1e-8)/(r20+1e-8))

def ellipticity(img):
    y,x = np.indices(img.shape)
    I = img.astype(float)
    tot = I.sum() + 1e-8
    x0 = (x*I).sum()/tot; y0 = (y*I).sum()/tot
    x2 = (I*(x-x0)**2).sum()/tot; y2 = (I*(y-y0)**2).sum()/tot
    xy = (I*(x-x0)*(y-y0)).sum()/tot
    trace = x2+y2; det = x2*y2-xy**2
    lam1 = trace/2 + np.sqrt(max((trace/2)**2-det,0))
    lam2 = trace/2 - np.sqrt(max((trace/2)**2-det,0))
    return 1 - (lam2+1e-8)/(lam1+1e-8)

def gini_coefficient(img):
    arr = img.ravel(); arr = arr[arr>0]
    if arr.size==0: return 0
    arr = np.sort(arr)
    n = arr.size
    idx = np.arange(1,n+1)
    return (2*(idx*arr).sum())/(n*arr.sum()) - (n+1)/n

def m20_moment(img):
    h,w = img.shape
    y,x = np.indices((h,w))
    I = img.astype(float)
    tot = I.sum()+1e-8
    x0 = (x*I).sum()/tot; y0 = (y*I).sum()/tot
    Mtot = (I*((x-x0)**2+(y-y0)**2)).sum()
    flat = I.ravel(); idx = np.argsort(flat)[::-1]
    cum = np.cumsum(flat[idx]); cut = 0.2*tot
    sel = idx[cum<=cut]
    M20 = (flat[sel]*((x.ravel()[sel]-x0)**2 + (y.ravel()[sel]-y0)**2)).sum()
    return np.log10((M20+1e-8)/(Mtot+1e-8))

def asymmetry(img):
    rot = rotate(img, 180, reshape=False)
    return np.abs(img-rot).sum()/(img.sum()+1e-8)

def smoothness(img):
    blur = gaussian_filter(img, sigma=1)
    return np.abs(img-blur).sum()/(img.sum()+1e-8)

def edge_density(img):
    dx = sobel(img, axis=0); dy = sobel(img, axis=1)
    mag = np.hypot(dx,dy)
    thr = np.percentile(mag,75)
    return np.count_nonzero(mag>thr)/mag.size

def hog_feat(img):
    return hog(img, pixels_per_cell=(16,16), cells_per_block=(1,1), feature_vector=True)

def color_hist(path):
    im = Image.open(path).convert('RGB')
    arr = np.array(im)
    hists=[]
    for c in range(3):
        h,_ = np.histogram(arr[:,:,c], bins=8, range=(0,255))
        hists.append(h/np.sum(h))
    return np.concatenate(hists)

def cnn_emb(path):
    img = Image.open(path).convert('RGB').resize((224,224))
    x   = img_to_array(img)[None]
    x   = resnet50.preprocess_input(x)
    return cnn_model.predict(x, verbose=0).flatten()

rows = []
print("Extracting features:")
for _, r in tqdm(df_map.iterrows(), total=len(df_map)):
    full_path = r.image_path if os.path.isabs(r.image_path) else os.path.join(IMAGE_DIR, os.path.basename(r.image_path))
    if not os.path.exists(full_path):
        tqdm.write(f"[WARN] Missing file: {full_path}")
        continue

    try:
        imL = np.array(Image.open(full_path).convert('L'))
    except Exception as e:
        tqdm.write(f"[ERROR] Loading L {full_path}: {e}")
        continue

    feats = {
      'label': r.galaxy_class,
      'C': concentration_index(imL),
      'ellipticity': ellipticity(imL),
      'gini': gini_coefficient(imL),
      'M20': m20_moment(imL),
      'A': asymmetry(imL),
      'S': smoothness(imL),
      'edge_density': edge_density(imL),
      'mean_intensity': imL.mean()
    }

    try:
        hogv = hog_feat(imL)[:10]
        feats.update({f'HOG{i}':hogv[i] for i in range(len(hogv))})
    except Exception as e:
        tqdm.write(f"[ERROR] HOG {full_path}: {e}")

    try:
        ch = color_hist(full_path)[:6]
        feats.update({f'CH{i}':ch[i] for i in range(len(ch))})
    except Exception as e:
        tqdm.write(f"[ERROR] ColorHist {full_path}: {e}")

    try:
        emb = cnn_emb(full_path)[:50]
        feats.update({f'CNN{i}':emb[i] for i in range(len(emb))})
    except Exception as e:
        tqdm.write(f"[ERROR] CNN {full_path}: {e}")

    rows.append(feats)

df_all = pd.DataFrame(rows)
os.makedirs(os.path.dirname(OUT_FEATURE), exist_ok=True)
df_all.to_csv(OUT_FEATURE, index=False)
print("Saved feature matrix:", df_all.shape)

print("Training RandomForest for importances…")
X = df_all.drop(columns=['label'])
y = df_all['label']
X_train,X_test,y_train,y_test = train_test_split(X, y, test_size=TEST_SIZE,
                                                stratify=y, random_state=RND_STATE)

rf0 = RandomForestClassifier(n_estimators=200, random_state=RND_STATE)
rf0.fit(X_train, y_train)
importances = pd.Series(rf0.feature_importances_, index=X.columns)
top15 = importances.nlargest(15).index.tolist()
print("Top15 features:", top15)

print("Retraining on top15")
rf1 = RandomForestClassifier(n_estimators=200, random_state=RND_STATE)
rf1.fit(X_train[top15], y_train)
y_pred = rf1.predict(X_test[top15])

print("\nClassification Report:")
print(classification_report(y_test, y_pred))
cm = confusion_matrix(y_test, y_pred, labels=rf1.classes_)
plt.figure(figsize=(6,5))
sns.heatmap(cm, annot=True, fmt='d',
            xticklabels=rf1.classes_, yticklabels=rf1.classes_,
            cmap='Blues')
plt.title("Confusion Matrix on Top 15 Features")
plt.ylabel('True'); plt.xlabel('Predicted')
plt.show()