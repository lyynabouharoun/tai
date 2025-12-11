# scripts/process_images_extended.py

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage import img_as_float, img_as_ubyte
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.exposure import equalize_hist, equalize_adapthist

# ------------------------------
# Paths
# ------------------------------
DATA_DIR = "../data/kodak"
RESULTS_DIR = "../results"
METRICS_CSV = os.path.join(RESULTS_DIR, "metrics.csv")

# ------------------------------
# Helper functions
# ------------------------------
def ensure_dir(path):
    os.makedirs(path, exist_ok=True)

def save_img(path, img):
    if img.dtype in [np.float32, np.float64]:
        img = img_as_ubyte(img)
    cv2.imwrite(path, img)

def to_gray(img):
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def save_histogram(gray, path):
    plt.figure(figsize=(5,3))
    plt.hist(gray.ravel(), bins=256)
    plt.title("Histogram")
    plt.savefig(path)
    plt.close()

# ------------------------------
# Contrast
# ------------------------------
def equalization(gray):
    eq = cv2.equalizeHist(gray)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(gray)
    return eq, cl

# ------------------------------
# Noise
# ------------------------------
def add_gaussian_noise(gray, var=0.005):
    g = img_as_float(gray)
    noise = np.random.normal(0, np.sqrt(var), g.shape)
    noisy = np.clip(g + noise, 0, 1)
    return img_as_ubyte(noisy)

def add_salt_pepper(gray, amount=0.01):
    out = gray.copy()
    h, w = out.shape
    num = int(amount*h*w)
    # Salt
    coords = (np.random.randint(0,h,num), np.random.randint(0,w,num))
    out[coords] = 255
    # Pepper
    coords = (np.random.randint(0,h,num), np.random.randint(0,w,num))
    out[coords] = 0
    return out

# ------------------------------
# Filters
# ------------------------------
def box_filter(gray, k=5):
    return cv2.blur(gray, (k,k))

def gaussian_filter(gray, k=5):
    return cv2.GaussianBlur(gray, (k,k), 1.0)

def median_filter(gray, k=5):
    return cv2.medianBlur(gray, k)

def bilateral_filter(gray, d=9, sigmaColor=75, sigmaSpace=75):
    return cv2.bilateralFilter(gray, d, sigmaColor, sigmaSpace)

# ------------------------------
# Morphology
# ------------------------------
def morphology(gray):
    _, thr = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    K = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5,5))
    morph_ops = {
        "erode": cv2.erode(thr, K),
        "dilate": cv2.dilate(thr, K),
        "open": cv2.morphologyEx(thr, cv2.MORPH_OPEN, K),
        "close": cv2.morphologyEx(thr, cv2.MORPH_CLOSE, K),
        "tophat": cv2.morphologyEx(thr, cv2.MORPH_TOPHAT, K)
    }
    return morph_ops

# ------------------------------
# Segmentation
# ------------------------------
def segment(gray):
    _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU)
    return mask

# ------------------------------
# Metrics
# ------------------------------
def compute_metrics(ref, test):
    return psnr(ref, test, data_range=255), ssim(ref, test, data_range=255)

# ------------------------------
# Main pipeline
# ------------------------------
def process_image(path, metrics_list):
    filename = os.path.basename(path)
    name = filename.split('.')[0]
    print(f"\nProcessing {filename}...")

    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    gray = to_gray(img)

    # Histogram
    save_histogram(gray, f"{RESULTS_DIR}/histograms/{name}_hist.png")

    # Contrast
    eq, clahe = equalization(gray)
    save_img(f"{RESULTS_DIR}/equalized/{name}_eq.png", eq)
    save_img(f"{RESULTS_DIR}/equalized/{name}_clahe.png", clahe)

    # Noise
    g_noisy = add_gaussian_noise(gray)
    sp_noisy = add_salt_pepper(gray)
    save_img(f"{RESULTS_DIR}/noisy/{name}_gaussian.png", g_noisy)
    save_img(f"{RESULTS_DIR}/noisy/{name}_saltpepper.png", sp_noisy)

    # Filters
    filters = {
        "box": box_filter(gray),
        "gaussian": gaussian_filter(gray),
        "median": median_filter(gray),
        "bilateral": bilateral_filter(gray)
    }

    for f_name, f_img in filters.items():
        save_img(f"{RESULTS_DIR}/filtered/{name}_{f_name}.png", f_img)
        p, s = compute_metrics(gray, f_img)
        metrics_list.append({
            "image": name,
            "method": f_name,
            "PSNR": p,
            "SSIM": s
        })

    # Morphology
    morph = morphology(gray)
    for op, m_img in morph.items():
        save_img(f"{RESULTS_DIR}/morpho/{name}_{op}.png", m_img)

    # Combined processing chain: CLAHE -> Gaussian -> Morphology (opening)
    chain = morphology(gaussian_filter(clahe))["open"]
    save_img(f"{RESULTS_DIR}/filtered/{name}_chain_open.png", chain)
    p, s = compute_metrics(gray, chain)
    metrics_list.append({
        "image": name,
        "method": "chain_open",
        "PSNR": p,
        "SSIM": s
    })

    # Segmentation
    seg = segment(gray)
    save_img(f"{RESULTS_DIR}/segmentation/{name}_seg.png", seg)

# ------------------------------
# Run on all images
# ------------------------------
def main():
    # Create folders
    for sub in ["equalized", "noisy", "filtered", "morpho", "segmentation", "histograms"]:
        ensure_dir(os.path.join(RESULTS_DIR, sub))

    metrics_list = []

    # Process images
    for file in os.listdir(DATA_DIR):
        if file.lower().endswith((".png",".jpg",".jpeg")):
            process_image(os.path.join(DATA_DIR, file), metrics_list)

    # Save metrics CSV
    df = pd.DataFrame(metrics_list)
    df.to_csv(METRICS_CSV, index=False)
    print(f"\nMetrics saved to {METRICS_CSV}")

if __name__ == "__main__":
    main()
