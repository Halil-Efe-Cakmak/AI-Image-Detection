import cv2
import numpy as np
import os
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

IMAGE_SIZE = (256, 256)
LBP_P = 8
LBP_R = 1

def extract_color_histogram(image, bins=(8, 8, 8)):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    hist = cv2.calcHist([image_rgb], [0, 1, 2], None, bins,
                        [0, 256, 0, 256, 0, 256])
    hist = cv2.normalize(hist, hist).flatten()
    return hist

def extract_grayscale_stats(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mean = np.mean(gray)
    std = np.std(gray)
    return np.array([mean, std])

def extract_edge_density(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Bu görselin float değil uint8 olduğundan emin ol
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)
    edges = cv2.Canny(gray, 100, 200)
    density = np.sum(edges) / edges.size
    return np.array([density])


def extract_lbp_hist(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if gray.dtype != np.uint8:
        gray = (gray * 255).astype(np.uint8)
    lbp = local_binary_pattern(gray, P=LBP_P, R=LBP_R, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, LBP_P + 3), range=(0, LBP_P + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist


def extract_features(image):
    if image.shape[2] == 4:
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

    resized = cv2.resize(image, IMAGE_SIZE)
    resized = resized.astype('float32') / 255.0
    features = np.concatenate([
        extract_color_histogram(resized),
        extract_grayscale_stats(resized),
        extract_edge_density(resized),
        extract_lbp_hist(resized)
    ])
    return features

def process_dataset(image_dir, label):
    X = []
    y = []
    print(f"📂 Processing: {image_dir}")
    for filename in tqdm(os.listdir(image_dir)):
        path = os.path.join(image_dir, filename)
        image = cv2.imread(path)
        if image is not None:
            try:
                feats = extract_features(image)
                X.append(feats)
                y.append(label)
            except Exception as e:
                print(f"❌ Error in {filename}: {e}")
    return np.array(X), np.array(y)
