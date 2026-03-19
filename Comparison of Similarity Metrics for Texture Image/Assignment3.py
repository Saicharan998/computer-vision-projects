import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import time

# === Configuration ===
INPUT_DIR = "C:/Users/owner/Desktop/Real_Brodatz"
OUTPUT_DIR = "C:/Users/owner/Desktop/sub img"
TARGET_IMG_SIZE = 512
SUB_IMG_SIZE = 128


# === STEP 1: Resize and Subdivide Images ===
def resize_and_split_image(img_path, save_dir, target_size=512, sub_size=128):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"Skipping {img_path}: Unable to read.")
        return

    img = cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_AREA)
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    idx = 0


    for i in range(0, target_size, sub_size):
        for j in range(0, target_size, sub_size):
            sub_img = img[i:i+sub_size, j:j+sub_size]
            sub_img_name = f"{base_name}_sub_{idx:02d}.png"
            sub_img_path = os.path.join(save_dir, sub_img_name)
            cv2.imwrite(sub_img_path, sub_img)
            idx += 1


def process_all_images(input_dir, output_dir):
    image_files = [f for f in os.listdir(input_dir)
                   if os.path.isfile(os.path.join(input_dir, f))]
    total = 0
    for file_name in image_files:
        img_path = os.path.join(input_dir, file_name)
        resize_and_split_image(img_path, output_dir)
        total += 16
    print(f"Finished: Generated {total} sub-images from {len(image_files)} images.")

# === Parameters as per paper ===
sigma_x = 2.5
sigma_y = 2.5
W_x_list = [0.05, 0.1, 0.2, 0.4]  # Frequency band values
orientations_deg = [0, 30, 60, 90, 120, 150]
kernel_size = 31
FEATURE_SAVE_PATH = "C:/Users/owner/Desktop/gabor_features.npy"

# === STEP 2: Gabor Filter Definition with Scaling ===
def gabor_function_from_paper(x, y, sigma_x, sigma_y, W_x, theta, a_power_m_factor):
    # Use fixed sigma_x and sigma_y (mother wavelet spread) passed in; do NOT recalc here

    # Rotate coordinates
    x_rot = x * np.cos(theta) + y * np.sin(theta)
    y_rot = -x * np.sin(theta) + y * np.cos(theta)

    # Gaussian envelope (no scaling of coordinates)
    exponent = -0.5 * ((x_rot**2) / (sigma_x**2) + (y_rot**2) / (sigma_y**2))

    # Gabor function with cosine modulation
    gabor = (1 / (2 * np.pi * sigma_x * sigma_y)) * np.exp(exponent) * np.cos(2 * np.pi * W_x * x_rot)

    # Apply energy normalization factor as an outer multiplier
    gabor *= a_power_m_factor

    return gabor


def generate_gabor_kernel(sigma_x, sigma_y, W_x, theta_deg, size, a_power_m_factor):
    theta = np.deg2rad(theta_deg)
    kernel = np.zeros((size, size), dtype=np.float32)
    half = size // 2
    for i in range(-half, half + 1):
        for j in range(-half, half + 1):
            kernel[i + half, j + half] = gabor_function_from_paper(i, j, sigma_x, sigma_y, W_x, theta, a_power_m_factor)
    return kernel


# === STEP 3: Convolution and Feature Extraction ===
def convolve_image(img, kernel):
    h, w = img.shape
    kh, kw = kernel.shape
    pad_h, pad_w = kh // 2, kw // 2
    padded = np.pad(img, ((pad_h, pad_h), (pad_w, pad_w)), mode='reflect')
    out = np.zeros_like(img, dtype=np.float32)

    for i in range(h):
        for j in range(w):
            region = padded[i:i+kh, j:j+kw]
            out[i, j] = np.sum(region * kernel)
    return out


def extract_features(img, kernel_bank):
    features = []
    for kernel in kernel_bank:
        filtered = convolve_image(img, kernel)
        features.append(np.mean(filtered))
        features.append(np.std(filtered))
    return np.array(features, dtype=np.float32)


# === STEP 4: Main Pipeline ===
def main():
    # Step 1: Preprocess images
    process_all_images(INPUT_DIR, OUTPUT_DIR)

    # Step 2: Generate Gabor kernel bank (24 kernels)
    kernel_bank = []
    for m, W_x in enumerate(W_x_list):  # m = 0 to 3
        a_power_m_factor = (2) ** (-m)  # a = 2, so a^-m
        for theta in orientations_deg:
            kernel = generate_gabor_kernel(sigma_x, sigma_y, W_x, theta, kernel_size, a_power_m_factor)
            kernel_bank.append(kernel)

    # Step 3: Extract features from each sub-image
    features = []
    file_list = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.jpg'))])
    for fname in file_list:
        path = os.path.join(OUTPUT_DIR, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"Skipping {fname}")
            continue
        vec = extract_features(img, kernel_bank)
        features.append(vec)

    # Step 4: Save features
    feature_matrix = np.array(features)
    np.save(FEATURE_SAVE_PATH, feature_matrix)
    print(f"Done! Extracted features for {len(features)} images. Saved at: {FEATURE_SAVE_PATH}")


# === Distance Metrics ===
def manhattan_distance(x, y):
    return np.sum(np.abs(x - y))

def weighted_mean_variance_distance(x, y, std):
    return np.sum(np.abs(x - y) / std)

def euclidean_distance(x, y):
    return np.sqrt(np.sum((x - y) ** 2))

def chebyshev_distance(x, y):
    return np.max(np.abs(x - y))

def mahalanobis_distance(x, y, inv_cov):
    diff = x - y
    val = np.dot(diff, np.dot(inv_cov, diff))
    val = max(val, 0)
    return np.sqrt(val)


def canberra_distance(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sum(np.abs(x - y) / (np.abs(x) + np.abs(y)))

def bray_curtis_distance(x, y):
    x = np.asarray(x)
    y = np.asarray(y)
    return np.sum(np.abs(x - y)) / np.sum(x + y)

def squared_chord_distance(x, y):
    return np.sum((np.sqrt(x) - np.sqrt(y)) ** 2)


def squared_chi_squared_distance(x, y):
    num = (x - y) ** 2
    denom = x + y
    return np.sum(num / denom)

# ===Evaluate retrieval accuracy for ALL queries ===

def evaluate_all_queries(top_N=16):
    start_time_total = time.time()

    database_features = np.load(FEATURE_SAVE_PATH)
    file_list = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.jpg'))])
    num_images = len(file_list)

    std = np.std(database_features, axis=0)
    std[std == 0] = 1e-5

    cov = np.cov(database_features.T)
    inv_cov = np.linalg.pinv(cov)

    total_relevant_retrieved = {metric: 0 for metric in [
        "Manhattan", "Weighted-Mean-Variance", "Euclidean", "Chebyshev", "Mahalanobis",
        "Canberra", "Bray-Curtis", "Squared-Chord", "Squared-Chi-Squared"
    ]}
    
    # Dictionary to store time taken for each metric
    metric_times = {metric: 0.0 for metric in total_relevant_retrieved.keys()}

    def get_parent_class(fname):
        return fname.split('_sub_')[0]

    for query_index in range(num_images):
        query_vec = database_features[query_index]
        query_name = file_list[query_index]
        query_class = get_parent_class(query_name)

        distances = {
            "Manhattan": [],
            "Weighted-Mean-Variance": [],
            "Euclidean": [],
            "Chebyshev": [],
            "Mahalanobis": [],
            "Canberra": [],
            "Bray-Curtis": [],
            "Squared-Chord": [],
            "Squared-Chi-Squared": []
        }

        for i, db_vec in enumerate(database_features):
            if i == query_index:
                continue
            db_name = file_list[i]
            
            # Time each distance calculation
            start = time.time()
            distances["Manhattan"].append((db_name, manhattan_distance(query_vec, db_vec)))
            metric_times["Manhattan"] += time.time() - start
            
            start = time.time()
            distances["Weighted-Mean-Variance"].append((db_name, weighted_mean_variance_distance(query_vec, db_vec, std)))
            metric_times["Weighted-Mean-Variance"] += time.time() - start
            
            start = time.time()
            distances["Euclidean"].append((db_name, euclidean_distance(query_vec, db_vec)))
            metric_times["Euclidean"] += time.time() - start
            
            start = time.time()
            distances["Chebyshev"].append((db_name, chebyshev_distance(query_vec, db_vec)))
            metric_times["Chebyshev"] += time.time() - start
            
            start = time.time()
            distances["Mahalanobis"].append((db_name, mahalanobis_distance(query_vec, db_vec, inv_cov)))
            metric_times["Mahalanobis"] += time.time() - start
            
            start = time.time()
            distances["Canberra"].append((db_name, canberra_distance(query_vec, db_vec)))
            metric_times["Canberra"] += time.time() - start
            
            start = time.time()
            distances["Bray-Curtis"].append((db_name, bray_curtis_distance(query_vec, db_vec)))
            metric_times["Bray-Curtis"] += time.time() - start
            
            start = time.time()
            distances["Squared-Chord"].append((db_name, squared_chord_distance(query_vec, db_vec)))
            metric_times["Squared-Chord"] += time.time() - start
            
            start = time.time()
            distances["Squared-Chi-Squared"].append((db_name, squared_chi_squared_distance(query_vec, db_vec)))
            metric_times["Squared-Chi-Squared"] += time.time() - start

        for metric in distances:
            top_results = sorted(distances[metric], key=lambda x: x[1])[:top_N]
            relevant = sum(1 for fname, _ in top_results if get_parent_class(fname) == query_class)
            total_relevant_retrieved[metric] += relevant

    max_relevant_per_query = min(top_N, 15)

    print(f"\nAverage Retrieval Accuracy (Precision@Top-{top_N}):")
    for metric, total_relevant in total_relevant_retrieved.items():
        acc = (total_relevant / (num_images * max_relevant_per_query)) * 100 if max_relevant_per_query > 0 else 0.0
        print(f"  {metric}: {acc:.2f}% (Time: {metric_times[metric]:.4f}s)")
    
    total_retrieval_time = time.time() - start_time_total
    print(f"\nTotal Retrieval Time for All Queries: {total_retrieval_time:.4f} seconds")


# === Display Retrieved Images for a Sample Query ===
    # Load saved features and file list
    database_features = np.load(FEATURE_SAVE_PATH)
    file_list = sorted([f for f in os.listdir(OUTPUT_DIR) if f.endswith(('.png', '.jpg'))])


    query_index = 31  # Change to any index to visualize
    top_N = 16

    query_vec = database_features[query_index]
    query_name = file_list[query_index]

    distances = []
    for i, db_vec in enumerate(database_features):
        db_name = file_list[i]
        if i != query_index:
            dist = canberra_distance(query_vec, db_vec)
            distances.append((db_name, dist))

    top_results = sorted(distances, key=lambda x: x[1])[:top_N]

    num_display_images = top_N + 1
    num_rows = 2
    num_cols = math.ceil(num_display_images / num_rows)

    fig_width = 3 * num_cols
    fig_height = 3 * num_rows

    fig, axes = plt.subplots(num_rows, num_cols, figsize=(fig_width, fig_height))
    fig.suptitle(f"Image Retrieval Results Using Canberra Metric", fontsize=14, y=1.02)
    axes_flat = axes.flatten()

    query_img = cv2.imread(os.path.join(OUTPUT_DIR, query_name), cv2.IMREAD_GRAYSCALE)
    axes_flat[0].imshow(query_img, cmap='gray')
    axes_flat[0].set_title("Query", fontsize=10)
    axes_flat[0].axis("off")

    for i, (fname, dist) in enumerate(top_results):
        if (i + 1) < len(axes_flat):
            img = cv2.imread(os.path.join(OUTPUT_DIR, fname), cv2.IMREAD_GRAYSCALE)
            axes_flat[i + 1].imshow(img, cmap='gray')
            display_fname = fname.split('_sub_')[0]
            axes_flat[i + 1].set_title(f"{display_fname}\nDist: {dist:.2f}", fontsize=9)
            axes_flat[i + 1].axis("off")

    for j in range(num_display_images, len(axes_flat)):
        axes_flat[j].axis("off")

    plt.subplots_adjust(wspace=0.1, hspace=0.3)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()



if __name__ == "__main__":
    # Uncomment below to regenerate features if needed
    main()

    evaluate_all_queries(top_N=16)
