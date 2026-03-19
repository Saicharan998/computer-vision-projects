import numpy as np
import cv2 as cv
import os
import matplotlib.pyplot as plt
import glob
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import pandas as pd
import seaborn as sns
from collections import Counter

# ------------------- Load Training Images -------------------
target_folders = ['Rambutan', 'Orange','Pineapple','Banana Lady Finger','Fig','Mulberry']
training_root = "/Users/bharath_sj/Downloads/Assignment 2/Fruit-Images-Dataset-master/Training"

train_images = []
train_labels = []

for folder_name in target_folders:
    folder_path = os.path.join(training_root, folder_name)
    if os.path.exists(folder_path):
        for img_path in glob.glob(os.path.join(folder_path, "*.jpg")):
            img = cv.imread(img_path, 1)
            if img is not None:
                train_images.append(img)
                train_labels.append(folder_name)

# ------------------- Load Test Images -------------------
test_root = "/Users/bharath_sj/Downloads/Assignment 2/Fruit-Images-Dataset-master/Test"
test_images = []
test_labels = []

for folder_name in target_folders:
    folder_path = os.path.join(test_root, folder_name)
    if os.path.exists(folder_path):
        for img_path in glob.glob(os.path.join(folder_path, "*.jpg")):
            img = cv.imread(img_path, 1)
            if img is not None:
                test_images.append(img)
                test_labels.append(folder_name)

print(f"Loaded {len(train_images)} training and {len(test_images)} test images.")

# ------------------- Logic to Convert Images to Features -------------------
def img2gray(img):
    return cv.cvtColor(img, cv.COLOR_BGR2GRAY)

def compute_glcm_manual(img, d_x=1, d_y=0, levels=8):
    img_scaled = np.clip((img.astype(np.float32) / (256.0 / levels)).astype(int), 0, levels-1)
    glcm = np.zeros((levels, levels), dtype=np.float32)
    
    for i in range(img_scaled.shape[0]):
        for j in range(img_scaled.shape[1]):
            if 0 <= i + d_y < img_scaled.shape[0] and 0 <= j + d_x < img_scaled.shape[1]:
                glcm[img_scaled[i, j], img_scaled[i + d_y, j + d_x]] += 1

    if np.sum(glcm) > 0:
        glcm /= np.sum(glcm)
    return glcm
#-------------------- Feature Extraction -------------------
def extract_glcm_features(glcm):
    contrast = energy = homogeneity = entropy = dissimilarity = 0.0
    
    for i in range(glcm.shape[0]):
        for j in range(glcm.shape[1]):
            pij = glcm[i, j]
            contrast += (i-j)**2 * pij
            dissimilarity += abs(i-j) * pij
            energy += pij**2
            homogeneity += pij/(1+abs(i-j))
            if pij > 0:
                entropy -= pij * np.log2(pij)
    
    return [contrast, energy, homogeneity, entropy, dissimilarity]

def extract_enhanced_features(img):
    gray = img2gray(img)
    glcm = compute_glcm_manual(gray)
    texture_features = extract_glcm_features(glcm)
    
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    h_mean, h_std = np.mean(hsv[:,:,0]), np.std(hsv[:,:,0])
    s_mean, s_std = np.mean(hsv[:,:,1]), np.std(hsv[:,:,1])
    v_mean, v_std = np.mean(hsv[:,:,2]), np.std(hsv[:,:,2])
    
    return texture_features + [h_mean, h_std, s_mean, s_std, v_mean, v_std]
#---------------------------- Feature Normalization -------------------
def normalize_features(features):
    features = np.array(features, dtype=np.float32)
    mean = np.mean(features, axis=0)
    std = np.std(features, axis=0)
    std[std == 0] = 1e-8
    return (features - mean)/std, mean, std

def euclidean_distance(a, b):
    return np.sqrt(np.sum((a - b)**2))
# ------------------- DBSCAN Clustering -------------------
def dbscan(data, eps, min_samples):
    n = len(data)
    labels = [-1] * n
    visited = [False] * n
    cluster_id = 0

    def region_query(idx):
        return [i for i in range(n) if euclidean_distance(data[idx], data[i]) <= eps]

    def expand_cluster(idx, neighbors, cluster_id):
        labels[idx] = cluster_id
        i = 0
        while i < len(neighbors):
            point = neighbors[i]
            if not visited[point]:
                visited[point] = True
                new_neighbors = region_query(point)
                if len(new_neighbors) >= min_samples:
                    neighbors += [nn for nn in new_neighbors if nn not in neighbors]
            if labels[point] == -1:
                labels[point] = cluster_id
            i += 1

    for i in range(n):
        if not visited[i]:
            visited[i] = True
            neighbors = region_query(i)
            if len(neighbors) >= min_samples:
                expand_cluster(i, neighbors, cluster_id)
                cluster_id += 1
    return labels
#--------------------- Feature Extraction and Normalization -------------------
print("Extracting features...")
train_features = [extract_enhanced_features(img) for img in train_images]
normalized_train_combined, train_mean_combined, train_std_combined = normalize_features(train_features)

test_features = [extract_enhanced_features(img) for img in test_images]
normalized_test_combined = (np.array(test_features) - train_mean_combined) / train_std_combined

train_labels_db = dbscan(normalized_train_combined, eps=0.5, min_samples=3)
# ------------------- Cluster Processing -------------------
cluster_points = {}
for i, lbl in enumerate(train_labels_db):
    if lbl != -1:
        if lbl not in cluster_points:
            cluster_points[lbl] = []
        cluster_points[lbl].append(normalized_train_combined[i])

centroids = {lbl: np.mean(cluster_points[lbl], axis=0) for lbl in cluster_points}

def assign_cluster(test_point, centroids_dict):
    min_dist = float('inf')
    assigned = -1
    for cluster_lbl, centroid_vec in centroids_dict.items():
        dist = euclidean_distance(test_point, centroid_vec)
        if dist < min_dist:
            min_dist = dist
            assigned = cluster_lbl
    return assigned

test_cluster_assignments = [assign_cluster(pt, centroids) for pt in normalized_test_combined]
# ------------------- Evaluation -------------------
unique_labels = sorted(list(set(train_labels + test_labels)))
label_map = {label: idx for idx, label in enumerate(unique_labels)}
reverse_label_map = {idx: label for label, idx in label_map.items()}

cluster_votes = {}
for cluster, true_class_str in zip(test_cluster_assignments, test_labels):
    if cluster != -1:
        true_class_numeric = label_map.get(true_class_str)
        if cluster not in cluster_votes:
            cluster_votes[cluster] = []
        cluster_votes[cluster].append(true_class_numeric)

cluster_to_class = {c: Counter(votes).most_common(1)[0][0] for c, votes in cluster_votes.items() if votes}

predicted_class_labels = [cluster_to_class.get(cluster, -1) for cluster in test_cluster_assignments]
valid_idx = [i for i, pred in enumerate(predicted_class_labels) if pred != -1]
y_true = [label_map[test_labels[i]] for i in valid_idx]
y_pred = [predicted_class_labels[i] for i in valid_idx]

if y_true and y_pred:
    cm = confusion_matrix(y_true, y_pred, labels=sorted(label_map.values()))
    df_cm = pd.DataFrame(cm,
                        index=[reverse_label_map[i] for i in sorted(label_map.values())],
                        columns=[reverse_label_map[i] for i in sorted(label_map.values())])
    print("\nConfusion Matrix:")
    print(df_cm)
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=[reverse_label_map[i] for i in sorted(label_map.values())], zero_division=0))
    print(f"\nAccuracy: {accuracy_score(y_true, y_pred)*100:.2f}%")
    
    plt.figure(figsize=(8,6))
    sns.heatmap(df_cm, annot=True, fmt='d', cmap='Blues')
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

num_noise = sum(1 for x in test_cluster_assignments if x == -1)
print(f"Test predictions: {len(test_cluster_assignments)-num_noise} assigned, {num_noise} as noise.")
# ------------------- Multi-Fruit Segmentation -------------------
FRUIT_COLORS = {
    'Rambutan': (0, 0, 255),              # Red
    'Orange': (0, 165, 255),              # Orange
    'Pineapple': (0, 255, 255),           # Yellow
    'Banana Lady Finger': (0, 255, 128),  # Light Green
    'Fig': (255, 0, 255),                 # Magenta
    'Mulberry': (0, 0, 128)               # Dark Red
}

def process_multi_fruit_image(image_path, trained_mean, trained_std, centroids, 
                            cluster_to_class_map, label_map, reverse_label_map):
    original_img = cv.imread(image_path)
    if original_img is None:
        print(f"Error loading image: {image_path}")
        return None, None, None
    if max(original_img.shape) > 1500:
        original_img = cv.resize(original_img, (0,0), fx=0.5, fy=0.5)
    
    gray = cv.cvtColor(original_img, cv.COLOR_BGR2GRAY)
    
    mask = cv.adaptiveThreshold(gray, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, 
                              cv.THRESH_BINARY_INV, 11, 2)
    
    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (11,11))  
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel, iterations=4)  
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel, iterations=2)   
    
    # Find contours
    contours, _ = cv.findContours(mask, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    
    
    colored_output = original_img.copy()
    final_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
    
    for contour in contours:
        area = cv.contourArea(contour)
        if area < 1000 or area > 100000:  
            continue
            
    
        hull = cv.convexHull(contour)
        x,y,w,h = cv.boundingRect(hull)

        padding = 15
        fruit_region = original_img[max(0,y-padding):min(original_img.shape[0],y+h+padding),
                                  max(0,x-padding):min(original_img.shape[1],x+w+padding)]
        
        if fruit_region.size == 0:
            continue
            
        try:
            features = extract_enhanced_features(fruit_region)
            normalized = (np.array(features) - trained_mean) / trained_std
            cluster = assign_cluster(normalized, centroids)
            predicted_class = reverse_label_map.get(cluster_to_class_map.get(cluster, -1), "Unknown")
            
            if predicted_class in FRUIT_COLORS:
                fruit_mask = np.zeros(original_img.shape[:2], dtype=np.uint8)
                cv.drawContours(fruit_mask, [hull], -1, 255, -1)
                final_mask = cv.bitwise_or(final_mask, fruit_mask)
                colored_output[fruit_mask == 255] = FRUIT_COLORS[predicted_class]

                M = cv.moments(hull)
                if M["m00"] > 0:
                    cX, cY = int(M["m10"]/M["m00"]), int(M["m01"]/M["m00"])
                    text_size = cv.getTextSize(predicted_class, cv.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
                    cv.rectangle(colored_output, 
                               (cX - text_size[0]//2 - 5, cY - text_size[1] - 5),
                               (cX + text_size[0]//2 + 5, cY + 5),
                               FRUIT_COLORS[predicted_class], -1)
                    cv.putText(colored_output, predicted_class, 
                             (cX - text_size[0]//2, cY), 
                             cv.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 1)
        
        except Exception as e:
            print(f"Error processing contour: {str(e)}")
            continue

    # ---------- Morphological Operations ----------
    final_mask = cv.morphologyEx(final_mask, cv.MORPH_CLOSE, kernel, iterations=2)
    
    
    plt.figure(figsize=(10,5))
    plt.subplot(1,2,1)
    plt.imshow(cv.cvtColor(original_img, cv.COLOR_BGR2RGB))
    plt.title("Original")
    plt.axis('off')
    
    plt.subplot(1,2,2)
    plt.imshow(cv.cvtColor(colored_output, cv.COLOR_BGR2RGB))
    plt.title("Identified Fruits")
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return original_img, final_mask, colored_output


image_path = "/Users/bharath_sj/Downloads/Assignment 2/Test_image_5fruits_group9.jpg"
if centroids and cluster_to_class and label_map and reverse_label_map:
    process_multi_fruit_image(
    image_path, 
    train_mean_combined, 
    train_std_combined, 
    centroids, 
    cluster_to_class, 
    label_map, 
    reverse_label_map
)