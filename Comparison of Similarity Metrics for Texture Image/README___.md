# Texture-Based Image Retrieval using Gabor Features and Distance Metrics

This project implements a complete pipeline for **texture-based image retrieval** using Gabor filter feature extraction and various distance metrics. The goal is to evaluate how well different metrics perform for retrieving similar texture patches from the Brodatz dataset.

---

##  Directory Structure

```
.
├── Real_Brodatz/              # Original grayscale Brodatz texture images
├── sub img/                   # 128x128 subdivided patches (auto-generated)
├── gabor_features.npy         # Saved feature vectors for all sub-images
├── main_script.py             # Main pipeline script (the code you provided)
└── README.md
```

---

##  Key Features

- Resize each image to **512x512** and split into **16 non-overlapping 128x128 patches**
- Apply **Gabor filters** with 24 combinations of scale and orientation
- Extract **mean and standard deviation** of each filtered result
- Store 48-dimensional feature vector per sub-image
- Compare retrieval accuracy using:
  - Manhattan
  - Euclidean
  - Mahalanobis
  - Chebyshev
  - Weighted Mean-Variance
  - Bray-Curtis
  - Canberra
  - Squared Chord
  - Squared Chi-Squared distances
- Visualize top-16 retrieved results for a sample query image

---

##  Requirements

Install the following Python packages:

```bash
pip install numpy opencv-python matplotlib
```

---

##  How It Works

### 1. **Image Preprocessing**

- Images in `Real_Brodatz/` are resized to `512x512`.
- Each image is split into `16` patches of size `128x128`.
- Resulting patches are saved to `sub img/`.

### 2. **Gabor Kernel Generation**

- Gabor filters are generated based on the paper:
  - `4 frequency bands (W_x)`
  - `6 orientations (0° to 150°)`
  - `a^-m` normalization for scale-invariant filtering
- `24` Gabor kernels total.

### 3. **Feature Extraction**

- Each patch is convolved with the 24 kernels.
- For each filtered result, mean and standard deviation are computed.
- Results in a `48-dimensional feature vector`.

### 4. **Distance-Based Retrieval**

- For each patch in the dataset, compute distances to all others.
- Rank results using different metrics.
- Evaluate **Precision@Top-16**: how many of the top 16 results belong to the same class.

### 5. **Visualization**

- Displays a query image and its top-16 retrieved matches using Bray-Curtis distance.

---

##  Running the Project

```python
# Inside your script
if __name__ == "__main__":
    main()  # Step 1-4: Preprocessing and Feature Extraction
    evaluate_all_queries(top_N=16)  # Step 5: Retrieval and Evaluation
```

---

##  Output

- Console shows retrieval accuracy (precision) for each metric.
- Also prints total time taken for evaluation.
- A visual display shows the top-16 retrieved patches for a chosen query.

---

##  Notes

- Modify `INPUT_DIR`, `OUTPUT_DIR`, and `FEATURE_SAVE_PATH` in the config section to match your system.
- Make sure `Real_Brodatz` contains valid grayscale images.
- Change `query_index` to visualize different queries in the plot.

---

##  To-Do / Improvements

- Add class labels for evaluation automation
- Speed up convolution with `cv2.filter2D` or `scipy.ndimage.convolve`
- GUI for dynamic query selection
- Integrate more texture descriptors (e.g., LBP, GLCM)

---

##  Contact

For questions or contributions, feel free to reach out.
