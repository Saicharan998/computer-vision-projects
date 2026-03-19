import numpy as np
import cv2 as cv
import time

# -----------------------------------------------------------------------------
# @file        rgb_to_hsv_conversion.py
# @brief       Manual and NumPy based RGB to HSV image conversion and comparison
# @author      1.Bharath Subramaniam Jayakumar
# @author      2.Sai Charan Kandepi
# -----------------------------------------------------------------------------

def rgb_to_hsv(img_array):
    """
    @brief Manually convert an RGB image to HSV.
    @param img_array Numpy array of the image in BGR format.
    """
    img_array = img_array.astype(np.float32) / 255.0
    img_array = img_array[:, :, ::-1]  # Convert BGR to RGB
    hsv_image = np.zeros_like(img_array)
    start_time = time.time()
    for i in range(img_array.shape[0]):
        for j in range(img_array.shape[1]):
            r, g, b = img_array[i, j]
            cmax = max(r, g, b)
            cmin = min(r, g, b)
            delta = cmax - cmin

            if delta == 0:
                h = 0
            elif cmax == r:
                h = (60 * ((g - b) / delta))
            elif cmax == g:
                h = (60 * ((b - r) / delta) + 120)
            else:
                h = (60 * ((r - g) / delta) + 240)

            if h < 0:
                h += 360

            s = 0 if cmax == 0 else (delta / cmax)
            v = cmax

            h = int(h / 2)        # Scale for OpenCV
            s = int(s * 255)
            v = int(v * 255)

            hsv_image[i, j] = [h, s, v]
    elapsed_time = time.time() - start_time
    return hsv_image.astype(np.uint8), elapsed_time

def rgb_to_hsv_numpy(img_array):
    """
    @brief Convert an RGB image to HSV using NumPy operations.
    @param img_array Numpy array of the image in BGR format.
    """
    img_array = img_array.astype(np.float32) / 255.0
    img_array = img_array[:, :, ::-1]
    start_time = time.time()

    cmax = np.max(img_array, axis=2)
    cmin = np.min(img_array, axis=2)
    delta = cmax - cmin

    hsv_image = np.zeros_like(img_array)
    hue = np.zeros_like(cmax)
    mask = delta != 0
    r, g, b = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2]
    safe_delta = np.where(mask, delta, 1)

    hue[mask & (cmax == r)] = (60 * ((g - b) / safe_delta))[mask & (cmax == r)]
    hue[mask & (cmax == g)] = (60 * ((b - r) / safe_delta) + 120)[mask & (cmax == g)]
    hue[mask & (cmax == b)] = (60 * ((r - g) / safe_delta) + 240)[mask & (cmax == b)]

    hue[hue < 0] += 360
    hue = (hue / 2).astype(np.uint8)

    safe_cmax = np.where(cmax == 0, 1, cmax)  # Avoid division by zero
    saturation = (delta / safe_cmax) * 255
    saturation = saturation.astype(np.uint8)

    value = (cmax * 255).astype(np.uint8)

    hsv_image[:, :, 0] = hue
    hsv_image[:, :, 1] = saturation
    hsv_image[:, :, 2] = value

    elapsed_time = time.time() - start_time
    return hsv_image.astype(np.uint8), elapsed_time

def adjust_saturation(hsv_image, saturation_factor):
    """
    @brief Adjust the saturation of an HSV image.
    @param hsv_image Input HSV image.
    @param saturation_factor Scaling factor for saturation.
    @return Saturation-adjusted HSV image.
    """
    hsv_image = hsv_image.astype(np.float32)
    hsv_image[:, :, 1] *= saturation_factor
    hsv_image[:, :, 1] = np.clip(hsv_image[:, :, 1], 0, 255)
    return hsv_image.astype(np.uint8)

def display_image(window_name, image, width=800, height=600):
    """
    @brief Display an image in a resizable window.
    @param window_name Title of the window.
    @param image Image to be displayed.
    @param width Width of the window.
    @param height Height of the window.
    """
    cv.namedWindow(window_name, cv.WINDOW_NORMAL)
    cv.resizeWindow(window_name, width, height)
    cv.imshow(window_name, image)

# -----------------------------------------------------------------------------
# Main Execution
# -----------------------------------------------------------------------------

img1 = cv.imread('many_cereals.jpg')
img2 = cv.imread('contrast_brightness_image.jpg')

if img1 is None or img2 is None:
    print("Error: Could not load one or both images. Please check file paths.")
    exit(1)

# Process First Image
print("\nProcessing First Image (output_image1.jpg)")
img_array = np.copy(img1)

hsv_image_numpy_1, elapsed_time_numpy_1 = rgb_to_hsv_numpy(img_array)
hsv_image_manual_1, elapsed_time_manual_1 = rgb_to_hsv(img_array)

print(f"NumPy method elapsed time: {elapsed_time_numpy_1:.3f}s")
print(f"Manual method elapsed time: {elapsed_time_manual_1:.3f}s")
print(f"Time difference: {elapsed_time_manual_1 - elapsed_time_numpy_1:.3f}s")

# Save results
cv.imwrite('hsv_image_numpy_img1.png', hsv_image_numpy_1)
cv.imwrite('hsv_image_manual_img1.png', hsv_image_manual_1)

# Process Second Image
print("\nProcessing Second Image (output_image2.jpg)")
img_array = np.copy(img2)

hsv_image_numpy_2, elapsed_time_numpy_2 = rgb_to_hsv_numpy(img_array)
hsv_image_manual_2, elapsed_time_manual_2 = rgb_to_hsv(img_array)

print(f"NumPy method elapsed time: {elapsed_time_numpy_2:.3f}s")
print(f"Manual method elapsed time: {elapsed_time_manual_2:.3f}s")
print(f"Time difference: {elapsed_time_manual_2 - elapsed_time_numpy_2:.3f}s")

# Save results
cv.imwrite('hsv_image_numpy_img2.png', hsv_image_numpy_2)
cv.imwrite('hsv_image_manual_img2.png', hsv_image_manual_2)

# Adjust Saturation
print("\nDo you want to adjust the saturation? (y/n)")
adjust_saturation_input = input()
if adjust_saturation_input.lower() == 'y':
    print("Enter saturation factor:")
    saturation_factor = float(input())

    hsv_image_numpy_1 = adjust_saturation(hsv_image_numpy_1, saturation_factor)
    hsv_image_manual_1 = adjust_saturation(hsv_image_manual_1, saturation_factor)
    hsv_image_numpy_2 = adjust_saturation(hsv_image_numpy_2, saturation_factor)
    hsv_image_manual_2 = adjust_saturation(hsv_image_manual_2, saturation_factor)


#convert back to BGR for display
hsv_rgb_image_1 = cv.cvtColor(hsv_image_numpy_1, cv.COLOR_HSV2BGR)
hsv_rgb_image_2 = cv.cvtColor(hsv_image_numpy_2, cv.COLOR_HSV2BGR)

# Display all Images
display_image('Original Image 1', img1)
display_image('HSV Manual Image 1', hsv_image_manual_1)
display_image('HSV NumPy Image 1', hsv_image_numpy_1)
display_image('HSV-RGB Image 1', hsv_rgb_image_1 )

display_image('Original Image 2', img2)
display_image('HSV Manual Image 2', hsv_image_manual_2)
display_image('HSV NumPy Image 2', hsv_image_numpy_2)
display_image('HSV-RGB Image 2', hsv_rgb_image_2)

cv.waitKey(0)
cv.destroyAllWindows()
