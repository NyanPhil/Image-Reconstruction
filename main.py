import os
import cv2
import numpy as np
import matplotlib.pyplot as plt

# Get the directory of the current script
current_dir = os.path.dirname(os.path.abspath(__file__))
image_path = os.path.join(current_dir, "wanted_person.png")

# Load image and check if it loaded successfully
orig = cv2.imread(image_path) 
if orig is None:
    raise FileNotFoundError(f"Could not load image from {image_path}. Please check if the file exists and is readable.")
 
B, G, R = orig[:, :, 0].astype(np.float32), orig[:, :, 1].astype(np.float32), orig[:, :, 2].astype(np.float32)
gray = 0.1140 * B + 0.5870 * G + 0.2989 * R  # Grayscale conversion (ITU-R BT.601)

# Convert to uint8 for histogram equalization
gray_uint8 = np.uint8(np.clip(gray, 0, 255))

# Apply adaptive histogram equalization first
claheObj = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
gray_hist = claheObj.apply(gray_uint8)

# Normalize to [0, 1] using min-max normalization
gray_norm = gray_hist.astype(np.float32)
gray_norm = (gray_norm - gray_norm.min()) / (gray_norm.max() - gray_norm.min() + 1e-8)

# Apply bilateral filter to reduce noise while preserving edges
gray_blur = cv2.bilateralFilter(gray_norm.astype(np.float32), 5, 50, 50)

# Adaptive unsharp masking with smaller kernel
alpha = 1.5
blurred_for_sharp = cv2.GaussianBlur(gray_blur, (3, 3), 0)
sharp_pre_upsample = gray_blur + alpha * (gray_blur - blurred_for_sharp)
sharp_pre_upsample = np.clip(sharp_pre_upsample, 0, 1)

# For visualization only
def norm_for_display(img):
    return (img - img.min()) / (img.max() - img.min() + 1e-8)

# Use Lanczos interpolation for upsampling
upsampled = cv2.resize(sharp_pre_upsample, None, fx=2, fy=2, interpolation=cv2.INTER_LANCZOS4)
upsampled = np.clip(upsampled, 0, 1)

# After upsampling and before further processing
upsampled_uint8 = np.uint8(255 * upsampled)

# Apply CLAHE with adjusted parameters
claheObj = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8,8))
claheImg = claheObj.apply(upsampled_uint8)

# Convert to BGR for detail enhancement
claheImg_bgr = cv2.cvtColor(claheImg, cv2.COLOR_GRAY2BGR)

# Apply detail enhancement
detail_enhanced = cv2.detailEnhance(claheImg_bgr, sigma_s=10, sigma_r=0.15)

# Convert back to grayscale
detail_enhanced_gray = cv2.cvtColor(detail_enhanced, cv2.COLOR_BGR2GRAY)

# Apply bilateral filter to final result to smooth while preserving edges
final_img = cv2.bilateralFilter(detail_enhanced_gray, 5, 50, 50)

# Show images and histograms for each major stage
plt.figure(figsize=(12, 8))

plt.subplot(2, 2, 1)
plt.title("Original Image (BGR)")
plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
plt.axis('off')

plt.subplot(2, 2, 2)
plt.title("Initial CLAHE")
plt.imshow(gray_hist, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 3)
plt.title("Upsampled (Lanczos)")
plt.imshow(upsampled_uint8, cmap='gray')
plt.axis('off')

plt.subplot(2, 2, 4)
plt.title("Final Enhanced Image")
plt.imshow(final_img, cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()

# Show zoomed comparison
plt.figure(figsize=(12, 6))

# Calculate center crop coordinates
h, w = orig.shape[:2]
crop_size = min(h, w) // 4
center_y, center_x = h // 2, w // 2
y1, y2 = center_y - crop_size, center_y + crop_size
x1, x2 = center_x - crop_size, center_x + crop_size

# Original zoomed (convert to grayscale first)
orig_gray = cv2.cvtColor(orig, cv2.COLOR_BGR2GRAY)
plt.subplot(1, 2, 1)
plt.title("Original (Zoomed)")
plt.imshow(orig_gray[y1:y2, x1:x2], cmap='gray')
plt.axis('off')

# Final zoomed
plt.subplot(1, 2, 2)
plt.title("Enhanced (Zoomed)")
plt.imshow(final_img[y1*2:y2*2, x1*2:x2*2], cmap='gray')
plt.axis('off')

plt.tight_layout()
plt.show()