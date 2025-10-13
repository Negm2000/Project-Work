import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm

print("Loading stereo image pair...")
# Load the left and right images in grayscale
img_left = cv2.imread('Datasets\\Stereo\\left.png', cv2.IMREAD_GRAYSCALE)
img_right = cv2.imread('Datasets\\Stereo\\right.png', cv2.IMREAD_GRAYSCALE)
# img_left = cv2.imread('Datasets\\Oven\\1.jpg', cv2.IMREAD_GRAYSCALE)
# img_right = cv2.imread('Datasets\\Oven\\2.jpg', cv2.IMREAD_GRAYSCALE)

# Check if images were loaded correctly
if img_left is None or img_right is None:
    print("Error: Could not load images. Check the file paths.")
else:
    print("Images loaded successfully.")

    # --- Stereo Matching Algorithm ---
    min_disp = 0
    num_disp = 80 # Smaller numDisparities for a smaller invalid border on the left
    stereo = cv2.StereoSGBM_create(
        minDisparity=min_disp,
        numDisparities=num_disp,
        blockSize=5,
        P1=8 * 3 * 5**2,
        P2=32 * 3 * 5**2,
        disp12MaxDiff=1,
        uniquenessRatio=10,
        speckleWindowSize=100,
        speckleRange=32
    )

    print("Computing disparity map...")
    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    
    # --- Convert Disparity to Depth ---
    print("Assuming calibration parameters to calculate real-world depth...")

    FOCAL_LENGTH = 2800
    BASELINE = 80
    
    height, width = img_left.shape
    cx = width / 2
    cy = height / 2

    Q = np.float32([
        [1, 0, 0, -cx],
        [0, 1, 0, -cy],
        [0, 0, 0, FOCAL_LENGTH],
        [0, 0, -1/BASELINE, 0]
    ])

    points_3D = cv2.reprojectImageTo3D(disparity, Q)
    depth_map_mm = points_3D[:, :, 2]
    
    depth_map_cm = depth_map_mm / 10
    
    depth_map_cm[disparity <= min_disp] = np.nan
    
    print("Depth map calculated. Displaying results... Click on either bottom map to measure distance.")

    # --- Create Depth Overlay Image ---
    depth_for_display = depth_map_cm.copy()
    min_val = np.nanmin(depth_for_display)
    max_val = np.nanmax(depth_for_display)
    normalized_depth = (depth_for_display - min_val) / (max_val - min_val)
    
    colored_depth_rgba = (cm.viridis(normalized_depth) * 255).astype(np.uint8)
    colored_depth = cv2.cvtColor(colored_depth_rgba, cv2.COLOR_RGBA2BGR)
    
    img_left_color = cv2.cvtColor(img_left, cv2.COLOR_GRAY2BGR)

    valid_depth_mask = ~np.isnan(depth_for_display)
    
    overlay = img_left_color.copy()
    alpha = 0.4
    overlay[valid_depth_mask] = cv2.addWeighted(
        img_left_color[valid_depth_mask], alpha,
        colored_depth[valid_depth_mask], 1 - alpha, 0
    )

    # --- Interactive Visualization ---
    fig, ax = plt.subplots(2, 2, figsize=(16, 10))

    ax[0, 0].set_title('Left Image')
    ax[0, 0].imshow(img_left, cmap='gray')

    ax[0, 1].set_title('Right Image')
    ax[0, 1].imshow(img_right, cmap='gray')

    ax[1, 0].set_title('Depth Map (cm)')
    ax[1, 0].imshow(depth_map_cm, cmap='viridis')

    ax[1, 1].set_title('Interactive Depth Overlay')
    ax[1, 1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))

    # --- Click Event Handler ---
    def onclick(event):
        # *** FIX: Check if the click was on EITHER of the bottom two plots ***
        if event.inaxes == ax[1, 0] or event.inaxes == ax[1, 1]:
            ix, iy = int(event.xdata), int(event.ydata)
            
            # Ensure coordinates are within image bounds
            if 0 <= iy < depth_map_cm.shape[0] and 0 <= ix < depth_map_cm.shape[1]:
                depth = depth_map_cm[iy, ix]
                if not np.isnan(depth):
                    print(f"Distance at pixel ({ix}, {iy}): {depth:.2f} cm")
                else:
                    print(f"Distance at pixel ({ix}, {iy}): Not available")

    fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.suptitle('Stereo Vision Depth Analysis Demo')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()