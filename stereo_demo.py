import cv2
import numpy as np
import matplotlib.pyplot as plt

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
    stereo = cv2.StereoSGBM_create(
        minDisparity=0, numDisparities=128, blockSize=5,
        P1=8 * 3 * 5**2, P2=32 * 3 * 5**2, disp12MaxDiff=1,
        uniquenessRatio=10, speckleWindowSize=100, speckleRange=32
    )

    print("Computing disparity map...")
    disparity = stereo.compute(img_left, img_right).astype(np.float32) / 16.0
    
    # --- Convert Disparity to Depth ---
    print("Assuming calibration parameters to calculate real-world depth...")

    # Assumed camera parameters for the demo
    FOCAL_LENGTH = 2800  # pixels
    BASELINE = 80        # mm (we will convert final depth to cm)
    
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
    
    # Convert depth from mm to cm
    depth_map_cm = depth_map_mm / 10
    depth_map_cm[disparity <= 0] = np.nan
    
    print("Depth map calculated. Displaying results... Click on the depth map to measure distance.")

    # --- Interactive Visualization ---
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 6))

    ax1.set_title('Left Image')
    ax1.imshow(img_left, cmap='gray')

    ax2.set_title('Right Image')
    ax2.imshow(img_right, cmap='gray')

    ax3.set_title('Interactive Depth Map (cm)')
    img_plot = ax3.imshow(depth_map_cm, cmap='viridis')
    fig.colorbar(img_plot, ax=ax3, label='Distance (cm)')

    # --- Click Event Handler ---
    def onclick(event):
        # Check if the click was on the depth map subplot
        if event.inaxes == ax3:
            ix, iy = int(event.xdata), int(event.ydata)
            depth = depth_map_cm[iy, ix]
            
            if not np.isnan(depth):
                print(f"Distance at pixel ({ix}, {iy}): {depth:.2f} cm")
            else:
                print(f"Distance at pixel ({ix}, {iy}): Not available")

    # Connect the click event to the handler
    fig.canvas.mpl_connect('button_press_event', onclick)
    
    plt.suptitle('Stereo Vision Depth Calculation Demo')
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to prevent title overlap
    plt.show()