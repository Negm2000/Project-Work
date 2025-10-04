import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ColorDetectorApp:
    """
    A GUI application for detecting specific colors in an image using HSV color space.
    The user can load an image, click to select a color, and then fine-tune the
    detection parameters with sliders for live feedback.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Component Color Detector")
        self.root.configure(bg="#2E2E2E")

        # --- Instance Variables ---
        self.original_image = None
        self.hsv_image = None
        self.hsv_color = None
        self.sensitivity = 15 # Initial hue sensitivity

        # --- Main Layout Frames ---
        # Top bar for loading
        top_frame = tk.Frame(root, bg="#3C3C3C", padx=10, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)

        # --- FIX: Use a PanedWindow for a stable, resizable layout ---
        main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg="#2E2E2E", sashwidth=8, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Image panels on the left pane
        image_frame = tk.Frame(main_pane, bg="#2E2E2E", padx=10, pady=10)
        main_pane.add(image_frame, stretch="always")

        # Controls/sliders on the right pane
        controls_frame = tk.Frame(main_pane, bg="#3C3C3C", padx=15, pady=15)
        main_pane.add(controls_frame, stretch="never")
        
        # --- Top Bar Widgets ---
        self.btn_load = tk.Button(
            top_frame, text="Load Image", command=self.load_image,
            bg="#555555", fg="white", relief="flat", padx=10, pady=5
        )
        self.btn_load.pack(side=tk.LEFT, padx=5)
        self.info_label = tk.Label(
            top_frame, text="Click on the image to pick a color.",
            bg="#3C3C3C", fg="white"
        )
        self.info_label.pack(side=tk.LEFT, padx=10)
        self.hsv_display_label = tk.Label(
            top_frame, text="Selected HSV: None",
            bg="#3C3C3C", fg="#A9A9A9"
        )
        self.hsv_display_label.pack(side=tk.RIGHT, padx=10)

        # --- Image Display Panels ---
        self.panel_original = tk.Label(image_frame, bg="#2E2E2E")
        self.panel_original.pack(side=tk.LEFT, padx=10, pady=5, expand=True)
        self.panel_original.bind("<Button-1>", self.pick_color)
        self.panel_mask = tk.Label(image_frame, bg="#2E2E2E")
        self.panel_mask.pack(side=tk.RIGHT, padx=10, pady=5, expand=True)
        tk.Label(image_frame, text="Original Image", bg="#2E2E2E", fg="white").place(in_=self.panel_original, relx=0.5, y=-20, anchor='n')
        tk.Label(image_frame, text="Processed Mask", bg="#2E2E2E", fg="white").place(in_=self.panel_mask, relx=0.5, y=-20, anchor='n')

        # --- Controls Panel (Sliders) ---
        self.create_control_sliders(controls_frame)

    def create_control_sliders(self, parent):
        """Creates all the sliders for HSV and filtering."""
        s = ttk.Style()
        s.configure("TScale", background="#3C3C3C", foreground="white")

        tk.Label(parent, text="HSV Fine-Tuning", font=("Helvetica", 12, "bold"), bg="#3C3C3C", fg="white").pack(pady=(0, 10))

        # Hue Sliders
        tk.Label(parent, text="Hue (H)", bg="#3C3C3C", fg="white").pack()
        self.h_lower_slider = tk.Scale(parent, from_=0, to=179, orient=tk.HORIZONTAL, command=self.update_mask, bg="#555", fg="white", troughcolor="#2E2E2E", length=200)
        self.h_lower_slider.pack()
        self.h_upper_slider = tk.Scale(parent, from_=0, to=179, orient=tk.HORIZONTAL, command=self.update_mask, bg="#555", fg="white", troughcolor="#2E2E2E", length=200)
        self.h_upper_slider.pack(pady=(0, 15))

        # Saturation Sliders
        tk.Label(parent, text="Saturation (S)", bg="#3C3C3C", fg="white").pack()
        self.s_lower_slider = tk.Scale(parent, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_mask, bg="#555", fg="white", troughcolor="#2E2E2E", length=200)
        self.s_lower_slider.pack()
        self.s_upper_slider = tk.Scale(parent, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_mask, bg="#555", fg="white", troughcolor="#2E2E2E", length=200)
        self.s_upper_slider.pack(pady=(0, 15))

        # Value Sliders
        tk.Label(parent, text="Value (V)", bg="#3C3C3C", fg="white").pack()
        self.v_lower_slider = tk.Scale(parent, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_mask, bg="#555", fg="white", troughcolor="#2E2E2E", length=200)
        self.v_lower_slider.pack()
        self.v_upper_slider = tk.Scale(parent, from_=0, to=255, orient=tk.HORIZONTAL, command=self.update_mask, bg="#555", fg="white", troughcolor="#2E2E2E", length=200)
        self.v_upper_slider.pack(pady=(0, 25))

        # Filter Control
        tk.Label(parent, text="Mask Filtering", font=("Helvetica", 12, "bold"), bg="#3C3C3C", fg="white").pack(pady=(10, 10))
        tk.Label(parent, text="Min Thickness (pixels)", bg="#3C3C3C", fg="white").pack()
        self.min_thickness_slider = tk.Scale(parent, from_=0, to=50, orient=tk.HORIZONTAL, command=self.update_mask, bg="#555", fg="white", troughcolor="#2E2E2E", length=200)
        self.min_thickness_slider.set(5) # Default value
        self.min_thickness_slider.pack()

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        try:
            self.original_image = cv2.imread(path)
            if self.original_image is None: raise ValueError("Could not read image.")
            self.hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            self.hsv_color = None
            self.hsv_display_label.config(text="Selected HSV: None")
            self.display_image(self.original_image, self.panel_original)
            self.panel_mask.config(image=''); self.panel_mask.image = ''
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def pick_color(self, event):
        if self.hsv_image is None: return
        
        # Scale coordinates from display to original image size
        img_w, img_h = self.panel_original.image.width(), self.panel_original.image.height()
        orig_h, orig_w, _ = self.original_image.shape
        orig_x = int(event.x * (orig_w / img_w))
        orig_y = int(event.y * (orig_h / img_h))
        orig_x = np.clip(orig_x, 0, orig_w - 1)
        orig_y = np.clip(orig_y, 0, orig_h - 1)

        self.hsv_color = self.hsv_image[orig_y, orig_x]
        
        # --- FIX: Convert numpy.uint8 to int to prevent overflow warnings/errors ---
        h, s, v = map(int, self.hsv_color)
        
        self.hsv_display_label.config(text=f"Selected HSV: [{h}, {s}, {v}]")
        
        # Initialize sliders around the picked color (now with safe integer math)
        self.h_lower_slider.set(max(0, h - self.sensitivity))
        self.h_upper_slider.set(min(179, h + self.sensitivity))
        self.s_lower_slider.set(max(0, s - 80))
        self.s_upper_slider.set(min(255, s + 80))
        self.v_lower_slider.set(max(0, v - 80))
        self.v_upper_slider.set(min(255, v + 80))

        self.update_mask()

    def update_mask(self, _=None): # The argument is for the slider command
        if self.hsv_image is None: return

        # Get current values from sliders
        lower_bound = np.array([self.h_lower_slider.get(), self.s_lower_slider.get(), self.v_lower_slider.get()])
        upper_bound = np.array([self.h_upper_slider.get(), self.s_upper_slider.get(), self.v_upper_slider.get()])
        
        mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)
        
        # Post-processing to reduce noise
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        # Filter by thickness
        min_thickness = self.min_thickness_slider.get()
        if min_thickness > 0:
            filtered_mask = np.zeros_like(mask)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                x, y, w, h = cv2.boundingRect(cnt)
                # Keep contours that are not too thin
                if w >= min_thickness and h >= min_thickness:
                    cv2.drawContours(filtered_mask, [cnt], -1, (255), -1)
            mask = filtered_mask

        self.display_image(mask, self.panel_mask)

    def display_image(self, img_data, panel):
        max_height = 500
        h, w = img_data.shape[:2]
        if h > max_height:
            ratio = max_height / h
            new_w, new_h = int(w * ratio), int(h * ratio)
            img_data = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        if len(img_data.shape) == 3:
            img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(img_data)
        img_tk = ImageTk.PhotoImage(image=img)

        panel.config(image=img_tk)
        panel.image = img_tk


if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ColorDetectorApp(root)
        root.geometry("1200x700") # Increased window size for controls
        root.mainloop()
    except ImportError:
        print("Error: Missing required libraries.")
        print("Please install them using: pip install opencv-python-headless pillow numpy")

