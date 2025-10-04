import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class ColorDetectorApp:
    """
    A GUI application for detecting electrical components. Users can click a color
    and then use sliders to fine-tune the HSV range, component area, and shape
    to isolate specific parts. The control panel is scrollable and slider ranges
    adapt to the loaded image size.
    """
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Component Detector")
        self.root.configure(bg="#2E2E2E")

        # --- Instance Variables ---
        self.original_image = None
        self.hsv_image = None
        self.sensitivity = 15 # Initial hue sensitivity for color picking

        # --- Main Layout Frames ---
        top_frame = tk.Frame(root, bg="#3C3C3C", padx=10, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg="#2E2E2E", sashwidth=8, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        image_frame = tk.Frame(main_pane, bg="#2E2E2E", padx=10, pady=10)
        main_pane.add(image_frame, stretch="always")
        controls_frame = tk.Frame(main_pane, bg="#3C3C3C", padx=15, pady=15)
        main_pane.add(controls_frame, stretch="never")
        
        # --- Top Bar Widgets ---
        self.btn_load = tk.Button(top_frame, text="Load Image", command=self.load_image, bg="#555555", fg="white", relief="flat", padx=10, pady=5)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        self.info_label = tk.Label(top_frame, text="Load an image and click on a color to begin.", bg="#3C3C3C", fg="white")
        self.info_label.pack(side=tk.LEFT, padx=10)
        
        # --- Image Display Panels ---
        self.panel_original = tk.Label(image_frame, bg="#2E2E2E")
        self.panel_original.pack(side=tk.LEFT, padx=10, pady=5, expand=True)
        self.panel_original.bind("<Button-1>", self.pick_color)
        self.panel_mask = tk.Label(image_frame, bg="#2E2E2E")
        self.panel_mask.pack(side=tk.RIGHT, padx=10, pady=5, expand=True)
        tk.Label(image_frame, text="Original Image", bg="#2E2E2E", fg="white").place(in_=self.panel_original, relx=0.5, y=-20, anchor='n')
        tk.Label(image_frame, text="Filtered Mask", bg="#2E2E2E", fg="white").place(in_=self.panel_mask, relx=0.5, y=-20, anchor='n')

        # --- Controls Panel ---
        self.create_filter_controls(controls_frame)

    def create_filter_controls(self, parent):
        """Creates all the sliders for filtering within a scrollable frame."""
        canvas = tk.Canvas(parent, bg="#3C3C3C", highlightthickness=0)
        scrollbar = ttk.Scrollbar(parent, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg="#3C3C3C")

        scrollable_frame.bind("<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        
        parent = scrollable_frame
        
        tk.Label(parent, text="Color Tuning (HSV)", font=("Helvetica", 12, "bold"), bg="#3C3C3C", fg="white").pack(pady=(0, 10), anchor='w')
        self.h_lower_slider = self.create_slider(parent, "Hue (H) Lower", 0, 179, self._h_lower_callback)
        self.h_upper_slider = self.create_slider(parent, "Hue (H) Upper", 0, 179, self._h_upper_callback)
        self.s_lower_slider = self.create_slider(parent, "Saturation (S) Lower", 0, 255, self._s_lower_callback)
        self.s_upper_slider = self.create_slider(parent, "Saturation (S) Upper", 0, 255, self._s_upper_callback)
        self.v_lower_slider = self.create_slider(parent, "Value (V) Lower", 0, 255, self._v_lower_callback)
        self.v_upper_slider = self.create_slider(parent, "Value (V) Upper", 0, 255, self._v_upper_callback)
        
        tk.Label(parent, text="Shape & Size Filters", font=("Helvetica", 12, "bold"), bg="#3C3C3C", fg="white").pack(pady=(20, 10), anchor='w')
        self.min_area_slider = self.create_slider(parent, "Min Area", 0, 5000, self._min_area_callback)
        self.min_area_slider.set(100)
        self.max_area_slider = self.create_slider(parent, "Max Area", 0, 50000, self._max_area_callback)
        self.max_area_slider.set(20000)
        self.min_aspect_slider = self.create_slider(parent, "Min Aspect Ratio (W/H)", 0, 100, self._min_aspect_callback)
        self.min_aspect_slider.set(1)
        self.max_aspect_slider = self.create_slider(parent, "Max Aspect Ratio (W/H)", 0, 100, self._max_aspect_callback)
        self.max_aspect_slider.set(50)

    def create_slider(self, parent, label_text, from_, to, command_func=None):
        if command_func is None: command_func = self.apply_filters
        tk.Label(parent, text=label_text, bg="#3C3C3C", fg="white").pack(pady=(10,0), anchor='w')
        slider = tk.Scale(parent, from_=from_, to=to, orient=tk.HORIZONTAL, command=command_func, bg="#555", fg="white", troughcolor="#2E2E2E", length=220)
        slider.pack(anchor='w')
        return slider
        
    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        try:
            self.original_image = cv2.imread(path)
            if self.original_image is None: raise ValueError("Could not read image.")
            self.hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)

            h, w, _ = self.original_image.shape
            total_area = h * w
            max_area_limit = total_area // 4
            self.max_area_slider.config(to=max_area_limit)
            self.max_area_slider.set(min(20000, max_area_limit))
            self.min_area_slider.config(to=max_area_limit // 10)
            self.min_area_slider.set(100)

            self.display_image(self.original_image, self.panel_original)
            blank_mask = np.zeros(self.original_image.shape[:2], dtype="uint8")
            self.display_image(blank_mask, self.panel_mask)
        except Exception as e:
            messagebox.showerror("Error", f"Failed to load image: {e}")

    def pick_color(self, event):
        if self.hsv_image is None: return

        img_w, img_h = self.panel_original.image.width(), self.panel_original.image.height()
        orig_h, orig_w, _ = self.original_image.shape
        orig_x = int(event.x * (orig_w/img_w)); orig_y = int(event.y * (orig_h/img_h))
        orig_x, orig_y = np.clip(orig_x, 0, orig_w - 1), np.clip(orig_y, 0, orig_h - 1)
        
        h, s, v = map(int, self.hsv_image[orig_y, orig_x])
        
        self.h_lower_slider.set(max(0, h - self.sensitivity))
        self.h_upper_slider.set(min(179, h + self.sensitivity))
        self.s_lower_slider.set(max(0, s - 80)); self.s_upper_slider.set(min(255, s + 80))
        self.v_lower_slider.set(max(0, v - 80)); self.v_upper_slider.set(min(255, v + 80))
        
        self.apply_filters()
        
    def _min_area_callback(self, val):
        min_val = int(val)
        if min_val > self.max_area_slider.get(): self.max_area_slider.set(min_val)
        self.apply_filters()

    def _max_area_callback(self, val):
        max_val = int(val)
        if max_val < self.min_area_slider.get(): self.min_area_slider.set(max_val)
        self.apply_filters()

    def _min_aspect_callback(self, val):
        min_val = int(val)
        if min_val > self.max_aspect_slider.get(): self.max_aspect_slider.set(min_val)
        self.apply_filters()

    def _max_aspect_callback(self, val):
        max_val = int(val)
        if max_val < self.min_aspect_slider.get(): self.min_aspect_slider.set(max_val)
        self.apply_filters()

    def _h_lower_callback(self, val):
        min_val = int(val)
        if min_val > self.h_upper_slider.get(): self.h_upper_slider.set(min_val)
        self.apply_filters()

    def _h_upper_callback(self, val):
        max_val = int(val)
        if max_val < self.h_lower_slider.get(): self.h_lower_slider.set(max_val)
        self.apply_filters()

    def _s_lower_callback(self, val):
        min_val = int(val)
        if min_val > self.s_upper_slider.get(): self.s_upper_slider.set(min_val)
        self.apply_filters()

    def _s_upper_callback(self, val):
        max_val = int(val)
        if max_val < self.s_lower_slider.get(): self.s_lower_slider.set(max_val)
        self.apply_filters()
        
    def _v_lower_callback(self, val):
        min_val = int(val)
        if min_val > self.v_upper_slider.get(): self.v_upper_slider.set(min_val)
        self.apply_filters()

    def _v_upper_callback(self, val):
        max_val = int(val)
        if max_val < self.v_lower_slider.get(): self.v_lower_slider.set(max_val)
        self.apply_filters()

    def apply_filters(self, _=None):
        if self.original_image is None: return

        lower_bound = np.array([self.h_lower_slider.get(), self.s_lower_slider.get(), self.v_lower_slider.get()])
        upper_bound = np.array([self.h_upper_slider.get(), self.s_upper_slider.get(), self.v_upper_slider.get()])
        messy_mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)

        kernel = np.ones((5, 5), np.uint8)
        messy_mask = cv2.morphologyEx(messy_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        messy_mask = cv2.morphologyEx(messy_mask, cv2.MORPH_CLOSE, kernel, iterations=1)

        contours, _ = cv2.findContours(messy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        clean_mask = np.zeros_like(messy_mask)
        min_area, max_area = self.min_area_slider.get(), self.max_area_slider.get()
        min_aspect = self.min_aspect_slider.get() / 10.0
        max_aspect = self.max_aspect_slider.get() / 10.0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if not (min_area < area < max_area): continue

            x, y, w, h = cv2.boundingRect(cnt)
            if h == 0: continue
            aspect_ratio = w / float(h)
            if not (min_aspect < aspect_ratio < max_aspect): continue
            
            cv2.drawContours(clean_mask, [cnt], -1, 255, -1)
        
        self.display_image(clean_mask, self.panel_mask)

    def display_image(self, img_data, panel, max_size=500):
        h, w = img_data.shape[:2]
        if max(h, w) > max_size:
            ratio = max_size / max(h, w)
            new_w, new_h = int(w * ratio), int(h * ratio)
            img_data = cv2.resize(img_data, (new_w, new_h), interpolation=cv2.INTER_AREA)
            
        if len(img_data.shape) == 3: img_data = cv2.cvtColor(img_data, cv2.COLOR_BGR2RGB)
        
        img = Image.fromarray(img_data); img_tk = ImageTk.PhotoImage(image=img)
        panel.config(image=img_tk); panel.image = img_tk

if __name__ == "__main__":
    try:
        root = tk.Tk()
        app = ColorDetectorApp(root)
        root.geometry("1200x700")
        root.mainloop()
    except ImportError:
        print("Error: Missing required libraries.")
        print("Please install them using: pip install opencv-python-headless pillow numpy")

