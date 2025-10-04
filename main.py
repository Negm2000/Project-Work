import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import cv2
import numpy as np
from PIL import Image, ImageTk

class RangeSlider(tk.Frame):
    """
    A custom Tkinter widget that acts as a slider with two handles for selecting a range.
    """
    def __init__(self, parent, label_text, min_val, max_val, initial_min, initial_max, command):
        super().__init__(parent, bg="#3C3C3C")
        self.command = command
        self.min_val = min_val
        self.max_val = max_val
        self.initial_min = initial_min
        self.initial_max = initial_max
        self.active_handle = None
        self.handle_width = 10

        # Layout
        self.label_frame = tk.Frame(self, bg="#3C3C3C")
        self.label_frame.pack(fill='x', pady=(10,0))
        tk.Label(self.label_frame, text=label_text, bg="#3C3C3C", fg="white").pack(side='left')
        self.value_label = tk.Label(self.label_frame, text=f"[{initial_min} - {initial_max}]", bg="#3C3C3C", fg="#A9A9A9")
        self.value_label.pack(side='right')

        self.canvas = tk.Canvas(self, bg="#2E2E2E", height=20, highlightthickness=0)
        self.canvas.pack(fill='x', pady=5)
        self.canvas.bind("<Configure>", self._on_resize)
        self.canvas.bind("<ButtonPress-1>", self._on_press)
        self.canvas.bind("<ButtonRelease-1>", self._on_release)
        self.canvas.bind("<B1-Motion>", self._on_drag)

    def _on_resize(self, event):
        self.canvas_width = event.width
        # This call creates the handles and then calls set() to position them,
        # fixing the initialization order issue.
        self._draw_slider()

    def _draw_slider(self):
        self.canvas.delete("all")
        # Trough
        self.canvas.create_line(self.handle_width / 2, 10, self.canvas_width - self.handle_width / 2, 10, fill="#555", width=2)
        # Handles
        self.handle1_id = self.canvas.create_rectangle(0, 0, 0, 0, fill="#A9A9A9", outline="#E0E0E0")
        self.handle2_id = self.canvas.create_rectangle(0, 0, 0, 0, fill="#A9A9A9", outline="#E0E0E0")
        self.set(self.initial_min, self.initial_max)

    def set(self, min_v, max_v):
        self.initial_min = min_v
        self.initial_max = max_v
        
        pos1 = self._value_to_coords(min_v)
        pos2 = self._value_to_coords(max_v)
        
        self.canvas.coords(self.handle1_id, pos1 - self.handle_width/2, 2, pos1 + self.handle_width/2, 18)
        self.canvas.coords(self.handle2_id, pos2 - self.handle_width/2, 2, pos2 + self.handle_width/2, 18)
        
        self.value_label.config(text=f"[{int(min_v)} - {int(max_v)}]")

    def get(self):
        coords1 = self.canvas.coords(self.handle1_id)
        coords2 = self.canvas.coords(self.handle2_id)
        val1 = self._coords_to_value(coords1[0] + self.handle_width/2)
        val2 = self._coords_to_value(coords2[0] + self.handle_width/2)
        return min(val1, val2), max(val1, val2)

    def _coords_to_value(self, x):
        range_ = self.canvas_width - self.handle_width
        if range_ <= 0: return self.min_val # Prevent division by zero if canvas is tiny
        return self.min_val + (x - self.handle_width/2) / range_ * (self.max_val - self.min_val)

    def _value_to_coords(self, val):
        range_ = self.canvas_width - self.handle_width
        value_range = self.max_val - self.min_val
        if value_range == 0: return self.handle_width / 2 # Prevent division by zero
        return self.handle_width/2 + (val - self.min_val) / value_range * range_

    def _on_press(self, event):
        x = event.x
        c1 = self.canvas.coords(self.handle1_id)
        c2 = self.canvas.coords(self.handle2_id)
        if c1[0] <= x <= c1[2]:
            self.active_handle = self.handle1_id
        elif c2[0] <= x <= c2[2]:
            self.active_handle = self.handle2_id

    def _on_release(self, event):
        self.active_handle = None
        min_v, max_v = self.get()
        self.initial_min, self.initial_max = min_v, max_v # Update state
        if self.command: self.command()

    def _on_drag(self, event):
        if not self.active_handle: return
        x = np.clip(event.x, self.handle_width/2, self.canvas_width - self.handle_width/2)
        
        # REMOVED the overly strict collision detection. The get() method already sorts the
        # min and max values, so we can let the handles pass each other freely.
        
        self.canvas.coords(self.active_handle, x - self.handle_width/2, 2, x + self.handle_width/2, 18)
        min_v, max_v = self.get()
        self.value_label.config(text=f"[{int(min_v)} - {int(max_v)}]")
        if self.command: self.command()

class ColorDetectorApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Component Detector")
        self.root.configure(bg="#2E2E2E")

        self.original_image = None; self.hsv_image = None; self.sensitivity = 15

        top_frame = tk.Frame(root, bg="#3C3C3C", padx=10, pady=10)
        top_frame.pack(side=tk.TOP, fill=tk.X)
        main_pane = tk.PanedWindow(root, orient=tk.HORIZONTAL, bg="#2E2E2E", sashwidth=8, sashrelief=tk.RAISED)
        main_pane.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        image_frame = tk.Frame(main_pane, bg="#2E2E2E", padx=10, pady=10)
        main_pane.add(image_frame, stretch="always")
        controls_frame = tk.Frame(main_pane, bg="#3C3C3C", padx=15, pady=15, width=280)
        main_pane.add(controls_frame, stretch="never")
        
        self.btn_load = tk.Button(top_frame, text="Load Image", command=self.load_image, bg="#555555", fg="white", relief="flat", padx=10, pady=5)
        self.btn_load.pack(side=tk.LEFT, padx=5)
        self.info_label = tk.Label(top_frame, text="Load an image and click on a color to begin.", bg="#3C3C3C", fg="white")
        self.info_label.pack(side=tk.LEFT, padx=10)
        
        self.panel_original = tk.Label(image_frame, bg="#2E2E2E"); self.panel_original.pack(side=tk.LEFT, padx=10, pady=5, expand=True)
        self.panel_original.bind("<Button-1>", self.pick_color)
        self.panel_mask = tk.Label(image_frame, bg="#2E2E2E"); self.panel_mask.pack(side=tk.RIGHT, padx=10, pady=5, expand=True)
        tk.Label(image_frame, text="Original Image", bg="#2E2E2E", fg="white").place(in_=self.panel_original, relx=0.5, y=-20, anchor='n')
        tk.Label(image_frame, text="Filtered Mask", bg="#2E2E2E", fg="white").place(in_=self.panel_mask, relx=0.5, y=-20, anchor='n')

        self.create_filter_controls(controls_frame)

    def create_filter_controls(self, parent):
        tk.Label(parent, text="Color Tuning (HSV)", font=("Helvetica", 12, "bold"), bg="#3C3C3C", fg="white").pack(pady=(0, 10), anchor='w')
        self.h_slider = RangeSlider(parent, "Hue (H)", 0, 179, 0, 179, self.apply_filters)
        self.s_slider = RangeSlider(parent, "Saturation (S)", 0, 255, 0, 255, self.apply_filters)
        self.v_slider = RangeSlider(parent, "Value (V)", 0, 255, 0, 255, self.apply_filters)
        self.h_slider.pack(fill='x'); self.s_slider.pack(fill='x'); self.v_slider.pack(fill='x')
        
        tk.Label(parent, text="Shape & Size Filters", font=("Helvetica", 12, "bold"), bg="#3C3C3C", fg="white").pack(pady=(20, 10), anchor='w')
        self.area_slider = RangeSlider(parent, "Area", 0, 50000, 100, 20000, self.apply_filters)
        self.aspect_slider = RangeSlider(parent, "Aspect Ratio (W/H)", 0, 100, 1, 50, self.apply_filters) # Represents 0.0-10.0
        self.area_slider.pack(fill='x'); self.aspect_slider.pack(fill='x')

    def load_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not path: return
        try:
            self.original_image = cv2.imread(path)
            if self.original_image is None: raise ValueError("Could not read image.")
            self.hsv_image = cv2.cvtColor(self.original_image, cv2.COLOR_BGR2HSV)
            h, w, _ = self.original_image.shape
            max_area_limit = (h * w) // 4
            self.area_slider.min_val = 0; self.area_slider.max_val = max_area_limit
            self.area_slider.set(100, min(20000, max_area_limit))

            self.display_image(self.original_image, self.panel_original)
            blank_mask = np.zeros(self.original_image.shape[:2], dtype="uint8")
            self.display_image(blank_mask, self.panel_mask)
        except Exception as e: messagebox.showerror("Error", f"Failed to load image: {e}")

    def pick_color(self, event):
        if self.hsv_image is None: return
        img_w, img_h = self.panel_original.image.width(), self.panel_original.image.height()
        orig_h, orig_w, _ = self.original_image.shape
        orig_x, orig_y = int(event.x * (orig_w/img_w)), int(event.y * (orig_h/img_h))
        orig_x, orig_y = np.clip(orig_x, 0, orig_w - 1), np.clip(orig_y, 0, orig_h - 1)
        
        h, s, v = map(int, self.hsv_image[orig_y, orig_x])
        
        self.h_slider.set(max(0, h - self.sensitivity), min(179, h + self.sensitivity))
        self.s_slider.set(max(0, s - 80), min(255, s + 80))
        self.v_slider.set(max(0, v - 80), min(255, v + 80))
        
        self.apply_filters()

    def apply_filters(self, _=None):
        if self.original_image is None: return
        
        h_min, h_max = self.h_slider.get(); s_min, s_max = self.s_slider.get(); v_min, v_max = self.v_slider.get()
        lower_bound = np.array([h_min, s_min, v_min]); upper_bound = np.array([h_max, s_max, v_max])
        messy_mask = cv2.inRange(self.hsv_image, lower_bound, upper_bound)

        kernel = np.ones((5, 5), np.uint8)
        messy_mask = cv2.morphologyEx(messy_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        messy_mask = cv2.morphologyEx(messy_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        contours, _ = cv2.findContours(messy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        clean_mask = np.zeros_like(messy_mask)
        min_area, max_area = self.area_slider.get()
        min_aspect, max_aspect = self.aspect_slider.get()
        min_aspect /= 10.0; max_aspect /= 10.0

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
        print("Error: Missing required libraries.\n"
              "Please install them using: pip install opencv-python-headless pillow numpy")



