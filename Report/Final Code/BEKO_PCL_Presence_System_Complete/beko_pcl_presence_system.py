#!/usr/bin/env python3
"""
BEKO PCL PRESENCE DETECTION SYSTEM
Interfaccia grafica per analisi PCB connector con modelli PCL Presence Detection.
Basato su beko_detection_system.py ma usa i nuovi modelli di classificazione PCL.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from tkinterdnd2 import DND_FILES, TkinterDnD
import torch
import torch.nn as nn
from torchvision.models import resnet18
from PIL import Image, ImageTk
from torchvision import transforms
import numpy as np
import cv2
import json
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Rectangle, ConnectionPatch
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
from dataclasses import dataclass
import tempfile
import threading
import sys
import random
import datetime
from matplotlib.gridspec import GridSpec
try:
    import pygame
    PYGAME_AVAILABLE = True
except ImportError:
    PYGAME_AVAILABLE = False

# ============================================================================
# CLASSI MODELLI
# ============================================================================

class OcclusionCNN(nn.Module):
    """CNN per classificazione OCCLUSION vs VISIBLE."""
    def __init__(self):
        super(OcclusionCNN, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(128 * 16 * 16, 512)
        self.dropout = nn.Dropout(0.5)
        self.fc2 = nn.Linear(512, 2)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.pool1(self.relu(self.bn1(self.conv1(x))))
        x = self.pool2(self.relu(self.bn2(self.conv2(x))))
        x = self.pool3(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x

class PCLPresenceClassifier(nn.Module):
    """ResNet18-based binary classifier for PCL presence detection."""
    
    def __init__(self, input_channels: int = 1, pretrained: bool = False):
        super().__init__()
        # Load ResNet18
        try:
            from torchvision.models import ResNet18_Weights
            weights = ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = resnet18(weights=weights)
        except (ImportError, AttributeError):
            self.backbone = resnet18(pretrained=pretrained)
        
        # Modify first layer for grayscale input
        if input_channels == 1:
            self.backbone.conv1 = nn.Conv2d(
                1, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
        
        # Replace classifier head for binary classification
        self.backbone.fc = nn.Linear(self.backbone.fc.in_features, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)

# ============================================================================
# FUNZIONI PREPROCESSING
# ============================================================================

@dataclass
class RelativeROI:
    name: str
    x_min_rel: float
    y_min_rel: float
    x_max_rel: float
    y_max_rel: float
    
    def to_pixel_box(self, width: int, height: int, margin: int = 0):
        x_min = int(self.x_min_rel * width)
        y_min = int(self.y_min_rel * height)
        x_max = int(self.x_max_rel * width)
        y_max = int(self.y_max_rel * height)
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(width, x_max + margin)
        y_max = min(height, y_max + margin)
        return x_min, y_min, x_max, y_max

def load_roi_config(roi_config_path):
    """Carica configurazione ROI da JSON."""
    with open(roi_config_path, 'r') as f:
        data = json.load(f)
    rois = []
    for entry in data:
        rois.append(RelativeROI(
            name=entry["name"],
            x_min_rel=float(entry["x_min_rel"]),
            y_min_rel=float(entry["y_min_rel"]),
            x_max_rel=float(entry["x_max_rel"]),
            y_max_rel=float(entry["y_max_rel"])
        ))
    return rois

def detect_homography(template_gray, candidate_gray, max_features=4000, good_match_percent=0.15):
    """Rileva omografia tra template e candidato."""
    orb = cv2.ORB_create(nfeatures=max_features, fastThreshold=5, scaleFactor=1.2)
    kp1, des1 = orb.detectAndCompute(template_gray, None)
    kp2, des2 = orb.detectAndCompute(candidate_gray, None)
    
    if des1 is None or des2 is None:
        raise RuntimeError("Could not extract ORB descriptors")
    
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    
    if len(matches) < 15:
        knn = matcher.knnMatch(des1, des2, k=2)
        matches = []
        for m, n in knn:
            if m.distance < 0.75 * n.distance:
                matches.append(m)
    
    if not matches:
        raise RuntimeError("No matches found")
    
    matches = sorted(matches, key=lambda m: m.distance)
    keep = max(4, int(len(matches) * good_match_percent))
    matches = matches[:keep]
    
    src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    
    H, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 4.0)
    if H is None or mask is None or mask.sum() < 8:
        raise RuntimeError("Homography estimation failed")
    return H

def normalize_lighting(image_bgr):
    """Normalizza l'illuminazione dell'immagine."""
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l = clahe.apply(l)
    lab = cv2.merge((l, a, b))
    balanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    
    # Gray-world white balance
    avg_b = np.mean(balanced[:, :, 0])
    avg_g = np.mean(balanced[:, :, 1])
    avg_r = np.mean(balanced[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    scale = np.array([avg_gray / avg_b, avg_gray / avg_g, avg_gray / avg_r])
    wb = balanced.astype(np.float32)
    wb *= scale
    wb = np.clip(wb, 0, 255).astype(np.uint8)
    return wb

def ecc_homography(template_gray, candidate_gray, iterations=200, epsilon=1e-6):
    """Dense alignment fallback usando ECC."""
    warp_matrix = np.eye(3, dtype=np.float32)
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, iterations, epsilon)
    template = template_gray.astype(np.float32) / 255.0
    candidate = candidate_gray.astype(np.float32) / 255.0
    try:
        _, warp_matrix = cv2.findTransformECC(
            template, candidate, warp_matrix, cv2.MOTION_HOMOGRAPHY, criteria, None, 5
        )
    except cv2.error as exc:
        raise RuntimeError(f"ECC optimization failed: {exc}") from exc
    return warp_matrix

def align_image(image_path, reference_path=None, crop_box=None):
    """Allinea un'immagine a un riferimento."""
    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Se non c'è riferimento, assume che l'immagine sia già allineata
    if reference_path is None:
        normalized = normalize_lighting(image)
        if crop_box:
            xmin, ymin, xmax, ymax = crop_box
            normalized = normalized[ymin:ymax, xmin:xmax]
        return normalized
    
    # Allinea usando omografia
    template = cv2.imread(str(reference_path), cv2.IMREAD_COLOR)
    if template is None:
        raise FileNotFoundError(f"Could not read reference: {image_path}")
    
    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Prova prima con ORB, poi con ECC se fallisce
    try:
        H = detect_homography(template_gray, gray)
    except RuntimeError:
        # Fallback a ECC
        H = ecc_homography(template_gray, gray)
    
    # Warp l'immagine
    warped = cv2.warpPerspective(image, H, (template.shape[1], template.shape[0]))
    
    # Applica crop box se fornito
    if crop_box:
        xmin, ymin, xmax, ymax = crop_box
        warped = warped[ymin:ymax, xmin:xmax]
    
    # Normalizza illuminazione
    normalized = normalize_lighting(warped)
    return normalized

def normalize_roi(img: np.ndarray) -> np.ndarray:
    """Convert ROI to grayscale, apply CLAHE, then normalize to float32 [0, 1]."""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    equalized = clahe.apply(gray)
    normalized = equalized.astype(np.float32) / 255.0
    return normalized

def extract_connectors(aligned_image, rois, margin=8):
    """Estrae i connettori da un'immagine allineata."""
    height, width = aligned_image.shape[:2]
    connectors = []
    
    for roi in rois:
        x_min, y_min, x_max, y_max = roi.to_pixel_box(width, height, margin=margin)
        crop_bgr = aligned_image[y_min:y_max, x_min:x_max].copy()
        
        # Normalizza come in crop_connectors.py (grayscale + CLAHE)
        normalized = normalize_roi(crop_bgr)
        
        # Converti a uint8 per salvare/visualizzare
        crop_normalized = (normalized * 255).astype(np.uint8)
        
        connectors.append({
            'name': roi.name,
            'crop': crop_normalized,  # Grayscale normalizzato
            'crop_bgr': crop_bgr,  # Mantieni BGR per visualizzazione originale
            'bbox': (x_min, y_min, x_max, y_max)
        })
    
    return connectors

# ============================================================================
# FUNZIONI CLASSIFICAZIONE
# ============================================================================

def preprocess_image(image_path, device=None):
    """Preprocessa immagine per classificatore OCCLUSION."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchvision import transforms
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Carica immagine (può essere grayscale o RGB)
    image = Image.open(image_path)
    # Se è grayscale, converti in RGB (ripeti canale 3 volte)
    if image.mode == 'L':
        image = image.convert('RGB')
    elif image.mode != 'RGB':
        image = image.convert('RGB')
    image_tensor = transform(image)
    image_tensor = image_tensor.unsqueeze(0).to(device)
    return image_tensor

def preprocess_image_for_pcl(image_path, device=None):
    """Preprocessa immagine per classificatore PCL Presence (grayscale)."""
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Carica immagine come grayscale
    img = cv2.imread(str(image_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(image_path).convert('L'))
    
    # Preprocess: aggiungi dimensione canale e normalizza
    if len(img.shape) == 2:
        img = img[np.newaxis, :, :]  # Add channel dimension
    else:
        img = img[:, :, 0:1].transpose(2, 0, 1)  # Take first channel
    
    img_tensor = torch.from_numpy(img).float() / 255.0
    img_tensor = img_tensor.unsqueeze(0).to(device)
    
    return img_tensor

def classify_connector_pcl(image_path, connector_name, occ_model, pcl_models_dict, device=None):
    """Classifica un connettore come OK, KO o OCCLUSION.
    
    STEP 1: Verifica occlusione
    STEP 2: Se non occluso, usa modello PCL Presence per OK/KO
    
    Args:
        image_path: Path all'immagine o array numpy
        connector_name: Nome del connettore (conn1, conn2, ...)
        occ_model: Modello per classificazione OCCLUSION
        pcl_models_dict: Dizionario con modelli PCL {connector_name: (model, threshold)}
        device: Device PyTorch
    
    Returns:
        (label, prob_ko, None, None): 
            - label: "OK", "KO" o "OCCLUSION"
            - prob_ko: Probabilità che il PCL sia mancante (KO) o 0.0 se OCCLUSION
            - None, None: Per compatibilità con il codice esistente
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # STEP 1: Verifica occlusione
    x_occ = preprocess_image(image_path, device)
    with torch.no_grad():
        logits = occ_model(x_occ)
        pred_vis = torch.argmax(logits, dim=1).item()
    
    if pred_vis == 0:  # OCCLUSION
        return "OCCLUSION", 0.0, None, None
    
    # STEP 2: Anomaly detection con PCL Presence
    if connector_name not in pcl_models_dict:
        raise ValueError(f"Modello non trovato per {connector_name}")
    
    model, threshold = pcl_models_dict[connector_name]
    
    x_pcl = preprocess_image_for_pcl(image_path, device)
    with torch.no_grad():
        output = model(x_pcl).squeeze()
        prob_ko = torch.sigmoid(output).item()
    
    if prob_ko > threshold:
        return "KO", prob_ko, None, None
    else:
        return "OK", prob_ko, None, None

# ============================================================================
# CONFIGURAZIONE
# ============================================================================

# Paths per suoni di alert (optional, relative to package)
KO_ALERT_SOUND_PATH = "sounds/alert_ko.wav"
OCCLUSION_ALERT_SOUND_PATH = "sounds/alert_occlusion.wav"

# Path per schematica board
BOARD_SCHEMATIC_PATH = "Codice/board_schematic.png"

# Intervallo simulazione (secondi)
SIMULATION_INTERVAL_SECONDS = 15

# Immagini KO per test
KO_TEST_IMAGES = [
    "Data/TOP 1/20251106131615_TOP.png",
    "Data/TOP 1/20251119083617_TOP.png",
    "Data/TOP 1/20251106164433_TOP.png",
    "Data/TOP 1/20251119083538_TOP.png",
    "Data/TOP 1/20251106164844_TOP.png",
    "Data/TOP 1/20251106164620_TOP.png",
]

# Numero di analisi da mostrare nel log
ANALYSIS_LOG_SIZE = 5

# ============================================================================
# INTERFACCIA GRAFICA
# ============================================================================

class BekoPCLPresenceSystem:
    def __init__(self, root):
        self.root = root
        self.root.title("BEKO PCL PRESENCE DETECTION SYSTEM")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#ffffff')
        
        # Variabili
        self.image_path = None
        self.raw_image_path = None
        self.aligned_image = None
        self.connectors = []
        self.results = []
        self.occ_model = None  # Modello per classificazione OCCLUSION
        self.pcl_models = {}  # Dizionario: connector_name -> (model, threshold)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Variabili simulazione
        self.simulation_active = False
        self.ko_test_queue = []
        self.ko_test_index = 0
        self.simulation_images = []
        self.analysis_log = []
        
        # Variabili audio
        self.sounds_muted = False
        if PYGAME_AVAILABLE:
            try:
                pygame.mixer.init()
            except:
                pass
        
        # Paths - All paths are relative to the package directory
        self.package_root = Path(__file__).parent
        self.weights_dir = self.package_root / "weights"
        self.models_dir = self.package_root / "models"
        self.roi_config_path = self.package_root / "config" / "roi_config.json"
        
        # Paths logo (optional, will work if present)
        self.logo_polimi_path = self.package_root / "Logo polimi.png"
        self.logo_beko_path = self.package_root / "Logo beko.jpg"
        
        # Paths per riferimento e crop box (optional)
        self.reference_path = None
        self.crop_box = None
        
        # Cerca riferimento in package directory (optional)
        reference_dir = self.package_root / "reference"
        if reference_dir.exists():
            refs = list(reference_dir.glob("*.png"))
            if refs:
                self.reference_path = refs[0]
                self.crop_box = None
        
        self.setup_ui()
        self.load_models()
        self.init_simulation_pool()
        self.print_threshold_info()
    
    def setup_ui(self):
        # Header con logo
        header_frame = tk.Frame(self.root, bg='#ffffff', height=80)
        header_frame.pack(fill=tk.X, padx=0, pady=0)
        header_frame.pack_propagate(False)
        
        header_content = tk.Frame(header_frame, bg='#ffffff')
        header_content.pack(fill=tk.BOTH, expand=True, padx=20, pady=12)
        
        # Logo sinistra (Politecnico)
        if self.logo_polimi_path.exists():
            try:
                logo_polimi_img = Image.open(self.logo_polimi_path)
                logo_polimi_img = logo_polimi_img.resize((180, 55), Image.Resampling.LANCZOS)
                logo_polimi_photo = ImageTk.PhotoImage(logo_polimi_img)
                logo_polimi_label = tk.Label(
                    header_content,
                    image=logo_polimi_photo,
                    bg='#ffffff'
                )
                logo_polimi_label.image = logo_polimi_photo
                logo_polimi_label.pack(side=tk.LEFT, padx=(0, 15))
            except Exception as e:
                print(f"Error loading Politecnico logo: {e}")
        
        # Titolo centrale
        title_frame = tk.Frame(header_content, bg='#ffffff')
        title_frame.pack(side=tk.LEFT, expand=True, fill=tk.BOTH)
        
        title_label = tk.Label(
            title_frame,
            text="BEKO PCL PRESENCE DETECTION SYSTEM",
            font=("Arial", 20, "bold"),
            bg='#ffffff',
            fg='#1a1a1a'
        )
        title_label.pack(anchor='w')
        
        subtitle_label = tk.Label(
            title_frame,
            text="PCB Connector PCL Presence Quality Control",
            font=("Arial", 11),
            bg='#ffffff',
            fg='#666666'
        )
        subtitle_label.pack(anchor='w', pady=(2, 0))
        
        
        # Logo destra (Beko)
        if self.logo_beko_path.exists():
            try:
                logo_beko_img = Image.open(self.logo_beko_path)
                aspect = logo_beko_img.width / logo_beko_img.height
                new_height = 55
                new_width = int(new_height * aspect)
                logo_beko_img = logo_beko_img.resize((new_width, new_height), Image.Resampling.LANCZOS)
                logo_beko_photo = ImageTk.PhotoImage(logo_beko_img)
                logo_beko_label = tk.Label(
                    header_content,
                    image=logo_beko_photo,
                    bg='#ffffff'
                )
                logo_beko_label.image = logo_beko_photo
                logo_beko_label.pack(side=tk.RIGHT, padx=(15, 0))
            except Exception as e:
                print(f"Error loading Beko logo: {e}")
        
        # Separator
        separator = tk.Frame(self.root, bg='#e0e0e0', height=1)
        separator.pack(fill=tk.X, padx=0, pady=0)
        
        # Main container
        main_frame = tk.Frame(self.root, bg='#F5F5F5')
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # LEFT COLUMN - Control Panel
        left_panel = tk.Frame(main_frame, bg='#ffffff', width=360)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 20))
        left_panel.pack_propagate(False)
        
        # 1. Image Input Section
        input_frame = tk.Frame(left_panel, bg='#ffffff')
        input_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        input_title = tk.Label(
            input_frame,
            text="Image Input",
            font=("Arial", 13, "bold"),
            bg='#ffffff',
            fg='#1a1a1a',
            anchor='w'
        )
        input_title.pack(fill=tk.X, pady=(0, 8))
        
        # Drag & Drop area
        upload_inner = tk.Frame(input_frame, bg='#F5F5F5', relief=tk.SOLID, bd=1)
        upload_inner.pack(fill=tk.X, pady=(0, 8))
        
        upload_label = tk.Label(
            upload_inner,
            text="Drag image here or click to select",
            font=("Arial", 10),
            bg='#F5F5F5',
            fg='#666666',
            justify=tk.CENTER,
            cursor='hand2',
            pady=20
        )
        upload_label.pack(fill=tk.X)
        
        upload_inner.drop_target_register(DND_FILES)
        upload_inner.dnd_bind('<<Drop>>', self.on_drop)
        upload_label.bind("<Button-1>", self.on_click_select)
        upload_inner.bind("<Button-1>", self.on_click_select)
        
        self.filename_label = tk.Label(
            input_frame,
            text="No file selected",
            font=("Arial", 9),
            bg='#ffffff',
            fg='#999999',
            anchor='w'
        )
        self.filename_label.pack(fill=tk.X, pady=(0, 4))
        
        self.timestamp_label = tk.Label(
            input_frame,
            text="",
            font=("Arial", 8),
            bg='#ffffff',
            fg='#999999',
            anchor='w'
        )
        self.timestamp_label.pack(fill=tk.X, pady=(0, 12))
        
        self.process_btn = tk.Button(
            input_frame,
            text="ANALYZE IMAGE",
            font=("Arial", 11),
            bg='#007bff',
            fg='#ffffff',
            command=self.process_image,
            state=tk.DISABLED,
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            pady=10,
            activebackground='#0056b3',
            activeforeground='#ffffff'
        )
        self.process_btn.pack(fill=tk.X)
        
        sep1 = tk.Frame(left_panel, bg='#e0e0e0', height=1)
        sep1.pack(fill=tk.X, padx=20, pady=20)
        
        # 2. Simulation Controls
        sim_frame = tk.Frame(left_panel, bg='#ffffff')
        sim_frame.pack(fill=tk.X, padx=20, pady=(0, 20))
        
        sim_title = tk.Label(
            sim_frame,
            text="Simulation",
            font=("Arial", 13, "bold"),
            bg='#ffffff',
            fg='#1a1a1a',
            anchor='w'
        )
        sim_title.pack(fill=tk.X, pady=(0, 10))
        
        self.start_sim_btn = tk.Button(
            sim_frame,
            text="Start",
            font=("Arial", 10),
            bg='#28a745',
            fg='#ffffff',
            command=self.start_simulation,
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            pady=8,
            activebackground='#218838',
            activeforeground='#ffffff'
        )
        self.start_sim_btn.pack(fill=tk.X, pady=(0, 6))
        
        desc1 = tk.Label(
            sim_frame,
            text="Start automatic analysis",
            font=("Arial", 8),
            bg='#ffffff',
            fg='#999999',
            anchor='w'
        )
        desc1.pack(fill=tk.X, pady=(0, 8))
        
        self.stop_sim_btn = tk.Button(
            sim_frame,
            text="Stop",
            font=("Arial", 10),
            bg='#dc3545',
            fg='#ffffff',
            command=self.stop_simulation,
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            pady=8,
            state=tk.DISABLED,
            activebackground='#c82333',
            activeforeground='#ffffff'
        )
        self.stop_sim_btn.pack(fill=tk.X, pady=(0, 6))
        
        desc2 = tk.Label(
            sim_frame,
            text="Stop simulation",
            font=("Arial", 8),
            bg='#ffffff',
            fg='#999999',
            anchor='w'
        )
        desc2.pack(fill=tk.X, pady=(0, 8))
        
        self.ko_test_btn = tk.Button(
            sim_frame,
            text="KO Test",
            font=("Arial", 10),
            bg='#ffc107',
            fg='#1a1a1a',
            command=self.add_ko_test,
            relief=tk.FLAT,
            bd=0,
            cursor='hand2',
            pady=8,
            activebackground='#e0a800',
            activeforeground='#1a1a1a'
        )
        self.ko_test_btn.pack(fill=tk.X, pady=(0, 6))
        
        desc3 = tk.Label(
            sim_frame,
            text="Queue KO test image",
            font=("Arial", 8),
            bg='#ffffff',
            fg='#999999',
            anchor='w'
        )
        desc3.pack(fill=tk.X, pady=(0, 12))
        
        self.mute_var = tk.BooleanVar(value=False)
        mute_check = tk.Checkbutton(
            sim_frame,
            text="Mute sounds",
            font=("Arial", 9),
            bg='#ffffff',
            fg='#1a1a1a',
            variable=self.mute_var,
            command=self.toggle_mute,
            anchor='w',
            selectcolor='#ffffff'
        )
        mute_check.pack(fill=tk.X)
        
        sep2 = tk.Frame(left_panel, bg='#e0e0e0', height=1)
        sep2.pack(fill=tk.X, padx=20, pady=20)
        
        # 3. Statistics Panel
        stats_frame = tk.Frame(left_panel, bg='#ffffff')
        stats_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=(0, 20))
        
        stats_title = tk.Label(
            stats_frame,
            text="Statistics",
            font=("Arial", 13, "bold"),
            bg='#ffffff',
            fg='#1a1a1a',
            anchor='w'
        )
        stats_title.pack(fill=tk.X, pady=(0, 12))
        
        self.status_label = tk.Label(
            stats_frame,
            text="System ready",
            font=("Arial", 10),
            bg='#ffffff',
            fg='#28a745',
            anchor='w'
        )
        self.status_label.pack(fill=tk.X, pady=(0, 12))
        
        self.stats_text = tk.Text(
            stats_frame,
            font=("Arial", 9),
            bg='#F5F5F5',
            fg='#1a1a1a',
            wrap=tk.WORD,
            state=tk.DISABLED,
            relief=tk.FLAT,
            bd=0,
            padx=12,
            pady=12
        )
        self.stats_text.pack(fill=tk.BOTH, expand=True)
        
        # RIGHT COLUMN - Main display
        right_panel = tk.Frame(main_frame, bg='#F5F5F5')
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        self.fig = plt.figure(figsize=(16, 12), facecolor='#F5F5F5', dpi=100)
        self.canvas = FigureCanvasTkAgg(self.fig, master=right_panel)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        ax = self.fig.add_subplot(111, facecolor='#F5F5F5')
        ax.text(0.5, 0.5, "Load an image to start analysis", 
                ha='center', va='center', fontsize=12, color='#999999',
                transform=ax.transAxes)
        ax.axis('off')
        self.canvas.draw()
    
    def load_models(self):
        """Carica i modelli PCL Presence addestrati."""
        try:
            self.status_label.config(text="Loading models...", fg='#ffc107')
            self.root.update()
            
            # Carica occlusion model
            occ_path = self.models_dir / "occlusion_cnn.pth"
            if not occ_path.exists():
                raise FileNotFoundError(f"Modello occlusion non trovato: {occ_path}")
            
            self.occ_model = OcclusionCNN().to(self.device)
            self.occ_model.load_state_dict(torch.load(occ_path, map_location=self.device))
            self.occ_model.eval()
            print("✅ Caricato modello OcclusionCNN")
            
            connectors = [f"conn{i}" for i in range(1, 10)]
            loaded_count = 0
            
            for connector_name in connectors:
                conn_dir = self.weights_dir / connector_name
                
                if not conn_dir.exists():
                    print(f"⚠️  Directory non trovata per {connector_name}: {conn_dir}")
                    continue
                
                # Trova file modello (può essere model.pt o model (N).pt)
                model_files = list(conn_dir.glob("model*.pt"))
                if not model_files:
                    print(f"⚠️  Nessun file modello trovato per {connector_name}")
                    continue
                model_path = model_files[0]
                
                # Trova file threshold (può essere threshold.json o threshold (N).json)
                threshold_files = list(conn_dir.glob("threshold*.json"))
                if not threshold_files:
                    print(f"⚠️  Nessun file threshold trovato per {connector_name}")
                    continue
                threshold_path = threshold_files[0]
                
                # Carica modello
                model = PCLPresenceClassifier(input_channels=1, pretrained=False)
                model.load_state_dict(torch.load(model_path, map_location=self.device))
                model = model.to(self.device)
                model.eval()
                
                # Carica threshold
                with open(threshold_path, 'r') as f:
                    threshold_data = json.load(f)
                threshold = threshold_data['threshold']
                
                self.pcl_models[connector_name] = (model, threshold)
                loaded_count += 1
                print(f"✅ Caricato modello per {connector_name}, threshold: {threshold:.6f}")
            
            if loaded_count == 0:
                raise FileNotFoundError("Nessun modello PCL Presence trovato")
            
            self.status_label.config(text=f"Models loaded ({loaded_count} connectors)", fg='#28a745')
            print(f"✅ Caricati {loaded_count} modelli PCL Presence")
            
        except Exception as e:
            messagebox.showerror("Errore", f"Errore caricamento modelli:\n{e}")
            self.status_label.config(text="Error loading models", fg='#dc3545')
    
    def print_threshold_info(self):
        """Stampa informazioni sui thresholds."""
        print(f"\n{'='*60}")
        print(f"🔍 INFORMAZIONI SUL SISTEMA PCL PRESENCE DETECTION")
        print(f"{'='*60}")
        
        if self.pcl_models:
            print(f"\n📊 THRESHOLDS PCL PRESENCE (per connettore):")
            for conn_name, (_, threshold) in sorted(self.pcl_models.items()):
                print(f"  {conn_name}: {threshold:.6f}")
        
        print(f"{'='*60}\n")
    
    def on_drop(self, event):
        """Gestisce il drag & drop."""
        files = self.root.tk.splitlist(event.data)
        if files:
            self.image_path = files[0].strip('{}')
            self.raw_image_path = self.image_path
            filename = Path(self.image_path).name
            self.filename_label.config(text=filename[:40] + "..." if len(filename) > 40 else filename)
            self.timestamp_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.status_label.config(text="Image loaded", fg='#28a745')
            self.process_btn.config(state=tk.NORMAL)
    
    def on_click_select(self, event):
        """Apre file dialog."""
        file_path = filedialog.askopenfilename(
            title="Select PCB Image",
            filetypes=[("Image files", "*.png *.jpg *.jpeg *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.image_path = file_path
            self.raw_image_path = file_path
            filename = Path(self.image_path).name
            self.filename_label.config(text=filename[:40] + "..." if len(filename) > 40 else filename)
            self.timestamp_label.config(text=datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
            self.status_label.config(text="Image loaded", fg='#28a745')
            self.process_btn.config(state=tk.NORMAL)
    
    def process_image(self):
        """Processa l'immagine in un thread separato."""
        if not self.image_path:
            return
        
        self.process_btn.config(state=tk.DISABLED)
        self.status_label.config(text="⏳ Processing...", fg='#ffc107')
        self.root.update()
        
        thread = threading.Thread(target=self._process_image_thread)
        thread.daemon = True
        thread.start()
    
    def _process_image_thread(self):
        """Thread per processare l'immagine."""
        try:
            # 1. Allinea immagine
            self.root.after(0, lambda: self.status_label.config(text="Aligning image...", fg='#ffc107'))
            
            ref_path = self.reference_path if self.reference_path and self.reference_path.exists() else None
            crop = self.crop_box
            
            if ref_path:
                self.root.after(0, lambda: self.status_label.config(
                    text=f"⏳ Aligning with reference: {Path(ref_path).name}...", fg='#ffc107'))
            
            self.aligned_image = align_image(self.image_path, reference_path=ref_path, crop_box=crop)
            
            # 2. Carica ROI config
            if not self.roi_config_path.exists():
                raise FileNotFoundError(f"ROI config not found: {self.roi_config_path}")
            rois = load_roi_config(self.roi_config_path)
            
            # 3. Estrai connettori
            self.root.after(0, lambda: self.status_label.config(text="⏳ Extracting connectors...", fg='#ffc107'))
            self.connectors = extract_connectors(self.aligned_image, rois, margin=8)
            
            # 4. Classifica connettori
            self.root.after(0, lambda: self.status_label.config(text="Classifying connectors...", fg='#ffc107'))
            temp_dir = Path(tempfile.mkdtemp())
            self.results = []
            
            for i, conn in enumerate(self.connectors):
                self.root.after(0, lambda idx=i: self.status_label.config(
                    text=f"⏳ Classifying {idx+1}/9...", fg='#ffc107'))
                
                temp_path = temp_dir / f"{conn['name']}.png"
                cv2.imwrite(str(temp_path), conn['crop'])
                
                label, prob_ko, _, _ = classify_connector_pcl(
                    str(temp_path), conn['name'], self.occ_model, self.pcl_models, self.device
                )
                
                threshold = self.pcl_models.get(conn['name'], (None, None))[1] if conn['name'] in self.pcl_models else None
                
                self.results.append({
                    'name': conn['name'],
                    'crop': conn['crop'],
                    'bbox': conn['bbox'],
                    'label': label,
                    'prob_ko': prob_ko,
                    'threshold': threshold
                })
            
            # 5. Calcola statistiche
            ok_count = sum(1 for r in self.results if r['label'] == 'OK')
            ko_count = sum(1 for r in self.results if r['label'] == 'KO')
            occ_count = sum(1 for r in self.results if r['label'] == 'OCCLUSION')
            
            log_entry = {
                'timestamp': datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                'filename': Path(self.image_path).name,
                'ko_count': ko_count,
                'occ_count': occ_count
            }
            self.analysis_log.append(log_entry)
            if len(self.analysis_log) > ANALYSIS_LOG_SIZE * 2:
                self.analysis_log = self.analysis_log[-ANALYSIS_LOG_SIZE:]
            
            if ko_count > 0:
                status_text = f"KO detected: {ko_count} connector{'s' if ko_count > 1 else ''}"
                status_color = '#dc3545'
            elif occ_count > 0:
                status_text = f"OCCLUSION detected: {occ_count} connector{'s' if occ_count > 1 else ''}"
                status_color = '#ffc107'
            else:
                status_text = "All connectors OK"
                status_color = '#28a745'
            
            if ko_count > 0:
                self.root.after(0, lambda: self.play_alert_sound("KO"))
            elif occ_count > 0:
                self.root.after(0, lambda: self.play_alert_sound("OCCLUSION"))
            
            self.root.after(0, self.visualize_results)
            self.root.after(0, lambda: self.status_label.config(text=status_text, fg=status_color))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
            
        except Exception as e:
            self.root.after(0, lambda: messagebox.showerror("Errore", f"Errore durante l'elaborazione:\n{e}"))
            self.root.after(0, lambda: self.status_label.config(text="Error", fg='#dc3545'))
            self.root.after(0, lambda: self.process_btn.config(state=tk.NORMAL))
    
    def visualize_results(self):
        """Visualizza i risultati."""
        self.fig.clear()
        self.fig.set_facecolor('#F5F5F5')
        
        # Layout: top area (conn1-5), board centrale, bottom area (conn6-9)
        gs = GridSpec(3, 1, figure=self.fig, 
                     height_ratios=[0.28, 0.44, 0.28],
                     left=0.04, right=0.97, top=0.95, bottom=0.05, 
                     hspace=0.15)
        
        # Area superiore per connettori 1-5
        self.render_top_area(gs[0, 0])
        
        # Board schematic al centro
        self.render_board_schematic(self.fig.add_subplot(gs[1, 0]))
        
        # Area inferiore per connettori 6-9
        self.render_bottom_area(gs[2, 0])
        
        # Renderizza immagini ingrandite con frecce dalla board schematic
        self.render_zoomed_crops_with_arrows()
        
        self.canvas.draw()
        self.update_statistics()
    
    def generate_board_schematic(self, aligned_image, roi_config):
        """Genera schematica board dall'immagine corrente con ROI colorate."""
        height, width = aligned_image.shape[:2]
        
        # Converti a RGB
        if len(aligned_image.shape) == 2:
            img_rgb = cv2.cvtColor(aligned_image, cv2.COLOR_GRAY2RGB)
        else:
            img_rgb = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2RGB)
        
        # Colori vivaci per ogni connettore (RGB)
        connector_colors = {
            'conn1': (255, 0, 0),        # Rosso acceso
            'conn2': (0, 255, 0),        # Verde acceso
            'conn3': (0, 100, 255),      # Blu acceso
            'conn4': (255, 255, 0),      # Giallo acceso
            'conn5': (255, 0, 255),      # Magenta acceso
            'conn6': (0, 255, 255),      # Cyan acceso
            'conn7': (255, 128, 0),      # Arancione acceso
            'conn8': (200, 0, 255),      # Viola acceso
            'conn9': (0, 255, 128),      # Verde acqua acceso
        }
        
        # Disegna ROI rettangoli con colori
        for roi in roi_config:
            name = roi.name
            x_min_rel = roi.x_min_rel
            y_min_rel = roi.y_min_rel
            x_max_rel = roi.x_max_rel
            y_max_rel = roi.y_max_rel
            
            # Converti coordinate relative a pixel
            x_min = int(x_min_rel * width)
            y_min = int(y_min_rel * height)
            x_max = int(x_max_rel * width)
            y_max = int(y_max_rel * height)
            
            # Colore per questo connettore
            color = connector_colors.get(name, (255, 255, 255))
            
            # Disegna rettangolo con bordo colorato (spessore 3)
            cv2.rectangle(img_rgb, (x_min, y_min), (x_max, y_max), color, 3)
            
            # Aggiungi label connettore
            label_x = x_min + 5
            label_y = y_min + 15
            
            # Sfondo per testo (rettangolo colorato più scuro)
            (text_width, text_height), baseline = cv2.getTextSize(
                name, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            bg_color = tuple(int(c * 0.7) for c in color)
            cv2.rectangle(img_rgb, 
                         (label_x - 2, label_y - text_height - 2),
                         (label_x + text_width + 2, label_y + baseline + 2),
                         bg_color, -1)
            
            # Testo bianco
            cv2.putText(img_rgb, name, (label_x, label_y), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return img_rgb
    
    def render_board_schematic(self, ax):
        """Renderizza la schematica del board generata dall'immagine corrente."""
        ax.set_facecolor('#F5F5F5')
        ax.axis('off')
        
        if self.aligned_image is None:
            ax.text(0.5, 0.5, "No image available", ha='center', va='center', 
                   fontsize=10, color='#999999', transform=ax.transAxes)
            schematic_h, schematic_w = 400, 400
        else:
            try:
                # Carica ROI config
                rois = load_roi_config(self.roi_config_path)
                
                # Genera schematica dall'immagine corrente
                schematic_rgb = self.generate_board_schematic(self.aligned_image, rois)
                
                orig_h, orig_w = schematic_rgb.shape[:2]
                
                ax.imshow(schematic_rgb, aspect='equal', extent=[0, orig_w, orig_h, 0])
                schematic_h, schematic_w = orig_h, orig_w
                
                # Bordo nero attorno alla schematica
                border_rect = Rectangle((0, 0), orig_w, orig_h,
                                     facecolor='none', edgecolor='black',
                                     linewidth=2, zorder=1)
                ax.add_patch(border_rect)
                
                ax.set_xlim(0, schematic_w)
                ax.set_ylim(schematic_h, 0)
                
            except Exception as e:
                print(f"Error generating schematic: {e}")
                import traceback
                traceback.print_exc()
                ax.text(0.5, 0.5, "Error generating schematic", ha='center', va='center', 
                       fontsize=10, color='#999999', transform=ax.transAxes)
                schematic_h, schematic_w = 400, 400
        
        # Evidenzia KO sulla schematica
        try:
            rois = load_roi_config(self.roi_config_path)
            COLORS = {
                'OK': '#28a745',
                'KO': '#dc3545',
                'OCCLUSION': '#ffc107'
            }
            
            ko_connectors = [r for r in self.results if r['label'] == 'KO']
            
            for r in self.results:
                roi = next((roi for roi in rois if roi.name == r['name']), None)
                if roi:
                    x_min = roi.x_min_rel * schematic_w
                    y_min = roi.y_min_rel * schematic_h
                    x_max = roi.x_max_rel * schematic_w
                    y_max = roi.y_max_rel * schematic_h
                    
                    color_hex = COLORS[r['label']]
                    r_val = int(color_hex[1:3], 16) / 255.0
                    g_val = int(color_hex[3:5], 16) / 255.0
                    b_val = int(color_hex[5:7], 16) / 255.0
                    color_rgba = (r_val, g_val, b_val, 0.35)
                    
                    rect = Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                    facecolor=color_rgba, edgecolor='none', 
                                    linewidth=0, zorder=10)
                    ax.add_patch(rect)
            
            self.schematic_ax = ax
            self.schematic_h = schematic_h
            self.schematic_w = schematic_w
            self.ko_connectors_for_arrows = ko_connectors
            
        except Exception as e:
            print(f"Error rendering schematic: {e}")
    
    def render_top_area(self, gs_spec):
        """Renderizza l'area superiore (vuota, per connettori 1-5)."""
        ax = self.fig.add_subplot(gs_spec)
        ax.set_facecolor('#F5F5F5')
        ax.axis('off')
    
    def render_bottom_area(self, gs_spec):
        """Renderizza l'area inferiore (vuota, per connettori 6-9)."""
        ax = self.fig.add_subplot(gs_spec)
        ax.set_facecolor('#F5F5F5')
        ax.axis('off')
        
        ko_connectors = [r for r in self.results if r['label'] == 'KO']
        occ_connectors = [r for r in self.results if r['label'] == 'OCCLUSION']
        
        if not ko_connectors and not occ_connectors:
            ax.text(0.5, 0.5, "All connectors OK", ha='center', va='center',
                   fontsize=14, color='#28a745', transform=ax.transAxes)
    
    def render_zoomed_crops_with_arrows(self):
        """Renderizza immagini ingrandite dei crop KO/OCCLUSION nell'area sottostante con frecce dalla board schematic."""
        ko_connectors = [r for r in self.results if r['label'] == 'KO']
        occ_connectors = [r for r in self.results if r['label'] == 'OCCLUSION']
        connectors_with_issues = ko_connectors + occ_connectors
        
        if not connectors_with_issues or not hasattr(self, 'schematic_ax'):
            return
        
        try:
            import numpy as np
            
            rois = load_roi_config(self.roi_config_path)
            schematic_ax = self.schematic_ax
            
            COLORS = {
                'OK': '#28a745',
                'KO': '#dc3545',
                'OCCLUSION': '#ffc107'
            }
            
            # Ottieni le aree sopra e sotto la board schematic
            schematic_bbox = schematic_ax.get_position()
            
            # Area superiore per connettori 1-5 (sopra la board)
            # La board finisce a schematic_bbox.y0 + schematic_bbox.height
            # L'area superiore inizia da lì e va verso l'alto
            top_area_y_bottom = schematic_bbox.y0 + schematic_bbox.height + 0.02  # Inizio area superiore
            top_area_y_top = 0.95  # Fine area superiore (vicino al top della figura)
            top_area_height = top_area_y_top - top_area_y_bottom
            
            # Area inferiore per connettori 6-9 (sotto la board)
            # La board inizia a schematic_bbox.y0
            # L'area inferiore inizia da lì e va verso il basso
            bottom_area_y_top = schematic_bbox.y0 - 0.02  # Inizio area inferiore
            bottom_area_y_bottom = 0.05  # Fine area inferiore (vicino al bottom della figura)
            bottom_area_height = bottom_area_y_top - bottom_area_y_bottom
            
            # Dimensioni immagine ingrandita (in frazioni della figura) - più grandi
            zoom_width = 0.18
            zoom_height = 0.18
            
            # Distribuisci le immagini nella zona sottostante
            n_issues = len(connectors_with_issues)
            
            # Calcola quante immagini per riga possono stare senza sovrapposizioni
            available_width = schematic_bbox.width
            start_x = schematic_bbox.x0
            
            # Spaziatura minima tra le immagini (per evitare sovrapposizioni) - aumentata
            min_spacing = zoom_width + 0.04  # Larghezza immagine + margine più grande
            
            # Calcola quante immagini per riga
            max_per_row = int(available_width / min_spacing)
            if max_per_row < 1:
                max_per_row = 1
            
            # Calcola numero di righe necessarie
            n_rows = (n_issues + max_per_row - 1) // max_per_row  # Divisione per eccesso
            
            # Altezza disponibile per riga
            row_height = bottom_area_height / n_rows if n_rows > 0 else bottom_area_height
            
            # Separa connettori 1-5 e 6-9, ordinati per numero
            connectors_1_5 = sorted([r for r in connectors_with_issues if int(r['name'][4:]) <= 5], 
                                   key=lambda x: int(x['name'][4:]))
            # Connettori 6-9 ordinati normalmente (conn6, conn7, conn8, conn9) per visualizzazione da dx a sx
            connectors_6_9 = sorted([r for r in connectors_with_issues if int(r['name'][4:]) >= 6], 
                                   key=lambda x: int(x['name'][4:]))
            
            # Renderizza connettori 1-5 nell'area superiore
            for idx, r in enumerate(connectors_1_5):
                conn_name = r['name']
                conn_num = int(conn_name[4:])
                roi = next((roi for roi in rois if roi.name == conn_name), None)
                if not roi or not hasattr(self, 'schematic_w') or not hasattr(self, 'schematic_h'):
                    continue
                
                # Tutte le immagini in una singola riga orizzontale
                n_top = len(connectors_1_5)
                
                # Calcola posizione Y nell'area superiore (centrata verticalmente nell'area)
                zoom_y = top_area_y_bottom + top_area_height / 2 - zoom_height / 2
                
                # Calcola posizione X - distribuzione da sinistra a destra in una riga
                if n_top == 1:
                    zoom_x = start_x + available_width / 2 - zoom_width / 2
                else:
                    # Distribuisci uniformemente da sinistra a destra con spaziatura molto aumentata
                    # Spaziatura tra le immagini (molto aumentata)
                    total_images_width = n_top * zoom_width
                    total_spacing = available_width - total_images_width
                    spacing_between = (total_spacing / (n_top + 1)) * 2.0 if n_top > 1 else 0  # Aumentata del 100%
                    # Posizione X: start_x + spacing + idx * (zoom_width + spacing)
                    zoom_x = start_x + spacing_between + idx * (zoom_width + spacing_between)
                
                zoom_x = max(0.01, min(zoom_x, 0.99 - zoom_width))
                zoom_y = max(0.01, min(zoom_y, 0.99 - zoom_height))
                
                # Crea e renderizza immagine ingrandita (stesso codice di prima)
                self._render_zoomed_image(r, roi, zoom_x, zoom_y, zoom_width, zoom_height, 
                                         schematic_ax, COLORS, rois, True)
            
            # Renderizza connettori 6-9 nell'area inferiore
            for idx, r in enumerate(connectors_6_9):
                conn_name = r['name']
                conn_num = int(conn_name[4:])
                roi = next((roi for roi in rois if roi.name == conn_name), None)
                if not roi or not hasattr(self, 'schematic_w') or not hasattr(self, 'schematic_h'):
                    continue
                
                # Tutte le immagini in una singola riga orizzontale
                n_bottom = len(connectors_6_9)
                
                # Calcola posizione Y nell'area inferiore (centrata verticalmente nell'area)
                zoom_y = bottom_area_y_bottom + bottom_area_height / 2 - zoom_height / 2
                
                # Calcola posizione X - distribuzione da destra a sinistra in una riga
                # Ordine: conn6, conn7, conn8, conn9 (da dx a sx significa conn6 a destra, conn9 a sinistra)
                if n_bottom == 1:
                    zoom_x = start_x + available_width / 2 - zoom_width / 2
                else:
                    # Distribuisci uniformemente da destra a sinistra con spaziatura molto aumentata
                    # Spaziatura tra le immagini (molto aumentata)
                    total_images_width = n_bottom * zoom_width
                    total_spacing = available_width - total_images_width
                    spacing_between = (total_spacing / (n_bottom + 1)) * 2.0 if n_bottom > 1 else 0  # Aumentata del 100%
                    # Posizione X: da destra a sinistra
                    # idx=0 (conn6) va a destra, idx=3 (conn9) va a sinistra
                    # Inverti l'indice per partire da destra
                    idx_reversed = n_bottom - 1 - idx
                    zoom_x = start_x + available_width - spacing_between - idx_reversed * (zoom_width + spacing_between) - zoom_width
                
                zoom_x = max(0.01, min(zoom_x, 0.99 - zoom_width))
                zoom_y = max(0.01, min(zoom_y, 0.99 - zoom_height))
                
                # Crea e renderizza immagine ingrandita (stesso codice di prima)
                self._render_zoomed_image(r, roi, zoom_x, zoom_y, zoom_width, zoom_height,
                                         schematic_ax, COLORS, rois, False)
                
        except Exception as e:
            print(f"Error rendering zoomed crops with arrows: {e}")
            import traceback
            traceback.print_exc()
    
    def _render_zoomed_image(self, r, roi, zoom_x, zoom_y, zoom_width, zoom_height,
                             schematic_ax, COLORS, rois, is_top):
        """Helper function per renderizzare una singola immagine ingrandita."""
        conn_name = r['name']
        
        # Crea subplot per immagine ingrandita
        zoom_ax = self.fig.add_axes([zoom_x, zoom_y, zoom_width, zoom_height])
        zoom_ax.set_facecolor('#ffffff')
        
        # Estrai il crop direttamente dall'immagine aligned corrente usando le coordinate ROI
        if self.aligned_image is None:
            return
        
        aligned_img = self.aligned_image
        height, width = aligned_img.shape[:2]
        
        # Converti coordinate relative a pixel
        x_min = int(roi.x_min_rel * width)
        y_min = int(roi.y_min_rel * height)
        x_max = int(roi.x_max_rel * width)
        y_max = int(roi.y_max_rel * height)
        
        # Aggiungi un margine per l'ingrandimento
        margin = 15
        x_min = max(0, x_min - margin)
        y_min = max(0, y_min - margin)
        x_max = min(width, x_max + margin)
        y_max = min(height, y_max + margin)
        
        # Estrai il crop dall'immagine aligned
        crop = aligned_img[y_min:y_max, x_min:x_max]
        
        # Prepara il crop per la visualizzazione
        if len(crop.shape) == 2:
            crop_display = crop
        else:
            crop_display = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        
        # Calcola aspect ratio e ridimensiona mantenendo proporzioni
        h, w = crop.shape[:2]
        aspect_ratio = w / h
        
        if aspect_ratio > 1:
            display_w = 1.0
            display_h = 1.0 / aspect_ratio
        else:
            display_h = 1.0
            display_w = 1.0 * aspect_ratio
        
        x_center = 0.5
        y_center = 0.5
        x_min_display = x_center - display_w / 2
        x_max_display = x_center + display_w / 2
        y_min_display = y_center - display_h / 2
        y_max_display = y_center + display_h / 2
        
        # Mostra immagine mantenendo proporzioni
        if len(crop.shape) == 2:
            zoom_ax.imshow(crop_display, cmap='gray', vmin=0, vmax=255, 
                         aspect='auto', interpolation='bilinear',
                         extent=[x_min_display, x_max_display, y_max_display, y_min_display])
        else:
            zoom_ax.imshow(crop_display, aspect='auto', interpolation='bilinear',
                         extent=[x_min_display, x_max_display, y_max_display, y_min_display])
        
        zoom_ax.axis('off')
        
        # Bordo colorato
        color_hex = COLORS[r['label']]
        border = Rectangle((0, 0), 1, 1, transform=zoom_ax.transAxes,
                         facecolor='none', edgecolor=color_hex,
                         linewidth=3, zorder=10)
        zoom_ax.add_patch(border)
        
        # Label sopra il riquadro (non dentro) - "Conn1 - KO"
        conn_name_upper = conn_name.upper()
        label_text = f"{conn_name_upper} - {r['label']}"
        # Posizione sopra il riquadro usando coordinate figura
        # Per area superiore: testo sopra il riquadro (verso l'alto, verso la board)
        # Per area inferiore: testo sotto il riquadro (verso il basso, verso la board)
        if is_top:
            label_y_fig = zoom_y + zoom_height + 0.015  # Sopra il riquadro per area superiore (verso la board)
            va_align = 'bottom'
        else:
            label_y_fig = zoom_y - 0.015  # Sotto il riquadro per area inferiore (verso la board)
            va_align = 'top'
        
        label_x_fig = zoom_x + zoom_width / 2  # Centrato sul riquadro
        self.fig.text(label_x_fig, label_y_fig, label_text,
                    fontsize=8, color=color_hex,
                    ha='center', va=va_align, weight='bold')
        
        # Posizione sulla board schematic (centro del connettore)
        schematic_x = (roi.x_min_rel + roi.x_max_rel) / 2 * self.schematic_w
        schematic_y = (roi.y_min_rel + roi.y_max_rel) / 2 * self.schematic_h
        
        # Posizione nel crop ingrandito - punto di arrivo della freccia
        # Per area superiore: arriva dal basso dell'immagine
        # Per area inferiore: arriva dall'alto dell'immagine
        if is_top:
            zoom_x_data = x_center
            zoom_y_data = y_min_display  # Parte inferiore dell'immagine
        else:
            zoom_x_data = x_center
            zoom_y_data = y_max_display  # Parte superiore dell'immagine
        
        # Calcola punto di arrivo della freccia considerando il testo
        # La freccia deve fermarsi prima del testo
        # Il testo è posizionato a label_y_fig in coordinate figura
        # Convertiamo in coordinate axes fraction per il punto di arrivo
        if is_top:
            # Per area superiore: freccia arriva dal basso dell'immagine, testo è sotto il riquadro
            # La freccia si ferma prima del testo (più in alto)
            # zoom_y è il bordo inferiore del riquadro, label_y_fig è sotto
            # Calcoliamo dove fermarsi: poco prima del bordo inferiore del riquadro
            arrow_end_y = zoom_y_data + 0.12  # Offset per fermarsi prima del testo (verso l'alto)
        else:
            # Per area inferiore: freccia arriva dall'alto dell'immagine, testo è sopra il riquadro
            # La freccia si ferma prima del testo (più in basso)
            # zoom_y + zoom_height è il bordo superiore del riquadro, label_y_fig è sopra
            # Calcoliamo dove fermarsi: poco prima del bordo superiore del riquadro
            arrow_end_y = zoom_y_data - 0.12  # Offset per fermarsi prima del testo (verso il basso)
        
        # Disegna freccia dalla board schematic all'immagine ingrandita
        arrow_color = color_hex
        arrow = ConnectionPatch(
            (schematic_x, schematic_y),
            (zoom_x_data, arrow_end_y),
            "data", "axes fraction",
            axesA=schematic_ax, axesB=zoom_ax,
            arrowstyle="->", shrinkA=5, shrinkB=5,
            mutation_scale=15, fc=arrow_color, ec=arrow_color,
            linewidth=2, zorder=100, alpha=0.8,
            connectionstyle="arc3,rad=0.1"  # Leggera curvatura
        )
        self.fig.patches.append(arrow)
    
    def update_statistics(self):
        """Aggiorna le statistiche."""
        ok_count = sum(1 for r in self.results if r['label'] == 'OK')
        ko_count = sum(1 for r in self.results if r['label'] == 'KO')
        occ_count = sum(1 for r in self.results if r['label'] == 'OCCLUSION')
        
        stats_text = f"Status Summary\n"
        stats_text += "-" * 30 + "\n"
        stats_text += f"OK        {ok_count}/9\n"
        stats_text += f"KO        {ko_count}/9\n"
        stats_text += f"OCCLUSION {occ_count}/9\n\n"
        
        if self.analysis_log:
            stats_text += "Recent Analyses\n"
            stats_text += "-" * 30 + "\n"
            for log_entry in self.analysis_log[-ANALYSIS_LOG_SIZE:]:
                filename_short = log_entry['filename'][:20] + "..." if len(log_entry['filename']) > 20 else log_entry['filename']
                stats_text += f"{log_entry['timestamp'][11:19]}\n"
                stats_text += f"  {filename_short}\n"
                stats_text += f"  KO:{log_entry['ko_count']} OCC:{log_entry.get('occ_count', 0)}\n\n"
        
        self.stats_text.config(state=tk.NORMAL)
        self.stats_text.delete(1.0, tk.END)
        self.stats_text.insert(1.0, stats_text)
        self.stats_text.config(state=tk.DISABLED)
    
    def init_simulation_pool(self):
        """Inizializza il pool di immagini per la simulazione."""
        self.simulation_images = []
        # Try package directory first, then fallback to parent
        raw_images_dir = self.package_root / "simulation_images"
        
        if raw_images_dir.exists():
            self.simulation_images.extend(list(raw_images_dir.glob("*.png")))
            self.simulation_images.extend(list(raw_images_dir.glob("*.jpg")))
        else:
            # Fallback: try parent directory (for backward compatibility)
            fallback_dir = self.package_root.parent / "Data" / "raw_images_no_occlusion"
            if fallback_dir.exists():
                self.simulation_images.extend(list(fallback_dir.glob("*.png")))
                self.simulation_images.extend(list(fallback_dir.glob("*.jpg")))
                print(f"📸 Using fallback simulation directory: {fallback_dir}")
            else:
                print(f"⚠️  Simulation images directory not found. Simulation feature will be disabled.")
                print(f"   Create 'simulation_images' folder in package directory to enable simulation.")
        
        print(f"📸 Simulation pool initialized: {len(self.simulation_images)} images from {raw_images_dir}")
    
    def toggle_mute(self):
        """Toggle mute sounds."""
        self.sounds_muted = self.mute_var.get()
    
    def play_alert_sound(self, sound_type):
        """Riproduce suono di alert."""
        if self.sounds_muted:
            return
        
        if sound_type == "KO":
            sound_path = self.package_root / KO_ALERT_SOUND_PATH
        elif sound_type == "OCCLUSION":
            sound_path = self.package_root / OCCLUSION_ALERT_SOUND_PATH
        else:
            return
        
        if PYGAME_AVAILABLE and sound_path.exists():
            try:
                pygame.mixer.music.load(str(sound_path))
                pygame.mixer.music.play()
            except:
                self.root.bell()
        elif sound_path.exists():
            try:
                import platform
                if platform.system() == "Darwin":
                    import subprocess
                    subprocess.Popen(["afplay", str(sound_path)])
                elif platform.system() == "Linux":
                    import subprocess
                    subprocess.Popen(["aplay", str(sound_path)])
                elif platform.system() == "Windows":
                    import winsound
                    winsound.PlaySound(str(sound_path), winsound.SND_FILENAME)
            except:
                self.root.bell()
        else:
            self.root.bell()
    
    def start_simulation(self):
        """Avvia la simulazione."""
        if not self.simulation_images:
            messagebox.showwarning("Warning", "No simulation images found.")
            return
        
        self.simulation_active = True
        self.start_sim_btn.config(state=tk.DISABLED)
        self.stop_sim_btn.config(state=tk.NORMAL)
        self.status_label.config(text="Simulation active", fg='#007bff')
        self.simulation_step()
    
    def stop_simulation(self):
        """Ferma la simulazione."""
        self.simulation_active = False
        self.start_sim_btn.config(state=tk.NORMAL)
        self.stop_sim_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Simulation stopped", fg='#6c757d')
    
    def add_ko_test(self):
        """Aggiunge un'immagine KO alla coda."""
        if not KO_TEST_IMAGES:
            messagebox.showwarning("Warning", "No KO test images configured.")
            return
        
        ko_image_rel = KO_TEST_IMAGES[self.ko_test_index]
        # Try package directory first, then fallback to parent
        ko_image_path = self.package_root / ko_image_rel
        if not ko_image_path.exists():
            ko_image_path = self.package_root.parent / ko_image_rel
        
        if not ko_image_path.exists():
            messagebox.showwarning("Warning", f"KO test image not found: {ko_image_path}")
            self.ko_test_index = (self.ko_test_index + 1) % len(KO_TEST_IMAGES)
            return
        
        self.ko_test_queue.append(str(ko_image_path))
        self.ko_test_index = (self.ko_test_index + 1) % len(KO_TEST_IMAGES)
        self.status_label.config(text=f"KO test queued ({len(self.ko_test_queue)} in queue)", fg='#ffc107')
    
    def simulation_step(self):
        """Esegue un passo della simulazione."""
        if not self.simulation_active:
            return
        
        if self.ko_test_queue:
            image_path = self.ko_test_queue.pop(0)
        else:
            if not self.simulation_images:
                self.stop_simulation()
                messagebox.showwarning("Warning", "No simulation images available.")
                return
            image_path = str(random.choice(self.simulation_images))
        
        self.process_image_from_path(image_path)
        
        if self.simulation_active:
            self.root.after(SIMULATION_INTERVAL_SECONDS * 1000, self.simulation_step)
    
    def process_image_from_path(self, image_path):
        """Processa un'immagine da un path."""
        self.image_path = image_path
        self.raw_image_path = image_path
        self.process_image()
    

def main():
    try:
        import tkinterdnd2
    except ImportError:
        print("⚠️  tkinterdnd2 non installato. Installazione...")
        import subprocess
        subprocess.check_call([sys.executable, "-m", "pip", "install", "tkinterdnd2"])
        import tkinterdnd2
    
    root = TkinterDnD.Tk()
    app = BekoPCLPresenceSystem(root)
    root.mainloop()

if __name__ == "__main__":
    main()

