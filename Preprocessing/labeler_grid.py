from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Dict, List, Optional

import cv2
import numpy as np
import pandas as pd
import tkinter as tk
from pathlib import Path

CONNECTOR_ORDER: List[str] = [
    "conn1",
    "conn2",
    "conn3",
    "conn4",
    "conn5",
    "conn6",
    "conn7",
    "conn8",
    "conn9",
]

LABEL_COLORS = {
    "OK": (0, 200, 0),
    "KO": (0, 0, 200),
    "OCCLUSION": (0, 165, 255),  # Arancione
    "PARTIAL OCCLUSION": (255, 165, 0),  # Arancione scuro/Giallo
    None: (120, 120, 120),
}


@dataclass
class BoardGroup:
    board_id: str
    rows: pd.DataFrame


def read_features_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"features.csv not found at {path}")
    df = pd.read_csv(path)
    required = {"sample_id", "connector_name", "filename"}
    if not required.issubset(df.columns):
        missing = ", ".join(sorted(required - set(df.columns)))
        raise ValueError(f"features.csv is missing required columns: {missing}")
    df["board_id"] = df["sample_id"].astype(str).str.split("_", n=1).str[0]
    if "label" not in df.columns:
        df["label"] = pd.NA
    return df


def validate_groups(df: pd.DataFrame) -> List[BoardGroup]:
    groups: List[BoardGroup] = []
    for board_id, group in df.groupby("board_id", sort=True):
        group = group.set_index("connector_name")
        missing = [c for c in CONNECTOR_ORDER if c not in group.index]
        if missing:
            raise ValueError(f"Board {board_id} is missing connectors: {missing}")
        ordered = group.loc[CONNECTOR_ORDER]
        groups.append(BoardGroup(board_id=board_id, rows=ordered))
    return groups


def load_image(root: Path, connector: str, filename: str) -> np.ndarray:
    image_path = root / connector / filename
    img = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if img is None:
        print(f"Warning: failed to load {image_path}. Using blank placeholder.")
        return np.zeros((200, 200, 3), dtype=np.uint8)
    return img


def build_grid_image(
    images: List[np.ndarray],
    labels: Dict[str, Optional[str]],
    active_index: int,
    tile_size: int = 220,
) -> np.ndarray:
    rows, cols = 3, 3
    canvas = np.zeros((rows * tile_size, cols * tile_size, 3), dtype=np.uint8)
    for idx, (connector, img) in enumerate(zip(CONNECTOR_ORDER, images)):
        r = idx // cols
        c = idx % cols
        y0, y1 = r * tile_size, (r + 1) * tile_size
        x0, x1 = c * tile_size, (c + 1) * tile_size
        resized = cv2.resize(img, (tile_size, tile_size), interpolation=cv2.INTER_AREA)
        canvas[y0:y1, x0:x1] = resized

        color = LABEL_COLORS[labels.get(connector)]
        cv2.rectangle(canvas, (x0 + 2, y0 + 2), (x1 - 3, y1 - 3), color, 4)
        text = connector.upper()
        cv2.putText(
            canvas,
            text,
            (x0 + 8, y0 + 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        label_value = labels.get(connector)
        if label_value:
            cv2.putText(
                canvas,
                label_value,
                (x0 + 8, y1 - 12),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                color,
                2,
                cv2.LINE_AA,
            )

    if 0 <= active_index < len(CONNECTOR_ORDER):
        r = active_index // cols
        c = active_index % cols
        y0, y1 = r * tile_size, (r + 1) * tile_size
        x0, x1 = c * tile_size, (c + 1) * tile_size
        cv2.rectangle(canvas, (x0 + 6, y0 + 6), (x1 - 7, y1 - 7), (0, 255, 255), 5)
    return canvas


class LabelerApp:
    def __init__(self, df: pd.DataFrame, groups: List[BoardGroup], image_root: Path):
        self.df = df
        self.groups = groups
        self.image_root = image_root
        self.board_index = 0
        self.connector_index = 0
        self.board_labels: Dict[str, Dict[str, Optional[str]]] = {
            board.board_id: {
                connector: (
                    str(board.rows.loc[connector, "label"]).upper()
                    if pd.notna(board.rows.loc[connector, "label"])
                    else None
                )
                for connector in CONNECTOR_ORDER
            }
            for board in groups
        }
        self.image_cache: Dict[str, List[np.ndarray]] = {}

        self.root = tk.Tk()
        self.root.title("Connector Labeler 3x3")
        self.photo: Optional[tk.PhotoImage] = None

        self.info_label = tk.Label(
            self.root,
            text="Premi O = OK, K = KO, C = OCCLUSION, P = PARTIAL OCCLUSION, B = indietro, Esc = esci",
            font=("Helvetica", 14),
        )
        self.info_label.pack(pady=6)

        self.canvas_label = tk.Label(self.root)
        self.canvas_label.pack()

        self.status_label = tk.Label(self.root, font=("Helvetica", 12))
        self.status_label.pack(pady=6)

        self.root.bind("<Key>", self.handle_keypress)
        self.root.protocol("WM_DELETE_WINDOW", self.handle_exit)

        self.refresh_view()

    def run(self) -> None:
        self.root.mainloop()

    def handle_keypress(self, event: tk.Event) -> None:
        key = event.keysym.lower()
        if key == "o":
            self.assign_label("OK")
        elif key == "k":
            self.assign_label("KO")
        elif key == "c":
            self.assign_label("OCCLUSION")
        elif key == "p":
            self.assign_label("PARTIAL OCCLUSION")
        elif key == "b":
            self.move_previous()
        elif key == "escape":
            self.handle_exit()

    def move_previous(self) -> None:
        if self.connector_index > 0:
            # Vai al connettore precedente nella stessa board
            self.connector_index -= 1
            self.refresh_view()
        elif self.board_index > 0:
            # Se siamo al primo connettore, vai alla board precedente
            self.board_index -= 1
            self.connector_index = len(CONNECTOR_ORDER) - 1  # Ultimo connettore della board precedente
            self.refresh_view()

    def assign_label(self, label: str) -> None:
        board = self.groups[self.board_index]
        connector = CONNECTOR_ORDER[self.connector_index]
        self.board_labels[board.board_id][connector] = label
        if self.connector_index < len(CONNECTOR_ORDER) - 1:
            self.connector_index += 1
        else:
            if self.is_board_complete(board.board_id):
                self.commit_board(board.board_id)
                if self.board_index == len(self.groups) - 1:
                    print("All boards labeled.")
                    self.persist_and_quit()
                    return
                self.board_index += 1
                self.connector_index = 0
            else:
                print("Completa tutte le caselle prima di avanzare.")
        self.refresh_view()

    def is_board_complete(self, board_id: str) -> bool:
        labels = self.board_labels[board_id]
        return all(label in {"OK", "KO", "OCCLUSION", "PARTIAL OCCLUSION"} for label in labels.values())

    def commit_board(self, board_id: str) -> None:
        labels = self.board_labels[board_id]
        for connector, value in labels.items():
            mask = (self.df["board_id"] == board_id) & (
                self.df["connector_name"] == connector
            )
            self.df.loc[mask, "label"] = value
        print(f"Board {board_id} labeled.")

    def refresh_view(self) -> None:
        board = self.groups[self.board_index]
        labels = self.board_labels[board.board_id]
        images = self.get_board_images(board)
        grid = build_grid_image(images, labels, self.connector_index)
        rgb = cv2.cvtColor(grid, cv2.COLOR_BGR2RGB)
        success, buffer = cv2.imencode(".png", rgb)
        if not success:
            raise RuntimeError("Failed to encode image for display.")
        data_b64 = base64.b64encode(buffer).decode("ascii")
        self.photo = tk.PhotoImage(data=data_b64)
        self.canvas_label.configure(image=self.photo)

        status = (
            f"Board {board.board_id} ({self.board_index + 1}/{len(self.groups)}) - "
            f"Connettore {CONNECTOR_ORDER[self.connector_index].upper()} "
            f"[O=OK, K=KO, C=OCCLUSION, P=PARTIAL OCCLUSION, B=indietro]"
        )
        self.status_label.configure(text=status)

    def get_board_images(self, board: BoardGroup) -> List[np.ndarray]:
        if board.board_id not in self.image_cache:
            self.image_cache[board.board_id] = [
                load_image(
                    self.image_root,
                    connector,
                    str(board.rows.loc[connector, "filename"]),
                )
                for connector in CONNECTOR_ORDER
            ]
        return self.image_cache[board.board_id]

    def handle_exit(self) -> None:
        print("Interruzione manuale. Salvo il progresso attuale...")
        self.persist_and_quit()

    def persist_and_quit(self) -> None:
        output_path = Path.cwd() / "features_labeled.csv"
        self.df.to_csv(output_path, index=False)
        print(f"Saved labeled features to {output_path}")
        self.root.destroy()


def main() -> None:
    project_root = Path.cwd()
    features_path = project_root / "features.csv"
    image_root = project_root / "Data" / "connectors"

    df = read_features_csv(features_path)
    groups = validate_groups(df)
    print(f"Loaded {len(df)} rows across {len(groups)} boards.")

    app = LabelerApp(df, groups, image_root)
    app.run()


if __name__ == "__main__":
    main()