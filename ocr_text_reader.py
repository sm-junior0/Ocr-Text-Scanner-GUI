"""
GUI-based OCR tool with:
- Image load and camera capture
- ROI (click + drag) selection
- Run OCR button with overlay preview of detected text
- Live camera input with start/stop controls
- Text display of extracted content
"""

import cv2
import pytesseract
from pytesseract import Output, TesseractNotFoundError
import numpy as np
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from PIL import Image, ImageTk
import os
import sys


def find_tesseract_path():
    """Try to find Tesseract executable on Windows."""
    if sys.platform.startswith('win'):
        # Common installation paths on Windows
        possible_paths = [
            r"C:\Program Files\Tesseract-OCR\tesseract.exe",
            r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
            r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(os.getenv('USERNAME', '')),
        ]
        
        for path in possible_paths:
            if os.path.isfile(path):
                return path
    
    # If not found or not Windows, return None (will use PATH)
    return None


# Configure Tesseract path if found
tesseract_path = find_tesseract_path()
if tesseract_path:
    pytesseract.pytesseract.tesseract_cmd = tesseract_path


class OCRApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Text Reader - OCR with ROI and Camera")

        # Canvas/display settings
        self.canvas_width = 900
        self.canvas_height = 600

        # State
        self.current_image = None  # OpenCV BGR image currently displayed/used
        self.video_capture = None
        self.running_camera = False
        self.pause_live_preview = False
        self.display_image_meta = None  # For coordinate transforms
        self.roi_start = None
        self.roi_box = None  # (x1, y1, x2, y2) in image coords
        self.photo_image = None  # Keep reference to avoid GC

        self._build_ui()
        self._check_tesseract()
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _build_ui(self) -> None:
        control_frame = tk.Frame(self.root)
        control_frame.pack(fill=tk.X, padx=8, pady=4)

        tk.Button(control_frame, text="Load Image", command=self.load_image).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(
            control_frame, text="Start/Resume Camera", command=self.start_camera
        ).pack(side=tk.LEFT, padx=4)
        tk.Button(control_frame, text="Stop Camera", command=self.stop_camera).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(control_frame, text="Run OCR", command=self.run_ocr).pack(
            side=tk.LEFT, padx=4
        )
        tk.Button(control_frame, text="Clear ROI", command=self.clear_roi).pack(
            side=tk.LEFT, padx=4
        )

        self.status_var = tk.StringVar(value="Load an image or start the camera.")
        tk.Label(control_frame, textvariable=self.status_var, fg="blue").pack(
            side=tk.LEFT, padx=10
        )

        # Canvas for image/overlay + ROI drawing
        self.canvas = tk.Canvas(
            self.root,
            width=self.canvas_width,
            height=self.canvas_height,
            bg="black",
            cursor="crosshair",
        )
        self.canvas.pack(padx=8, pady=4)
        self.canvas.bind("<ButtonPress-1>", self.on_mouse_press)
        self.canvas.bind("<B1-Motion>", self.on_mouse_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_mouse_release)

        # Text output area
        tk.Label(self.root, text="Extracted Text").pack(anchor="w", padx=8)
        self.text_output = scrolledtext.ScrolledText(self.root, height=10, wrap=tk.WORD)
        self.text_output.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)

    def _check_tesseract(self) -> None:
        """Check if Tesseract is available and show warning if not."""
        try:
            # Try to get Tesseract version to verify it's working
            pytesseract.get_tesseract_version()
        except TesseractNotFoundError:
            error_msg = (
                "Tesseract OCR is not installed or not found in PATH.\n\n"
                "To install Tesseract on Windows:\n"
                "1. Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "2. Install the .exe file\n"
                "3. Restart this application\n\n"
                "Common installation path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe\n\n"
                "The application will still run, but OCR features will not work until Tesseract is installed."
            )
            messagebox.showwarning("Tesseract Not Found", error_msg)
            self.status_var.set("Warning: Tesseract OCR not found. Install Tesseract to use OCR features.")

    # ----------------- Utility functions -----------------
    def _cv_to_tk_image(self, cv_img: np.ndarray):
        rgb = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
        img_h, img_w = pil_img.height, pil_img.width

        scale = min(self.canvas_width / img_w, self.canvas_height / img_h)
        new_w = int(img_w * scale)
        new_h = int(img_h * scale)
        resized = pil_img.resize((new_w, new_h), Image.LANCZOS)

        offset_x = (self.canvas_width - new_w) // 2
        offset_y = (self.canvas_height - new_h) // 2

        self.display_image_meta = {
            "scale": scale,
            "offset_x": offset_x,
            "offset_y": offset_y,
            "img_w": img_w,
            "img_h": img_h,
        }
        self.photo_image = ImageTk.PhotoImage(resized)
        return self.photo_image, offset_x, offset_y

    def _canvas_to_image_coords(self, x: float, y: float):
        if not self.display_image_meta:
            return None
        m = self.display_image_meta
        img_x = int((x - m["offset_x"]) / m["scale"])
        img_y = int((y - m["offset_y"]) / m["scale"])
        if 0 <= img_x < m["img_w"] and 0 <= img_y < m["img_h"]:
            return img_x, img_y
        return None

    def _image_to_canvas_coords(self, x: float, y: float):
        m = self.display_image_meta
        canvas_x = x * m["scale"] + m["offset_x"]
        canvas_y = y * m["scale"] + m["offset_y"]
        return canvas_x, canvas_y

    def display_image(self, cv_img: np.ndarray) -> None:
        if cv_img is None:
            return
        tk_img, offset_x, offset_y = self._cv_to_tk_image(cv_img)
        self.canvas.delete("all")
        self.canvas.create_image(offset_x, offset_y, image=tk_img, anchor=tk.NW, tags="img")
        self.draw_roi_box()

    def draw_roi_box(self) -> None:
        self.canvas.delete("roi")
        if not self.roi_box or not self.display_image_meta:
            return
        x1, y1, x2, y2 = self.roi_box
        c1 = self._image_to_canvas_coords(x1, y1)
        c2 = self._image_to_canvas_coords(x2, y2)
        self.canvas.create_rectangle(*c1, *c2, outline="yellow", width=2, tags="roi")

    # ----------------- Event handlers -----------------
    def on_mouse_press(self, event) -> None:
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords:
            self.roi_start = coords
            self.roi_box = None
            self.draw_roi_box()

    def on_mouse_drag(self, event) -> None:
        if not self.roi_start:
            return
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords:
            x0, y0 = self.roi_start
            x1, y1 = coords
            self.roi_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
            self.draw_roi_box()

    def on_mouse_release(self, event) -> None:
        if not self.roi_start:
            return
        coords = self._canvas_to_image_coords(event.x, event.y)
        if coords:
            x0, y0 = self.roi_start
            x1, y1 = coords
            self.roi_box = (min(x0, x1), min(y0, y1), max(x0, x1), max(y0, y1))
        self.roi_start = None
        self.draw_roi_box()

    # ----------------- Camera handling -----------------
    def start_camera(self) -> None:
        if self.video_capture is None:
            # Try different camera indices (0 is most common default)
            camera_found = False
            for camera_index in [0, 1, 2]:
                self.video_capture = cv2.VideoCapture(camera_index)
                if self.video_capture.isOpened():
                    # Test if we can actually read a frame
                    ret, test_frame = self.video_capture.read()
                    if ret and test_frame is not None:
                        camera_found = True
                        break
                    else:
                        self.video_capture.release()
                        self.video_capture = None
                else:
                    if self.video_capture is not None:
                        self.video_capture.release()
                        self.video_capture = None
            
            if not camera_found:
                messagebox.showerror("Camera Error", "Cannot open camera. Please check if a camera is connected.")
                return
        self.running_camera = True
        self.pause_live_preview = False
        self.status_var.set("Camera running. Drag to set ROI or click Run OCR.")
        self.update_camera_frame()

    def stop_camera(self) -> None:
        self.running_camera = False
        self.pause_live_preview = False
        if self.video_capture is not None:
            self.video_capture.release()
            self.video_capture = None
        self.status_var.set("Camera stopped.")

    def update_camera_frame(self) -> None:
        if not self.running_camera or self.pause_live_preview:
            return
        if self.video_capture is None:
            return
        ret, frame = self.video_capture.read()
        if not ret:
            self.status_var.set("Failed to read from camera.")
            return
        self.current_image = frame
        self.display_image(frame)
        # Schedule next frame
        self.root.after(30, self.update_camera_frame)

    # ----------------- OCR handling -----------------
    def _preprocess_for_ocr(self, img: np.ndarray) -> np.ndarray:
        """Lightweight preprocessing to boost OCR quality."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Reduce noise and improve contrast
        gray = cv2.bilateralFilter(gray, d=5, sigmaColor=50, sigmaSpace=50)
        # Adaptive threshold to handle uneven lighting
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            10,
        )
        return binary

    def _prepare_roi(self):
        if self.current_image is None:
            return None, None
        h, w = self.current_image.shape[:2]
        if not self.roi_box:
            return self.current_image.copy(), (0, 0)
        x1, y1, x2, y2 = self.roi_box
        x1 = max(0, min(w - 1, int(x1)))
        x2 = max(0, min(w, int(x2)))
        y1 = max(0, min(h - 1, int(y1)))
        y2 = max(0, min(h, int(y2)))
        if x2 <= x1 or y2 <= y1:
            return self.current_image.copy(), (0, 0)
        roi_img = self.current_image[y1:y2, x1:x2].copy()
        return roi_img, (x1, y1)

    def run_ocr(self) -> None:
        if self.current_image is None:
            messagebox.showwarning("No Image", "Load an image or start the camera first.")
            return

        # Freeze live preview to keep overlay visible
        if self.running_camera:
            self.pause_live_preview = True

        roi_img, (offset_x, offset_y) = self._prepare_roi()
        if roi_img is None:
            messagebox.showwarning("No Image", "Load an image or start the camera first.")
            return

        # Preprocess to improve OCR quality
        processed = self._preprocess_for_ocr(roi_img)

        # Run OCR with error handling
        ocr_config = "--psm 6"
        try:
            ocr_text = pytesseract.image_to_string(processed, config=ocr_config)
            data = pytesseract.image_to_data(processed, output_type=Output.DICT, config=ocr_config)
        except TesseractNotFoundError:
            error_msg = (
                "Tesseract OCR is not installed or not found in PATH.\n\n"
                "To install Tesseract on Windows:\n"
                "1. Download from: https://github.com/UB-Mannheim/tesseract/wiki\n"
                "2. Install the .exe file\n"
                "3. Add Tesseract to your PATH, or restart this application\n\n"
                "Common installation path: C:\\Program Files\\Tesseract-OCR\\tesseract.exe"
            )
            messagebox.showerror("Tesseract Not Found", error_msg)
            self.status_var.set("Tesseract OCR not found. Please install Tesseract.")
            return
        except Exception as e:
            error_msg = f"OCR Error: {str(e)}\n\nPlease check your Tesseract installation."
            messagebox.showerror("OCR Error", error_msg)
            self.status_var.set(f"OCR failed: {str(e)}")
            return

        annotated = self.current_image.copy()
        n = len(data["text"])
        for i in range(n):
            text = data["text"][i].strip()
            conf = float(data["conf"][i]) if data["conf"][i] != "-1" else -1.0
            if not text or conf < 10:
                continue
            x = data["left"][i] + offset_x
            y = data["top"][i] + offset_y
            w = data["width"][i]
            h = data["height"][i]
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated,
                text,
                (x, max(0, y - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 255),
                1,
                cv2.LINE_AA,
            )

        self.text_output.delete("1.0", tk.END)
        stripped = ocr_text.strip()
        if stripped:
            self.text_output.insert(tk.END, stripped)
            self.status_var.set("OCR complete. Overlay shows detected text. Resume camera if needed.")
        else:
            self.text_output.insert(tk.END, "[No text detected]")
            self.status_var.set("No text detected. Try a larger ROI, better lighting, or move closer.")
        self.display_image(annotated)

    # ----------------- Misc helpers -----------------
    def clear_roi(self) -> None:
        self.roi_box = None
        self.draw_roi_box()
        self.status_var.set("ROI cleared.")

    def load_image(self) -> None:
        file_path = filedialog.askopenfilename(
            title="Select an image",
            filetypes=[
                ("Image files", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff"),
                ("All files", "*.*"),
            ],
        )
        if not file_path:
            return
        img = cv2.imread(file_path)
        if img is None:
            messagebox.showerror("Load Error", "Could not load the selected image.")
            return
        self.current_image = img
        self.pause_live_preview = False
        self.running_camera = False
        self.display_image(img)
        self.status_var.set("Image loaded. Select ROI or run OCR.")

    def on_close(self) -> None:
        self.stop_camera()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    app = OCRApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()