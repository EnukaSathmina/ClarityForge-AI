import os
import sys
import traceback
import ctypes
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QIcon, QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSlider,
    QSpacerItem,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)


APP_NAME = "ClarityForge AI"
IS_FROZEN = getattr(sys, "frozen", False)
APP_DIR = Path(sys.executable).resolve().parent if IS_FROZEN else Path(__file__).resolve().parent
RESOURCE_DIR = Path(getattr(sys, "_MEIPASS", APP_DIR))
ROOT_DIR = APP_DIR
MODELS_DIR = RESOURCE_DIR / "models"
OUTPUT_DIR = APP_DIR / "output"
ASSETS_DIR = RESOURCE_DIR / "assets"
REALESRGAN_X4PLUS = "RealESRGAN_x4plus.pth"
REALESRGAN_GENERAL_X4V3 = "realesr-general-x4v3.pth"


@dataclass
class ProcessingOptions:
    model_name: str
    scale_mode: str
    face_enhance: bool
    denoise: bool
    safe_restore: bool
    sharpness: int
    contrast: int
    color_boost: int


def ensure_project_dirs() -> None:
    for folder in (OUTPUT_DIR,):
        folder.mkdir(exist_ok=True)


def app_icon() -> QIcon:
    for icon_path in (APP_DIR / "icon.ico", RESOURCE_DIR / "icon.ico", ASSETS_DIR / "icon.ico"):
        if icon_path.exists():
            return QIcon(str(icon_path))
    return QIcon()


def configure_windows_taskbar_icon() -> None:
    if sys.platform != "win32":
        return

    try:
        app_id = "ClarityForge.ClarityForgeAI.Desktop.1"
        ctypes.windll.shell32.SetCurrentProcessExplicitAppUserModelID(app_id)
    except Exception:
        pass


def detect_device() -> tuple[str, str]:
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda", f"GPU: {torch.cuda.get_device_name(0)}"
        return "cpu", "CPU mode"
    except Exception:
        return "cpu", "CPU mode"


def pil_to_qpixmap(image: Image.Image, max_width: int = 760, max_height: int = 540) -> QPixmap:
    preview = image.copy()
    preview.thumbnail((max_width, max_height), Image.Resampling.LANCZOS)
    rgba = preview.convert("RGBA")
    data = rgba.tobytes("raw", "RGBA")
    qimage = QImage(data, rgba.width, rgba.height, QImage.Format.Format_RGBA8888)
    return QPixmap.fromImage(qimage)


def cv_to_pil(image: np.ndarray) -> Image.Image:
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)


def pil_to_cv(image: Image.Image) -> np.ndarray:
    rgb = np.array(image.convert("RGB"))
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


class ImageProcessor(QThread):
    progress = Signal(int, str)
    finished = Signal(object, str)
    failed = Signal(str)

    def __init__(self, image_path: Path, options: ProcessingOptions):
        super().__init__()
        self.image_path = image_path
        self.options = options
        self.device, _ = detect_device()

    def run(self) -> None:
        try:
            self.progress.emit(5, "Loading image")
            image = Image.open(self.image_path).convert("RGB")
            face_detected = self.detect_face(image)

            self.progress.emit(20, "Running Real-ESRGAN AI upscaling")
            enhanced = self.upscale(image)

            if self.options.safe_restore or (not face_detected and self.has_dark_silhouette(image)):
                self.progress.emit(48, "Preserving silhouette shapes")
                enhanced = self.preserve_dark_silhouettes(image, enhanced)

            if self.options.face_enhance and not self.options.safe_restore and face_detected:
                self.progress.emit(55, "Restoring faces")
                enhanced = self.restore_faces(enhanced)
            elif self.options.face_enhance and not face_detected:
                self.progress.emit(55, "No face detected; skipping face restoration")
            elif self.options.safe_restore:
                self.progress.emit(55, "Safe Mode active; skipping face restoration")

            if self.options.denoise:
                self.progress.emit(70, "Reducing noise")
                enhanced = self.denoise(enhanced)

            self.progress.emit(82, "Applying finishing adjustments")
            enhanced = self.apply_adjustments(enhanced)
            enhanced = self.resize_after_upscale(image, enhanced)

            self.progress.emit(95, "Finalizing preview")
            self.finished.emit(enhanced, "Processing complete")
        except Exception as exc:
            details = "".join(traceback.format_exception_only(type(exc), exc)).strip()
            self.failed.emit(details)

    def upscale(self, image: Image.Image) -> Image.Image:
        upsampler = self.load_realesrgan()
        output, _ = upsampler.enhance(pil_to_cv(image), outscale=4)
        return cv_to_pil(output)

    def load_realesrgan(self):
        try:
            from realesrgan import RealESRGANer
        except ImportError as exc:
            raise RuntimeError(
                "Real-ESRGAN is not installed. Install dependencies with:\n"
                "pip install -r requirements.txt"
            ) from exc

        model_path, model = self.realesrgan_model_config()
        return RealESRGANer(
            scale=4,
            model_path=str(model_path),
            model=model,
            tile=256,
            tile_pad=10,
            pre_pad=0,
            half=self.device == "cuda",
            device=self.device,
        )

    def realesrgan_model_config(self):
        x4plus_path = MODELS_DIR / REALESRGAN_X4PLUS
        general_path = MODELS_DIR / REALESRGAN_GENERAL_X4V3

        if self.options.model_name == "RealESRGAN_x4plus":
            if not x4plus_path.exists():
                raise FileNotFoundError(
                    "Selected model weights were not found.\n\n"
                    f"Expected file:\n{x4plus_path}\n\n"
                    "Download it with:\npy download_models.py x4plus"
                )
            try:
                from basicsr.archs.rrdbnet_arch import RRDBNet
            except ImportError as exc:
                raise RuntimeError(
                    "RealESRGAN_x4plus requires basicsr. Install dependencies with:\n"
                    "pip install -r requirements.txt"
                ) from exc

            model = RRDBNet(
                num_in_ch=3,
                num_out_ch=3,
                num_feat=64,
                num_block=23,
                num_grow_ch=32,
                scale=4,
            )
            return x4plus_path, model

        if not general_path.exists():
            raise FileNotFoundError(
                "Selected model weights were not found.\n\n"
                f"Expected file:\n{general_path}\n\n"
                "Download it with:\npy download_models.py general"
            )

        try:
            from realesrgan.archs.srvgg_arch import SRVGGNetCompact
        except ImportError as exc:
            raise RuntimeError(
                "realesr-general-x4v3 requires Real-ESRGAN's SRVGG architecture. "
                "Install dependencies with:\n"
                "pip install -r requirements.txt"
            ) from exc

        model = SRVGGNetCompact(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=64,
            num_conv=32,
            upscale=4,
            act_type="prelu",
        )
        return general_path, model

    def target_size(self, original: Image.Image, upscaled: Image.Image) -> tuple[int, int]:
        if self.options.safe_restore or self.options.scale_mode == "2x":
            return max(1, original.width * 2), max(1, original.height * 2)
        if self.options.scale_mode == "4x":
            return upscaled.width, upscaled.height

        long_edge = max(upscaled.width, upscaled.height)
        if long_edge <= 0:
            return upscaled.width, upscaled.height

        ratio = 3840 / long_edge
        return max(1, int(upscaled.width * ratio)), max(1, int(upscaled.height * ratio))

    def resize_after_upscale(self, original: Image.Image, upscaled: Image.Image) -> Image.Image:
        target = self.target_size(original, upscaled)
        if target == upscaled.size:
            return upscaled
        return upscaled.resize(target, Image.Resampling.LANCZOS)

    def preserve_dark_silhouettes(self, original: Image.Image, enhanced: Image.Image) -> Image.Image:
        original_cv = pil_to_cv(original)
        enhanced_cv = pil_to_cv(enhanced)

        gray = cv2.cvtColor(original_cv, cv2.COLOR_BGR2GRAY)
        dark_mask = cv2.inRange(gray, 0, 58)
        dark_ratio = float(np.count_nonzero(dark_mask)) / dark_mask.size

        if dark_ratio < 0.08:
            return enhanced

        mask = cv2.resize(dark_mask, (enhanced_cv.shape[1], enhanced_cv.shape[0]), interpolation=cv2.INTER_LINEAR)
        mask = cv2.GaussianBlur(mask, (0, 0), 1.5).astype(np.float32) / 255.0
        mask = mask[:, :, None] * 0.72

        reference = cv2.resize(original_cv, (enhanced_cv.shape[1], enhanced_cv.shape[0]), interpolation=cv2.INTER_CUBIC)
        blended = enhanced_cv.astype(np.float32) * (1.0 - mask) + reference.astype(np.float32) * mask
        return cv_to_pil(np.clip(blended, 0, 255).astype(np.uint8))

    def has_dark_silhouette(self, image: Image.Image) -> bool:
        gray = cv2.cvtColor(pil_to_cv(image), cv2.COLOR_BGR2GRAY)
        dark_mask = cv2.inRange(gray, 0, 58)
        dark_ratio = float(np.count_nonzero(dark_mask)) / dark_mask.size
        return dark_ratio >= 0.12

    def detect_face(self, image: Image.Image) -> bool:
        cv_image = pil_to_cv(image)
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"

        if not cascade_path.exists():
            return False

        detector = cv2.CascadeClassifier(str(cascade_path))
        min_side = max(24, min(image.width, image.height) // 12)
        faces = detector.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(min_side, min_side),
        )
        for x, y, width, height in faces:
            roi = gray[y : y + height, x : x + width]
            if roi.size == 0:
                continue

            brightness = float(np.mean(roi))
            texture = float(np.std(roi))
            if 35 <= brightness <= 225 and texture >= 18:
                return True

        return False

    def restore_faces(self, image: Image.Image) -> Image.Image:
        model_path = MODELS_DIR / "GFPGANv1.4.pth"

        try:
            from gfpgan import GFPGANer

            if not model_path.exists():
                raise FileNotFoundError(f"GFPGAN weights missing: {model_path}")

            restorer = GFPGANer(
                model_path=str(model_path),
                upscale=1,
                arch="clean",
                channel_multiplier=2,
                bg_upsampler=None,
                device=self.device,
            )
            _, _, restored = restorer.enhance(
                pil_to_cv(image),
                has_aligned=False,
                only_center_face=False,
                paste_back=True,
            )
            return cv_to_pil(restored)
        except Exception:
            return image

    def denoise(self, image: Image.Image) -> Image.Image:
        cv_image = pil_to_cv(image)
        strength = 3 if self.options.safe_restore else 6
        denoised = cv2.fastNlMeansDenoisingColored(cv_image, None, strength, strength, 7, 21)
        return cv_to_pil(denoised)

    def apply_adjustments(self, image: Image.Image) -> Image.Image:
        cv_image = pil_to_cv(image)
        sharpness = min(self.options.sharpness, 8) if self.options.safe_restore else self.options.sharpness
        contrast = min(self.options.contrast, 4) if self.options.safe_restore else self.options.contrast
        color_boost = min(self.options.color_boost, 0) if self.options.safe_restore else self.options.color_boost

        if sharpness != 0:
            amount = sharpness / 100.0
            blurred = cv2.GaussianBlur(cv_image, (0, 0), 1.2)
            cv_image = cv2.addWeighted(cv_image, 1.0 + amount, blurred, -amount, 0)

        if contrast != 0:
            alpha = 1.0 + contrast / 100.0
            cv_image = cv2.convertScaleAbs(cv_image, alpha=alpha, beta=0)

        if color_boost != 0:
            hsv = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV).astype(np.float32)
            hsv[:, :, 1] *= 1.0 + color_boost / 100.0
            hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
            cv_image = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

        return cv_to_pil(cv_image)


class PreviewPanel(QFrame):
    def __init__(self, title: str):
        super().__init__()
        self.setObjectName("previewCard")
        self.image_label = QLabel("No image loaded")
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setObjectName("previewImage")
        self.image_label.setMinimumSize(360, 300)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        title_label = QLabel(title)
        title_label.setObjectName("panelTitle")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(14)
        layout.addWidget(title_label)
        layout.addWidget(self.image_label, 1)

    def set_image(self, image: Image.Image) -> None:
        self.image_label.setPixmap(pil_to_qpixmap(image))

    def clear(self) -> None:
        self.image_label.setPixmap(QPixmap())
        self.image_label.setText("No image loaded")


class ControlSlider(QWidget):
    def __init__(self, title: str, minimum: int, maximum: int, value: int):
        super().__init__()
        self.value_label = QLabel(f"{value:+d}")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(minimum, maximum)
        self.slider.setValue(value)
        self.slider.valueChanged.connect(lambda val: self.value_label.setText(f"{val:+d}"))

        label = QLabel(title)
        label.setObjectName("controlLabel")
        self.value_label.setObjectName("valuePill")

        top = QHBoxLayout()
        top.addWidget(label)
        top.addStretch(1)
        top.addWidget(self.value_label)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        layout.addLayout(top)
        layout.addWidget(self.slider)

    def value(self) -> int:
        return self.slider.value()


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        ensure_project_dirs()
        self.setWindowTitle(APP_NAME)
        self.setWindowIcon(app_icon())
        self.resize(1320, 840)

        self.image_path: Path | None = None
        self.original_image: Image.Image | None = None
        self.processed_image: Image.Image | None = None
        self.processor: ImageProcessor | None = None

        self.device_key, device_label = detect_device()
        self.device_label = QLabel(device_label)
        self.device_label.setObjectName("deviceBadge")

        self.before_panel = PreviewPanel("Before")
        self.after_panel = PreviewPanel("After")

        self.upload_button = QPushButton("Upload Image")
        self.process_button = QPushButton("Enhance Image")
        self.export_png_button = QPushButton("Export PNG")
        self.export_jpg_button = QPushButton("Export JPG")

        self.model_combo = QComboBox()
        self.model_combo.addItems(["realesr-general-x4v3", "RealESRGAN_x4plus"])

        self.scale_combo = QComboBox()
        self.scale_combo.addItems(["2x", "4x", "4K target"])

        self.face_toggle = QCheckBox("Face enhancement")
        self.denoise_toggle = QCheckBox("Denoise")
        self.safe_restore_toggle = QCheckBox("Safe Mode")
        self.warning_label = QLabel(
            "Very low-resolution images may produce AI-generated artifacts because missing details must be guessed."
        )
        self.warning_label.setObjectName("warningText")
        self.warning_label.setWordWrap(True)
        self.face_toggle.setChecked(False)

        self.sharpness_slider = ControlSlider("Sharpness", -50, 100, 6)
        self.contrast_slider = ControlSlider("Contrast", -50, 100, 3)
        self.color_slider = ControlSlider("Color boost", -50, 100, 2)

        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        self.status_label = QLabel("Ready")
        self.status_label.setObjectName("statusText")

        self.build_ui()
        self.connect_signals()
        self.update_action_state()

    def build_ui(self) -> None:
        central = QWidget()
        central.setObjectName("appRoot")
        self.setCentralWidget(central)

        header_title = QLabel(APP_NAME)
        header_title.setObjectName("appTitle")
        header_subtitle = QLabel("AI upscaling, restoration, and finishing for low-quality images")
        header_subtitle.setObjectName("appSubtitle")

        header_text = QVBoxLayout()
        header_text.addWidget(header_title)
        header_text.addWidget(header_subtitle)

        header = QHBoxLayout()
        header.addLayout(header_text)
        header.addStretch(1)
        header.addWidget(self.device_label)

        preview_grid = QGridLayout()
        preview_grid.setSpacing(18)
        preview_grid.addWidget(self.before_panel, 0, 0)
        preview_grid.addWidget(self.after_panel, 0, 1)
        preview_grid.setColumnStretch(0, 1)
        preview_grid.setColumnStretch(1, 1)

        controls = self.build_controls()

        content = QHBoxLayout()
        preview_wrap = QVBoxLayout()
        preview_wrap.addLayout(preview_grid, 1)
        preview_wrap.addWidget(self.progress)
        preview_wrap.addWidget(self.status_label)
        content.addLayout(preview_wrap, 1)
        content.addWidget(controls)

        root = QVBoxLayout(central)
        root.setContentsMargins(24, 22, 24, 24)
        root.setSpacing(22)
        root.addLayout(header)
        root.addLayout(content, 1)

    def build_controls(self) -> QFrame:
        card = QFrame()
        card.setObjectName("controlCard")
        card.setFixedWidth(330)

        title = QLabel("Enhancement Studio")
        title.setObjectName("sectionTitle")

        model_label = QLabel("Model")
        model_label.setObjectName("controlLabel")

        scale_label = QLabel("Upscale mode")
        scale_label.setObjectName("controlLabel")

        layout = QVBoxLayout(card)
        layout.setContentsMargins(20, 20, 20, 20)
        layout.setSpacing(16)
        layout.addWidget(title)
        layout.addWidget(self.upload_button)
        layout.addSpacing(6)
        layout.addWidget(model_label)
        layout.addWidget(self.model_combo)
        layout.addWidget(scale_label)
        layout.addWidget(self.scale_combo)
        layout.addWidget(self.warning_label)
        layout.addWidget(self.safe_restore_toggle)
        layout.addWidget(self.face_toggle)
        layout.addWidget(self.denoise_toggle)
        layout.addWidget(self.sharpness_slider)
        layout.addWidget(self.contrast_slider)
        layout.addWidget(self.color_slider)
        layout.addSpacerItem(QSpacerItem(20, 20, QSizePolicy.Policy.Minimum, QSizePolicy.Policy.Expanding))
        layout.addWidget(self.process_button)
        layout.addWidget(self.export_png_button)
        layout.addWidget(self.export_jpg_button)
        return card

    def connect_signals(self) -> None:
        self.upload_button.clicked.connect(self.load_image)
        self.process_button.clicked.connect(self.process_image)
        self.export_png_button.clicked.connect(lambda: self.export_image("PNG"))
        self.export_jpg_button.clicked.connect(lambda: self.export_image("JPG"))
        self.safe_restore_toggle.toggled.connect(self.on_safe_restore_toggled)
        self.scale_combo.currentTextChanged.connect(self.on_scale_mode_changed)

    def load_image(self) -> None:
        file_name, _ = QFileDialog.getOpenFileName(
            self,
            "Choose an image",
            str(ROOT_DIR),
            "Images (*.png *.jpg *.jpeg *.bmp *.webp *.tif *.tiff)",
        )
        if not file_name:
            return

        try:
            self.image_path = Path(file_name)
            self.original_image = Image.open(self.image_path).convert("RGB")
            self.processed_image = None
            self.before_panel.set_image(self.original_image)
            self.after_panel.clear()
            safe_mode_applied = self.apply_auto_safe_mode(self.original_image)
            if not safe_mode_applied:
                self.status_label.setText(f"Loaded {self.image_path.name}")
            self.progress.setValue(0)
            self.update_action_state()
        except Exception as exc:
            QMessageBox.critical(self, "Image error", f"Could not open image:\n{exc}")

    def apply_auto_safe_mode(self, image: Image.Image) -> bool:
        if self.is_very_small_or_blurry(image):
            self.safe_restore_toggle.setChecked(True)
            self.scale_combo.setCurrentText("2x")
            self.status_label.setText("Very small or blurry image detected; Safe Mode set to 2x")
            return True
        return False

    def is_very_small_or_blurry(self, image: Image.Image) -> bool:
        small = min(image.width, image.height) < 320 or (image.width * image.height) < 160_000
        gray = cv2.cvtColor(pil_to_cv(image), cv2.COLOR_BGR2GRAY)
        blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
        blurry = blur_score < 45
        return small or blurry

    def collect_options(self) -> ProcessingOptions:
        return ProcessingOptions(
            model_name=self.model_combo.currentText(),
            scale_mode=self.scale_combo.currentText(),
            face_enhance=self.face_toggle.isChecked(),
            denoise=self.denoise_toggle.isChecked(),
            safe_restore=self.safe_restore_toggle.isChecked(),
            sharpness=self.sharpness_slider.value(),
            contrast=self.contrast_slider.value(),
            color_boost=self.color_slider.value(),
        )

    def on_safe_restore_toggled(self, enabled: bool) -> None:
        if enabled:
            self.face_toggle.setChecked(False)
            self.denoise_toggle.setChecked(True)
            self.scale_combo.blockSignals(True)
            self.scale_combo.setCurrentText("2x")
            self.scale_combo.blockSignals(False)
            self.sharpness_slider.slider.setValue(4)
            self.contrast_slider.slider.setValue(2)
            self.color_slider.slider.setValue(0)
        self.face_toggle.setEnabled(not enabled)
        self.scale_combo.setEnabled(True)

    def on_scale_mode_changed(self, mode: str) -> None:
        if mode != "2x" and self.safe_restore_toggle.isChecked():
            self.safe_restore_toggle.blockSignals(True)
            self.safe_restore_toggle.setChecked(False)
            self.safe_restore_toggle.blockSignals(False)
            self.face_toggle.setEnabled(True)
            self.status_label.setText("Safe Mode turned off for larger upscale output")

    def process_image(self) -> None:
        if not self.image_path:
            QMessageBox.information(self, "No image", "Upload an image before processing.")
            return

        self.set_processing_state(True)
        self.processor = ImageProcessor(self.image_path, self.collect_options())
        self.processor.progress.connect(self.on_progress)
        self.processor.finished.connect(self.on_finished)
        self.processor.failed.connect(self.on_failed)
        self.processor.start()

    def on_progress(self, value: int, message: str) -> None:
        self.progress.setValue(value)
        self.status_label.setText(message)

    def on_finished(self, image: Image.Image, message: str) -> None:
        self.processed_image = image
        self.after_panel.set_image(image)
        self.progress.setValue(100)
        self.status_label.setText(message)
        self.processor = None
        self.set_processing_state(False)
        self.update_action_state()

    def on_failed(self, message: str) -> None:
        self.set_processing_state(False)
        self.status_label.setText("Processing failed")
        self.processor = None
        QMessageBox.critical(self, "Processing error", message)
        self.update_action_state()

    def export_image(self, image_format: str) -> None:
        if self.processed_image is None:
            QMessageBox.information(self, "No enhanced image", "Enhance an image before exporting.")
            return

        suffix = ".png" if image_format == "PNG" else ".jpg"
        default_name = OUTPUT_DIR / f"clarityforge_enhanced{suffix}"
        file_name, _ = QFileDialog.getSaveFileName(
            self,
            f"Export {image_format}",
            str(default_name),
            f"{image_format} image (*{suffix})",
        )
        if not file_name:
            return

        try:
            save_path = Path(file_name)
            if image_format == "JPG":
                self.processed_image.convert("RGB").save(save_path, "JPEG", quality=96, optimize=True)
            else:
                self.processed_image.save(save_path, "PNG", optimize=True)
            self.status_label.setText(f"Exported {save_path.name}")
        except Exception as exc:
            QMessageBox.critical(self, "Export error", f"Could not export image:\n{exc}")

    def set_processing_state(self, processing: bool) -> None:
        self.upload_button.setEnabled(not processing)
        self.process_button.setEnabled(not processing and self.image_path is not None)
        self.model_combo.setEnabled(not processing)
        self.scale_combo.setEnabled(not processing)
        self.face_toggle.setEnabled(not processing and not self.safe_restore_toggle.isChecked())
        self.denoise_toggle.setEnabled(not processing)
        self.safe_restore_toggle.setEnabled(not processing)
        self.export_png_button.setEnabled(not processing and self.processed_image is not None)
        self.export_jpg_button.setEnabled(not processing and self.processed_image is not None)

    def update_action_state(self) -> None:
        self.process_button.setEnabled(self.image_path is not None)
        self.export_png_button.setEnabled(self.processed_image is not None)
        self.export_jpg_button.setEnabled(self.processed_image is not None)


def apply_styles(app: QApplication) -> None:
    app.setStyleSheet(
        """
        QWidget#appRoot {
            background: #0b0f17;
            color: #e8edf7;
            font-family: "Segoe UI", Arial, sans-serif;
            font-size: 14px;
        }
        QLabel#appTitle {
            color: #f7f9ff;
            font-size: 32px;
            font-weight: 700;
        }
        QLabel#appSubtitle {
            color: #8f9bb3;
            font-size: 14px;
        }
        QLabel#deviceBadge, QLabel#valuePill {
            background: #162235;
            border: 1px solid #26364f;
            border-radius: 12px;
            color: #b9d4ff;
            padding: 6px 10px;
        }
        QFrame#previewCard, QFrame#controlCard {
            background: #111827;
            border: 1px solid #243047;
            border-radius: 18px;
        }
        QLabel#panelTitle, QLabel#sectionTitle {
            color: #f4f7fb;
            font-size: 18px;
            font-weight: 650;
        }
        QLabel#previewImage {
            background: #090d14;
            border: 1px dashed #2b3a55;
            border-radius: 14px;
            color: #64748b;
        }
        QLabel#controlLabel {
            color: #c8d2e4;
            font-weight: 600;
        }
        QLabel#statusText {
            color: #9aa7bd;
            padding-left: 4px;
        }
        QLabel#warningText {
            background: #171f2e;
            border: 1px solid #334155;
            border-radius: 10px;
            color: #d6b86a;
            padding: 10px;
            line-height: 1.25;
        }
        QPushButton {
            background: #1f6feb;
            border: 0;
            border-radius: 12px;
            color: #ffffff;
            font-weight: 700;
            padding: 12px 14px;
        }
        QPushButton:hover {
            background: #2b7cff;
        }
        QPushButton:disabled {
            background: #202939;
            color: #68758d;
        }
        QComboBox {
            background: #0c1220;
            border: 1px solid #283954;
            border-radius: 10px;
            color: #e7edf8;
            padding: 10px;
        }
        QCheckBox {
            color: #d9e2f2;
            spacing: 10px;
        }
        QCheckBox::indicator {
            width: 18px;
            height: 18px;
            border-radius: 5px;
            border: 1px solid #41536f;
            background: #0c1220;
        }
        QCheckBox::indicator:checked {
            background: #2dd4bf;
            border: 1px solid #2dd4bf;
        }
        QSlider::groove:horizontal {
            height: 6px;
            background: #27364f;
            border-radius: 3px;
        }
        QSlider::sub-page:horizontal {
            background: #2dd4bf;
            border-radius: 3px;
        }
        QSlider::handle:horizontal {
            background: #f8fafc;
            border: 2px solid #2dd4bf;
            width: 18px;
            margin: -7px 0;
            border-radius: 9px;
        }
        QProgressBar {
            background: #111827;
            border: 1px solid #26364f;
            border-radius: 9px;
            color: #dce7f7;
            height: 18px;
            text-align: center;
        }
        QProgressBar::chunk {
            background: #2dd4bf;
            border-radius: 8px;
        }
        """
    )


def main() -> int:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    configure_windows_taskbar_icon()
    ensure_project_dirs()

    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)
    app.setOrganizationName("ClarityForge")
    app.setWindowIcon(app_icon())
    apply_styles(app)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
