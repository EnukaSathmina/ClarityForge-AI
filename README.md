# ClarityForge AI

ClarityForge AI is a professional Python desktop app for enhancing low-quality, pixelated, or blurry images with Real-ESRGAN AI upscaling and optional GFPGAN face restoration.

## What The Enhance Button Does

The **Enhance Image** button runs Real-ESRGAN inference. It does not use `cv2.resize()` or Pillow resizing as the main enhancement path.

After Real-ESRGAN finishes, the app can apply optional OpenCV finishing:

- Denoise
- Sharpen
- Contrast
- Color boost

## Features

- Modern dark PySide6 desktop interface
- Before and after image preview panels
- Real-ESRGAN upscale modes: `2x`, `4x`, and `4K target`
- Model selector for `realesr-general-x4v3` and `RealESRGAN_x4plus`
- Optional GFPGAN face enhancement only when a face is detected
- Safe Mode for very small, blurry, low-quality images and silhouettes
- CUDA GPU auto-detection when PyTorch can access a compatible NVIDIA GPU
- CPU fallback when CUDA is unavailable
- PNG and JPG export
- Clear error message when Real-ESRGAN model weights are missing

## Folder Structure

```text
ClarityForge/
|-- main.py
|-- requirements.txt
|-- README.md
|-- models/
|-- output/
`-- assets/
```

## Setup

Create and activate a virtual environment:

```bash
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

If you already installed dependencies and see a NumPy error such as
`RuntimeError: Numpy is not available` or `_ARRAY_API not found`, downgrade
NumPy in the same Python environment:

```bash
python -m pip install --force-reinstall "numpy>=1.24,<2"
```

For CUDA acceleration, install the PyTorch build that matches your CUDA version from the official PyTorch instructions:

https://pytorch.org/get-started/locally/

## Download Real-ESRGAN Model Weights

Download at least one Real-ESRGAN upscaler model and place it in the `models/` folder.

Fastest option:

```bash
py download_models.py general
```

To download every supported model, including optional face restoration:

```bash
py download_models.py --all
```

Recommended general model:

```text
models/realesr-general-x4v3.pth
```

Download:

```text
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth
```

Higher-quality x4plus model:

```text
models/RealESRGAN_x4plus.pth
```

Download:

```text
https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth
```

The app defaults to `realesr-general-x4v3` for general images. Use the Model selector to choose `RealESRGAN_x4plus`.

## Optional Face Restoration Weights

For face enhancement, place GFPGAN weights here:

```text
models/GFPGANv1.4.pth
```

Download:

```text
https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth
```

Face enhancement is optional. Real-ESRGAN upscaling requires one of the Real-ESRGAN model files above.

## Run

```bash
python main.py
```

## Windows EXE

A packaged Windows build is created here:

```text
dist/ClarityForge AI/ClarityForge AI.exe
```

Run the app by opening:

```text
dist\ClarityForge AI\ClarityForge AI.exe
```

Keep the full `dist/ClarityForge AI/` folder together. The EXE depends on the bundled `_internal/` folder and bundled model files.

Enhanced exports are saved by default to:

```text
dist/ClarityForge AI/output/
```

## Build The EXE

Install PyInstaller in the same Python environment:

```bash
python -m pip install pyinstaller
```

Build from the project root:

```bash
python -m PyInstaller --noconfirm --windowed --name "ClarityForge AI" --icon icon.ico --add-data "icon.ico;." --add-data "models;models" main.py
```

After the first build, you can rebuild from the generated spec file:

```bash
python -m PyInstaller --noconfirm "ClarityForge AI.spec"
```

## Notes

- Large 4K exports can use significant memory, especially on CPU.
- If no Real-ESRGAN model file exists in `models/`, processing stops with an error instead of silently resizing the image.
- If CUDA is available, the app uses GPU half precision. Otherwise, it runs on CPU.
- Very low-resolution images may produce AI-generated artifacts because missing details must be guessed.
- For silhouette, sunset, landscape, and other non-face images, leave face enhancement off or use Safe Mode. The app also skips GFPGAN automatically when no face is detected.
- Safe Mode forces conservative 2x output, disables face restoration, uses low denoise, very low sharpening, and very low contrast.
- Very small or blurry uploads automatically switch to Safe Mode and 2x output. Choosing `4x` or `4K target` turns Safe Mode off.
- `4K target` runs Real-ESRGAN x4 directly on the original image first, then resizes the enhanced output to a 3840px long edge while keeping aspect ratio.
