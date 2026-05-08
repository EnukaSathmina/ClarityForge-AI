@ -1,2 +1,142 @@
# ClarityForge-AI
ClarityForge AI is a professional Python desktop app for enhancing low-quality, pixelated, or blurry images with Real-ESRGAN AI upscaling and optional GFPGAN face restoration.
# 🖼️ ClarityForge AI

**ClarityForge AI** is a professional Python desktop application for enhancing low-quality, pixelated, or blurry images using **Real-ESRGAN AI upscaling** with optional **GFPGAN face restoration**.

It is designed to upscale images, reduce visual artifacts, improve clarity, and export enhanced images in high resolution.

---

## ✨ Features

- 🖥️ Modern dark-themed **PySide6** desktop interface
- 🖼️ Before and after image preview panels
- 🤖 Real AI upscaling with **Real-ESRGAN**
- 🔍 Upscale modes: `2x`, `4x`, and `4K target`
- 🧠 Model selector:
  - `realesr-general-x4v3`
  - `RealESRGAN_x4plus`
- 😀 Optional **GFPGAN face enhancement**
- 🛡️ Safe Mode for very small, blurry, low-quality, or silhouette images
- ⚡ CUDA GPU auto-detection for supported NVIDIA GPUs
- 🐢 CPU fallback when CUDA is unavailable
- 🎚️ Optional OpenCV finishing tools:
  - Denoise
  - Sharpen
  - Contrast
  - Color boost
- 📤 Export enhanced images as PNG or JPG
- ✅ Clear error messages when required model files are missing

---

## 📷 Preview

> ![ClarityForge Preview](https://github.com/EnukaSathmina/ClarityForge-AI/blob/main/img.png?raw=true)

---

## 🚀 What The Enhance Button Does

The **Enhance Image** button runs real **Real-ESRGAN inference**.

It does **not** use `cv2.resize()` or Pillow resizing as the main enhancement method.

Processing flow:

```text
Original image
↓
Real-ESRGAN AI upscaling
↓
Optional denoise / sharpen / contrast / color boost
↓
Optional 4K target resize
↓
Export PNG or JPG
```

## 🧠 AI Models Used

ClarityForge AI uses **Real-ESRGAN** for AI image upscaling and optional **GFPGAN** for face restoration.

| Model | Type | Best For |
|---|---|---|
| `realesr-general-x4v3` | Upscaling | General images, safer results, fewer artifacts |
| `RealESRGAN_x4plus` | Upscaling | Higher-quality 4x enhancement for detailed images |
| `GFPGANv1.4` | Face Restoration | Improving visible human faces |

> **Note:** Face enhancement works best only when a clear face is visible.  
> For landscapes, sunsets, silhouettes, and non-face images, keep face enhancement disabled or use **Safe Mode**.

###⚠️ Important Note About AI Restoration

- ClarityForge AI can improve image quality, but it cannot perfectly recover details that do not exist in the original image.

- Very low-resolution images may produce AI-generated artifacts because the AI has to guess missing details.

- Best results come from images that are low-quality but still contain some visible structure and detail.

# 🛠️ Setup

Follow these steps to run **ClarityForge AI** on your PC.

### 1️⃣ Clone the repository

```bash
git clone https://github.com/your-username/ClarityForge.git
cd ClarityForge
```
### 2️⃣ Create a virtual environment
```bash
python -m venv .venv
```
### 3️⃣ Activate the virtual environment

Windows:
```bash
.venv\Scripts\activate
```

Linux / macOS:
```bash
source .venv/bin/activate
```
### 4️⃣ Install dependencies
```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 5️⃣ Download AI model weights

Place the Real-ESRGAN model inside the models/ folder.

Recommended model:

- models/realesr-general-x4v3.pth

Optional face restoration model:

- models/GFPGANv1.4.pth

You can download models manually or run:

- python download_models.py general

To download all supported models:

- python download_models.py --all
### 6️⃣ Run the app
```bash
python main.py
```

## ⚡ GPU Support

For NVIDIA GPU acceleration, install a CUDA-supported PyTorch version from the official PyTorch website.

Example for CUDA 11.8:
```bash
python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```
If CUDA is not available, the app will automatically run on CPU.

<h2 align="center">👨‍💻 Author</h2>

<p align="center">
  Made by <b>Enuka Sathmina</b>
</p>
