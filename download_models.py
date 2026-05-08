import argparse
import sys
import urllib.request
from pathlib import Path


MODELS_DIR = Path(__file__).resolve().parent / "models"

MODELS = {
    "general": {
        "filename": "realesr-general-x4v3.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-general-x4v3.pth",
        "description": "Real-ESRGAN general x4 model, fast and good for most images",
    },
    "x4plus": {
        "filename": "RealESRGAN_x4plus.pth",
        "url": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
        "description": "Real-ESRGAN x4plus model, larger and higher quality",
    },
    "gfpgan": {
        "filename": "GFPGANv1.4.pth",
        "url": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth",
        "description": "Optional GFPGAN face restoration model",
    },
}


def format_size(bytes_count: int) -> str:
    size = float(bytes_count)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:.1f} {unit}"
        size /= 1024
    return f"{bytes_count} B"


def download(name: str) -> Path:
    model = MODELS[name]
    MODELS_DIR.mkdir(exist_ok=True)
    target = MODELS_DIR / model["filename"]

    if target.exists() and target.stat().st_size > 0:
        print(f"Already exists: {target}")
        return target

    print(f"Downloading {model['filename']}")
    print(model["description"])
    print(model["url"])

    def progress(blocks: int, block_size: int, total_size: int) -> None:
        downloaded = blocks * block_size
        if total_size > 0:
            percent = min(100.0, downloaded * 100 / total_size)
            sys.stdout.write(
                f"\r{percent:6.2f}%  {format_size(min(downloaded, total_size))} / {format_size(total_size)}"
            )
        else:
            sys.stdout.write(f"\rDownloaded {format_size(downloaded)}")
        sys.stdout.flush()

    try:
        urllib.request.urlretrieve(model["url"], target, progress)
    except Exception:
        if target.exists():
            target.unlink()
        raise

    print(f"\nSaved to: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Download ClarityForge AI model weights.")
    parser.add_argument(
        "models",
        nargs="*",
        choices=sorted(MODELS),
        default=["general"],
        help="Models to download. Default: general",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download Real-ESRGAN general, RealESRGAN x4plus, and GFPGAN.",
    )
    args = parser.parse_args()

    selected = list(MODELS) if args.all else args.models
    for name in selected:
        download(name)

    print("\nDone. You can now run: py main.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
