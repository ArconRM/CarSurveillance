import math
import numpy as np
import cv2
from PIL import Image, ImageEnhance
from pathlib import Path
from upscalers import UpscalerManager


def rotate_and_zoom(img: Image.Image, angle: float) -> Image.Image:
    """
    Rotates the image, zooms to remove the black fields, and returns the frame to its original size.
    """
    w, h = img.size

    rotated = img.rotate(angle, expand=True)

    radians = math.radians(abs(angle))
    sin_a = math.sin(radians)
    cos_a = math.cos(radians)

    new_w = w * cos_a + h * sin_a
    new_h = w * sin_a + h * cos_a

    zoom_factor = max(new_w / w, new_h / h)

    zoomed_w = int(rotated.size[0] * zoom_factor)
    zoomed_h = int(rotated.size[1] * zoom_factor)
    zoomed = rotated.resize((zoomed_w, zoomed_h), Image.Resampling.LANCZOS)

    left = (zoomed_w - w) // 2
    top = (zoomed_h - h) // 2
    right = left + w
    bottom = top + h

    return zoomed.crop((left, top, right, bottom))


def crop_center_vertical(img: Image.Image, keep_ratio: float = 0.6) -> Image.Image:
    """
    Leaves the center 60% vertically (or another keep_ratio value).
    """
    w, h = img.size

    crop_h = int(h * keep_ratio)
    offset = (h - crop_h) // 2  # 20% сверху и 20% снизу при keep_ratio=0.6

    top = offset
    bottom = offset + crop_h

    return img.crop((0, top, w, bottom))


def enhance_contrast(img: Image.Image, factor: float = 1.2) -> Image.Image:
    enhancer = ImageEnhance.Contrast(img)
    return enhancer.enhance(factor)


def preprocess_crop(img_path: str, upscaler_manager: UpscalerManager,
                   upscaler_name: str = None, angle: float = 15.0):
    """
    Preprocesses a single crop with optional upscaler selection:
    1. Opens the image from disk.
    2. Rotates and zooms.
    3. Crops vertically.
    4. Applies upscaling (if upscaler_name is provided).
    5. Enhances contrast.
    Returns a processed numpy array for OCR.
    """
    try:
        img = Image.open(img_path)

        # 1. Rotate and Crop
        img = rotate_and_zoom(img, angle)
        img = crop_center_vertical(img, keep_ratio=0.6)

        # 2. Apply upscaling (if specified)
        if upscaler_name:
            # Temporarily switch to the specified upscaler
            original_upscaler = upscaler_manager.current_upscaler
            upscaler_manager.set_upscaler(upscaler_name)
            img = upscaler_manager.process_pil(img)
            upscaler_manager.current_upscaler = original_upscaler

        # 3. Enhance contrast
        img = enhance_contrast(img, factor=1.15)

        # Convert to numpy array for OCR
        img_np = np.array(img)

        if img_np.ndim == 2:  # Grayscale
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        return img_np

    except Exception as e:
        print(f"Error processing {img_path} with upscaler {upscaler_name}: {e}")
        return None