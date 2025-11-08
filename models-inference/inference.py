import json
import math
import numpy as np
import os, cv2, glob
import torch

from PIL import Image, ImageEnhance
from fastapi import FastAPI
from paddleocr import PaddleOCR
from pydantic import BaseModel, Field
from ultralytics import YOLO
from pathlib import Path
from fast_plate_ocr import LicensePlateRecognizer
from abc import ABC, abstractmethod


class BaseUpscaler(ABC):
    """Base class for all upscalers"""

    @abstractmethod
    def process_pil(self, img: Image.Image) -> Image.Image:
        pass

    @abstractmethod
    def get_name(self) -> str:
        pass


class RealESRGANUpscaler(BaseUpscaler):
    """Real-ESRGAN NCNN upscaler (works on Mac/Windows, fails in Docker)"""

    def __init__(self, gpuid: int = 0):
        try:
            from realesrgan_ncnn_py import Realesrgan
            self.upscaler = Realesrgan(gpuid=gpuid)
            self.available = True
            print(f"✓ Real-ESRGAN loaded (GPU: {gpuid})")
        except Exception as e:
            print(f"✗ Real-ESRGAN unavailable: {e}")
            self.available = False

    def process_pil(self, img: Image.Image) -> Image.Image:
        if not self.available:
            raise RuntimeError("Real-ESRGAN not available")
        return self.upscaler.process_pil(img)

    def get_name(self) -> str:
        return "Real-ESRGAN"


def process_pil(self, img: Image.Image) -> Image.Image:
    if not self.available:
        raise RuntimeError("Real-ESRGAN not available")

    # Convert PIL to numpy array (RGB)
    img_np = np.array(img)

    # Upscale (output is BGR numpy array)
    output, _ = self.upsampler.enhance(img_np, outscale=4)

    # Convert BGR back to RGB PIL Image
    output_rgb = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
    return Image.fromarray(output_rgb)


def get_name(self) -> str:
    return "Real-ESRGAN"


class LanczosUpscaler(BaseUpscaler):
    """Lanczos interpolation upscaler (fast, works everywhere)"""

    def __init__(self, scale: int = 4):
        self.scale = scale
        self.available = True
        print(f"✓ Lanczos upscaler loaded (x{scale})")

    def process_pil(self, img: Image.Image) -> Image.Image:
        w, h = img.size
        new_size = (w * self.scale, h * self.scale)
        return img.resize(new_size, Image.Resampling.LANCZOS)

    def get_name(self) -> str:
        return f"Lanczos-x{self.scale}"


class EnhancedLanczosUpscaler(BaseUpscaler):
    """Lanczos + sharpening + denoising (best traditional method for plates)"""

    def __init__(self, scale: int = 4):
        self.scale = scale
        self.available = True
        print(f"✓ Enhanced Lanczos upscaler loaded (x{scale})")

    def process_pil(self, img: Image.Image) -> Image.Image:
        # Convert to OpenCV for preprocessing
        img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

        # Denoise before upscaling
        denoised = cv2.fastNlMeansDenoisingColored(img_cv, None, 10, 10, 7, 21)

        # Upscale with Lanczos
        h, w = denoised.shape[:2]
        upscaled = cv2.resize(denoised, (w * self.scale, h * self.scale),
                              interpolation=cv2.INTER_LANCZOS4)

        # Sharpen
        kernel = np.array([[-1, -1, -1],
                           [-1, 9, -1],
                           [-1, -1, -1]])
        sharpened = cv2.filter2D(upscaled, -1, kernel)

        # Convert back to PIL
        result_rgb = cv2.cvtColor(sharpened, cv2.COLOR_BGR2RGB)
        return Image.fromarray(result_rgb)

    def get_name(self) -> str:
        return f"Enhanced-Lanczos-x{self.scale}"


class NoUpscaler(BaseUpscaler):
    """No upscaling - baseline comparison"""

    def __init__(self):
        self.available = True
        print("✓ No upscaling (baseline)")

    def process_pil(self, img: Image.Image) -> Image.Image:
        return img

    def get_name(self) -> str:
        return "None"


class UpscalerManager:
    """Manages multiple upscalers with automatic fallback"""

    def __init__(self, preferred_method: str = "auto"):
        self.upscalers = []
        self.current_upscaler = None
        self.preferred_method = preferred_method

        # Try to initialize upscalers in order of preference
        self._initialize_upscalers()

    def _initialize_upscalers(self):
        """Initialize all available upscalers"""

        # Try Real-ESRGAN first (best quality)
        try:
            upscaler = RealESRGANUpscaler()
            if upscaler.available:
                self.upscalers.append(upscaler)
        except Exception as e:
            print(f"Skipping Real-ESRGAN: {e}")

        # Enhanced Lanczos (fast, decent quality for plates)
        self.upscalers.append(EnhancedLanczosUpscaler())

        # Standard Lanczos (fast fallback)
        self.upscalers.append(LanczosUpscaler())

        # No upscaling (baseline)
        self.upscalers.append(NoUpscaler())

        # Set current upscaler based on preference
        if self.preferred_method == "auto":
            self.current_upscaler = self.upscalers[0]
        else:
            for upscaler in self.upscalers:
                if self.preferred_method.lower() in upscaler.get_name().lower():
                    self.current_upscaler = upscaler
                    break
            if not self.current_upscaler:
                self.current_upscaler = self.upscalers[0]

        print(f"\n{'=' * 60}")
        print(f"Active Upscaler: {self.current_upscaler.get_name()}")
        print(f"Available Upscalers: {[u.get_name() for u in self.upscalers]}")
        print(f"{'=' * 60}\n")

    def process_pil(self, img: Image.Image) -> Image.Image:
        """Process image with current upscaler"""
        return self.current_upscaler.process_pil(img)

    def set_upscaler(self, method: str):
        """Switch to a different upscaler"""
        for upscaler in self.upscalers:
            if method.lower() in upscaler.get_name().lower():
                self.current_upscaler = upscaler
                print(f"Switched to: {upscaler.get_name()}")
                return True
        print(f"Upscaler '{method}' not found")
        return False

    def get_available_methods(self) -> list:
        """Get list of available upscaling methods"""
        return [u.get_name() for u in self.upscalers]



fastplate_model = LicensePlateRecognizer('cct-s-v1-global-model')

torch.set_num_threads(3)
torch.set_num_interop_threads(3)

image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")

app = FastAPI()
detection_model = YOLO("models/best.pt")

ocr_model = PaddleOCR(
    lang='en',
    det=False,
    rec=True,
    use_angle_cls=False
)

upscaler_manager = UpscalerManager()

class CropToLicensePlatesRequest(BaseModel):
    raw_data_dir: str = Field(alias="RawDataPath")
    crops_data_dir: str = Field(alias="CropsDataPath")


class RecognizeLicensePlatesRequest(BaseModel):
    crops_data_dir: str = Field(alias="CropsDataPath")
    result_data_dir: str = Field(alias="ResultDataPath")


class FastPlateOCRRequest(BaseModel):
    crops_data_dir: str = Field(alias="CropsDataPath")
    result_data_dir: str = Field(alias="ResultDataPath")


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


def preprocess_crop(img_path: str, upscaler_name: str = None, angle: float = 15.0):
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
        import numpy as np
        img_np = np.array(img)

        if img_np.ndim == 2:  # Grayscale
            img_np = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)

        return img_np

    except Exception as e:
        print(f"Error processing {img_path} with upscaler {upscaler_name}: {e}")
        return None

@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/cropToLicensePlates")
async def crop_to_license_plates(req: CropToLicensePlatesRequest):
    """
    Run YOLO license plate detection on raw images and crop them
    """
    raw_data_dir = req.raw_data_dir
    crops_data_dir = req.crops_data_dir

    frame_paths = []
    for ext in image_extensions:
        frame_paths.extend(glob.glob(os.path.join(raw_data_dir, "**", ext), recursive=True))
    frame_paths = sorted(frame_paths)

    print(f"Found {len(frame_paths)} frames in {raw_data_dir}")

    os.makedirs(crops_data_dir, exist_ok=True)

    crop_index = 0
    overall_index = 0
    for fp in frame_paths:
        time = Path(fp).stem.split("_")[1]
        print("Processing frame #", overall_index, time)
        img = cv2.imread(fp)
        if img is None:
            continue

        results = detection_model.predict(
            source=img,
            batch=10,
            save=False,
            device='mps'
        )
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                crop = img[y1 - 30:y2 + 30, x1 - 30:x2 + 30].copy()
                outp = os.path.join(crops_data_dir, f"crop_{time}.png")
                cv2.imwrite(outp, crop)
                crop_index += 1

        overall_index += 1

    print("Detection+crop finished.", crop_index, "Crops saved to", crops_data_dir)
    return {"status": "ok", "crops_saved": crop_index}


@app.post("/api/recognizeLicensePlates")
async def recognize_license_plates(req: RecognizeLicensePlatesRequest):
    """
    Run OCR on pre-cropped license plate images with all upscalers
    """
    crops_data_dir = req.crops_data_dir
    result_data_dir = req.result_data_dir

    crop_paths = []
    for ext in image_extensions:
        crop_paths.extend(glob.glob(os.path.join(crops_data_dir, ext)))
    crop_paths = sorted(crop_paths)

    print(f"Found {len(crop_paths)} crops in {crops_data_dir}")

    os.makedirs(result_data_dir, exist_ok=True)

    results = []
    available_upscalers = upscaler_manager.get_available_methods()

    for idx, cp in enumerate(crop_paths):
        img_raw = cv2.imread(cp)
        if img_raw is None:
            continue

        filename = Path(cp).name
        time = Path(cp).stem.split("_")[1].split("-")[0]

        result_raw = ocr_model.ocr(img_raw)
        try:
            text_raw, confidence_raw = result_raw[0][0][1]
        except:
            text_raw, confidence_raw = "N/A", 0.0

        result_entry = {
            "time": time,
            "filename": filename,
            "plate_text_raw": text_raw,
            "confidence_raw": float(confidence_raw),
        }

        for upscaler_name in available_upscalers:
            try:
                # Preprocess with specific upscaler
                processed_img = preprocess_crop(cp, upscaler_name=upscaler_name)

                if processed_img is None:
                    continue

                # Run OCR on processed image
                result_processed = ocr_model.ocr(processed_img)
                try:
                    text_processed, confidence_processed = result_processed[0][0][1]
                except:
                    text_processed, confidence_processed = "N/A", 0.0

                # Create field names based on upscaler name
                field_suffix = upscaler_name.lower().replace("-", "_").replace(" ", "_")

                result_entry[f"plate_text_processed_{field_suffix}"] = text_processed
                result_entry[f"confidence_processed_{field_suffix}"] = float(confidence_processed)

                print(f"  [{upscaler_name}] {text_processed} (conf: {confidence_processed:.3f})")

            except Exception as e:
                print(f"  [{upscaler_name}] Error: {e}")
                field_suffix = upscaler_name.lower().replace("-", "_").replace(" ", "_")
                result_entry[f"plate_text_processed_{field_suffix}"] = "ERROR"
                result_entry[f"confidence_processed_{field_suffix}"] = 0.0

        results.append(result_entry)

        print(f"[{idx + 1}/{len(crop_paths)}] {filename}: RAW={text_raw} (conf: {confidence_raw:.3f})")

    # Save results
    result_file = os.path.join(result_data_dir, "recognition_results.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"\nRecognition finished. Results saved to {result_file}")

    return {
        "status": "ok",
        "total_processed": len(results),
        "results": results
    }


@app.post("/api/recognizeLicensePlatesFast")
async def recognize_license_plates_fast(req: FastPlateOCRRequest):
    """
    Run fast-plate-ocr on pre-cropped license plate images (numpy arrays ONLY).
    """
    crops_data_dir = req.crops_data_dir
    result_data_dir = req.result_data_dir

    crop_paths = []
    for ext in image_extensions:
        crop_paths.extend(glob.glob(os.path.join(crops_data_dir, ext)))
    crop_paths = sorted(crop_paths)

    print(f"[FastPlateOCR] Found {len(crop_paths)} crops in {crops_data_dir}")

    os.makedirs(result_data_dir, exist_ok=True)

    results = []

    for idx, cp in enumerate(crop_paths):
        img_bgr = cv2.imread(cp)
        if img_bgr is None:
            print(f"[FastPlateOCR] Cannot read {cp}")
            continue

        img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

        try:
            out = fastplate_model.run(img)
        except Exception as e:
            print(f"[FastPlateOCR] Error in fastplate_model.run for {cp}: {e}")
            continue

        if not isinstance(out, dict) or "text" not in out:
            print(f"[FastPlateOCR] No valid result for {cp}")
            continue

        text = out["text"]
        confidence = float(out.get("confidence", 0.0))

        filename = Path(cp).name
        time = Path(cp).stem.split("_")[1].split("-")[0]

        results.append({
            "time": time,
            "filename": filename,
            "plate_text_fast": text,
            "confidence_fast": confidence
        })

        print(f"[{idx + 1}/{len(crop_paths)}] {filename}: {text} (conf: {confidence:.3f})")

    result_file = os.path.join(result_data_dir, "recognition_results_fast.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"[FastPlateOCR] Finished. Results saved to {result_file}")

    return {
        "status": "ok",
        "total_processed": len(results),
        "results": results
    }
