import json
import math
from PIL import Image, ImageEnhance, ImageFilter
from fastapi import FastAPI
from paddleocr import PaddleOCR
from pydantic import BaseModel, Field
from ultralytics import YOLO
import os, cv2, glob
import torch
from pathlib import Path
from fast_plate_ocr import LicensePlateRecognizer

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


def denoise_image(img: Image.Image) -> Image.Image:
    return img.filter(ImageFilter.MedianFilter(size=3))


def preprocess_crop(img_path: str, angle: float = 15.0):
    try:
        img = Image.open(img_path)

        img = rotate_and_zoom(img, angle)
        img = crop_center_vertical(img, keep_ratio=0.6)
        img = enhance_contrast(img, factor=1.15)
        img = denoise_image(img)
        print(f"✓ Processed {img_path}")

        return img

    except Exception as e:
        print(f"Error processing {img_path}: {e}")


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
    Run OCR on pre-cropped license plate images
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
    for idx, cp in enumerate(crop_paths):
        img = cv2.imread(cp)
        if img is None:
            continue

        processed_img = preprocess_crop(cp)
        if processed_img is None:
            continue

        result_raw = ocr_model.ocr(img)
        try:
            text_raw, confidence_raw = result_raw[0][0][1]
        except:
            continue

        result_processed = ocr_model.ocr(processed_img)
        try:
            text_processed, confidence_processed = result_processed[0][0][1]
        except:
            continue

        filename = Path(cp).name
        time = Path(cp).stem.split("_")[1].split("-")[0]
        results.append({
            "time": time,
            "filename": filename,
            "plate_text_raw": text_raw,
            "confidence_raw": float(confidence_raw),
            "plate_text_processed": text_processed,
            "confidence_processed": float(confidence_processed),
        })
        print(f"[{idx + 1}/{len(crop_paths)}] {filename}: {text_raw} (conf: {confidence_raw:.3f}), {text_processed} (conf: {confidence_processed:.3f})")

    result_file = os.path.join(result_data_dir, "recognition_results.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Recognition finished. Results saved to {result_file}")
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
