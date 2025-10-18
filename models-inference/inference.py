import json
from fastapi import FastAPI
from paddleocr import PaddleOCR
from pydantic import BaseModel, Field
from ultralytics import YOLO
import os, cv2, glob
import numpy as np
import torch
from pathlib import Path

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
    result_data_dir: str = Field(alias="ResultDataPath")


class RecognizeLicensePlatesRequest(BaseModel):
    crops_data_dir: str = Field(alias="CropsDataPath")
    result_data_dir: str = Field(alias="ResultDataPath")


def unwarp_plate(img, min_tilt_ratio=0.12):
    """
    Corrects rotation and mild perspective tilt of a license plate crop.
    Works safely — never distorts aspect ratio or overcorrects.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)

    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    contour = max(contours, key=cv2.contourArea)
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect

    # Only rotate if plate clearly not horizontal
    if abs(angle) > 5:
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        img = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]),
                             flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    edges = cv2.Canny(blur, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return img

    contour = max(contours, key=cv2.contourArea)
    peri = cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, 0.02 * peri, True)

    if len(approx) == 4:
        pts = approx.reshape(4, 2).astype(np.float32)
        (x, y, w, h) = cv2.boundingRect(pts)
        rect = np.array([[x, y],
                         [x + w, y],
                         [x + w, y + h],
                         [x, y + h]], dtype=np.float32)
        diff_ratio = np.mean(np.abs(pts - rect)) / max(w, h)
        if diff_ratio > min_tilt_ratio:
            dst_pts = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], dtype=np.float32)
            M = cv2.getPerspectiveTransform(pts, dst_pts)
            warped = cv2.warpPerspective(img, M, (w, h))
            return warped

    return img


def is_valid_crop(img):
    h, w = img.shape[:2]
    aspect_ratio = w / h
    return w >= 95 and aspect_ratio >= 1


def preprocess_crop(path):
    img = cv2.imread(path)
    if not is_valid_crop(img):
        return None

    img = unwarp_plate(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    gray = cv2.convertScaleAbs(gray, alpha=1.2, beta=0)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    h, w = gray.shape
    target_h = 100
    new_w = int(w * (target_h / h))
    resized = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_LINEAR)
    norm = cv2.normalize(resized, None, 0, 255, cv2.NORM_MINMAX)
    return norm


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/cropToLicensePlates")
async def crop_to_license_plates(req: CropToLicensePlatesRequest):
    """
    Run YOLO license plate detection on raw images and crop them
    """
    raw_data_dir = req.raw_data_dir
    result_data_dir = req.result_data_dir

    frame_paths = []
    for ext in image_extensions:
        frame_paths.extend(glob.glob(os.path.join(raw_data_dir, "**", ext), recursive=True))
    frame_paths = sorted(frame_paths)

    print(f"Found {len(frame_paths)} frames in {raw_data_dir}")

    os.makedirs(result_data_dir, exist_ok=True)

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
                crop = img[y1:y2, x1:x2].copy()
                outp = os.path.join(result_data_dir, f"crop_{time}.png")
                cv2.imwrite(outp, crop)
                crop_index += 1

        overall_index += 1

    print("Detection+crop finished.", crop_index, "Crops saved to", result_data_dir)
    return {"status": "ok", "crops_saved": crop_index}


@app.post("/api/recognizeLicensePlates")
async def recognize_license_plates(req: RecognizeLicensePlatesRequest):
    """
    Run OCR on pre-cropped license plate images
    """
    raw_data_dir = req.crops_data_dir
    result_data_dir = req.result_data_dir

    crop_paths = []
    for ext in image_extensions:
        crop_paths.extend(glob.glob(os.path.join(raw_data_dir, ext)))
    crop_paths = sorted(crop_paths)

    print(f"Found {len(crop_paths)} crops in {raw_data_dir}")

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
