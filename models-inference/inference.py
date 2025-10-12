import json
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ultralytics import YOLO
import os, cv2, glob
import torch
from pathlib import Path
from paddleocr import PaddleOCR

torch.set_num_threads(3)
torch.set_num_interop_threads(3)

image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")

app = FastAPI()
model = YOLO("models/best.pt")
ocr = PaddleOCR(
    use_angle_cls=False,
    lang='en',
    det=False,
    rec_model_dir='models/best_model',
    rec_char_dict_path='models/best_model/latin_plate_dict.txt',
    use_gpu=False,
    use_mp=True,
    total_process_num=3,
    show_log=False
)


class CropToLicensePlatesRequest(BaseModel):
    raw_data_dir: str = Field(alias="RawDataPath")
    result_data_dir: str = Field(alias="ResultDataPath")


class RecognizeLicensePlatesRequest(BaseModel):
    crops_data_dir: str = Field(alias="CropsDataPath")
    result_data_dir: str = Field(alias="ResultDataPath")


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

        res = model.predict(source=img, conf=0.25, imgsz=1280, verbose=False)
        for r in res:
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
    raw_data_dir = req.raw_data_dir
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

        result = ocr.ocr(img, cls=False)

        if result and result[0]:
            text = result[0][0][1][0]
            confidence = result[0][0][1][1]

            filename = Path(cp).name
            time = Path(cp).stem.split("_")[1]
            results.append({
                "time": time,
                "filename": filename,
                "plate_text": text,
                "confidence": float(confidence)
            })
            print(f"[{idx + 1}/{len(crop_paths)}] {filename}: {text} (conf: {confidence:.3f})")
        else:
            print(f"[{idx + 1}/{len(crop_paths)}] {Path(cp).name}: No text detected")
            results.append({
                "time": time,
                "filename": Path(cp).name,
                "plate_text": "",
                "confidence": 0.0
            })

    result_file = os.path.join(result_data_dir, "recognition_results.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Recognition finished. Results saved to {result_file}")
    return {
        "status": "ok",
        "total_processed": len(results),
        "results": results
    }
