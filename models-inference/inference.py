from fastapi import FastAPI
from pydantic import BaseModel, Field
from ultralytics import YOLO
import os, cv2, glob
import torch
from pathlib import Path

torch.set_num_threads(2)
torch.set_num_interop_threads(2)

image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")

app = FastAPI()
model = YOLO("models/best.pt")


class CropToLicensePlatesRequest(BaseModel):
    raw_data_dir: str = Field(alias="RawDataPath")
    result_data_dir: str = Field(alias="ResultDataPath")


@app.post("/api/cropToLicensePlates")
async def crop_to_license_plates(req: CropToLicensePlatesRequest):
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
