import json
import numpy as np
from fastapi import FastAPI
from pydantic import BaseModel, Field
from ultralytics import YOLO
import os, cv2, glob
import torch
from pathlib import Path
import onnxruntime as ort

torch.set_num_threads(3)
torch.set_num_interop_threads(3)

image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")

app = FastAPI()
model = YOLO("models/best.pt")


def load_char_dict(dict_path):
    with open(dict_path, 'r', encoding='utf-8') as f:
        chars = [line.strip() for line in f.readlines()]
    return chars


char_dict = load_char_dict('models/latin_plate_dict.txt')

onnx_session = ort.InferenceSession(
    'models/plate_rec.onnx',
    providers=['CPUExecutionProvider']
)

input_name = onnx_session.get_inputs()[0].name
output_name = onnx_session.get_outputs()[0].name


def preprocess_image(img):
    """Preprocess image for ONNX model"""
    h, w = img.shape[:2]

    ratio = 32.0 / h
    new_w = int(w * ratio)
    if new_w > 320:
        new_w = 320
        ratio = 320.0 / w
        new_h = int(h * ratio)
    else:
        new_h = 32

    img_resized = cv2.resize(img, (new_w, new_h))

    img_norm = img_resized.astype('float32') / 255.0
    # img_norm = (img_norm - 0.5) / 0.5

    # HWC to CHW
    img_chw = img_norm.transpose((2, 0, 1))

    img_batch = np.expand_dims(img_chw, axis=0)

    return img_batch


def softmax(x):
    """Softmax function"""
    exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def decode_ctc(preds):
    """Decode CTC output"""
    preds_idx = np.argmax(preds, axis=2)[0]

    # CTC decoding: remove consecutive duplicates and blank (last index)
    last_idx = -1
    result = []
    blank_idx = len(char_dict)

    for idx in preds_idx:
        if idx != last_idx and idx != blank_idx:
            if idx < len(char_dict):
                result.append(char_dict[idx])
        last_idx = idx

    return ''.join(result)


def recognize_plate(img):
    """Run ONNX inference on plate image"""
    input_data = preprocess_image(img)

    outputs = onnx_session.run([output_name], {input_name: input_data})
    preds = outputs[0]

    text = decode_ctc(preds)

    probs = softmax(preds)
    confidence = np.mean(np.max(probs, axis=2))

    return text, float(confidence)


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

        results = model.predict(
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

        # Run OCR
        text, confidence = recognize_plate(img)

        filename = Path(cp).name
        time = Path(cp).stem.split("_")[1]
        results.append({
            "time": time,
            "filename": filename,
            "plate_text": text,
            "confidence": float(confidence)
        })
        print(f"[{idx + 1}/{len(crop_paths)}] {filename}: {text} (conf: {confidence:.3f})")

    result_file = os.path.join(result_data_dir, "recognition_results.json")
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)

    print(f"Recognition finished. Results saved to {result_file}")
    return {
        "status": "ok",
        "total_processed": len(results),
        "results": results
    }
