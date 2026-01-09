import asyncio
import concurrent.futures
import json
import os
import glob
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import torch
from fastapi import FastAPI
from pydantic import BaseModel, Field
from paddleocr import PaddleOCR
from ultralytics import YOLO
from fast_plate_ocr import LicensePlateRecognizer

from processors import preprocess_crop
from upscalers import UpscalerManager

# Initialize models globally
fastplate_model = LicensePlateRecognizer('cct-s-v1-global-model')
torch.set_num_threads(3)
torch.set_num_interop_threads(3)
image_extensions = ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tiff", "*.webp")


class CropToLicensePlatesRequest(BaseModel):
    model: str = Field(alias="Model")
    raw_data_dir: str = Field(alias="RawDataPath")
    crops_data_dir: str = Field(alias="CropsDataPath")


class RecognizeLicensePlatesRequest(BaseModel):
    crops_data_dir: str = Field(alias="CropsDataPath")
    result_data_dir: str = Field(alias="ResultDataPath")


def create_api() -> FastAPI:
    """Factory function to create FastAPI app with dependencies"""
    app = FastAPI()

    # Initialize models
    detection_model_yolov5 = YOLO("models/best_v5.pt")
    detection_model_yolov8 = YOLO("models/best_v8.pt")
    detection_model_yolov8l = YOLO("models/best_v8l.pt")
    detection_model_yolov11 = YOLO("models/best_v11.pt")

    ocr_model = PaddleOCR(lang='en', det=False, rec=True, use_angle_cls=False)
    upscaler_manager = UpscalerManager()

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.post("/api/cropToLicensePlatesYolo")
    async def crop_to_license_plates_yolo(req: CropToLicensePlatesRequest):
        """
        Run YOLO license plate detection on raw images and crop them with parallelism
        """
        from tqdm import tqdm
        import sys
        import time as time_module

        BATCH_SIZE = 64
        MAX_WORKERS = 4
        DEVICE = "cuda" if torch.cuda.is_available() else ("mps" if torch.mps.is_available() else "cpu")

        if req.model == "yolov5":
            detection_model = detection_model_yolov5
        elif req.model == "yolov8":
            detection_model = detection_model_yolov8
        elif req.model == "yolov8l":
            detection_model = detection_model_yolov8l
        elif req.model == "yolov11":
            detection_model = detection_model_yolov11
        else:
            detection_model = detection_model_yolov8

        raw_data_dir = req.raw_data_dir
        crops_data_dir = req.crops_data_dir

        # Gather file paths
        frame_paths = []
        for ext in image_extensions:
            frame_paths.extend(glob.glob(os.path.join(raw_data_dir, "**", ext), recursive=True))
        frame_paths = sorted(frame_paths)

        print(f"Found {len(frame_paths)} frames in {raw_data_dir}")
        os.makedirs(crops_data_dir, exist_ok=True)

        all_detections = []
        crop_index = 0

        def read_image_batch(paths: List[str]) -> List[Tuple[str, np.ndarray]]:
            images = []
            for path in paths:
                img = cv2.imread(path)
                if img is not None:
                    images.append((path, img))
                else:
                    print(f"Warning: Could not read image {path}", flush=True)
            return images

        def save_crops_batch(crop_data_list):
            results = []
            for crop_data in crop_data_list:
                success = cv2.imwrite(crop_data['path'], crop_data['image'])
                if success:
                    results.append(crop_data['info'])
                else:
                    print(f"Error: Failed to save crop to {crop_data['path']}", flush=True)
            return results

        executor = concurrent.futures.ThreadPoolExecutor(max_workers=MAX_WORKERS)

        total_batches = (len(frame_paths) + BATCH_SIZE - 1) // BATCH_SIZE
        start_time = time_module.time()
        
        pbar = tqdm(
            total=len(frame_paths),
            desc=f"Processing {req.model.upper()}",
            unit="img",
            ncols=100,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]",
            file=sys.stdout,
            colour='green'
        )

        # Process batches
        for batch_idx, batch_start in enumerate(range(0, len(frame_paths), BATCH_SIZE), 1):
            batch_end = min(batch_start + BATCH_SIZE, len(frame_paths))
            batch_paths = frame_paths[batch_start:batch_end]
            
            pbar.set_postfix({
                "batch": f"{batch_idx}/{total_batches}",
                "crops": crop_index,
                "device": DEVICE
            })

            loop = asyncio.get_event_loop()
            images_data = await loop.run_in_executor(executor, read_image_batch, batch_paths)

            if not images_data:
                pbar.update(len(batch_paths))
                continue

            paths_batch = [path for path, _ in images_data]
            images_batch = [img for _, img in images_data]

            tqdm.write(f"Batch {batch_idx}/{total_batches}: Processing {len(images_batch)} images...")

            results = detection_model.predict(
                source=images_batch,
                batch=len(images_batch),
                save=False,
                device=DEVICE,
                workers=MAX_WORKERS,
                verbose=False,
                half=True,
            )

            crops_to_save = []
            plates_in_batch = 0

            for idx, (fp, img, r) in enumerate(zip(paths_batch, images_batch, results)):
                time = Path(fp).stem.split("_")[1]
                img_height, img_width = img.shape[:2]

                for box in r.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

                    if x1 >= x2 or y1 >= y2:
                        continue

                    crop_x1 = max(0, x1 - 30)
                    crop_y1 = max(0, y1 - 30)
                    crop_x2 = min(img_width, x2 + 30)
                    crop_y2 = min(img_height, y2 + 30)

                    if crop_x1 >= crop_x2 or crop_y1 >= crop_y2:
                        continue

                    crop = img[crop_y1:crop_y2, crop_x1:crop_x2].copy()

                    if crop.size == 0:
                        continue

                    crop_filename = f"crop_{time}_{crop_index}.png"
                    outp = os.path.join(crops_data_dir, crop_filename)

                    detection_info = {
                        "frame": fp,
                        "crop_index": crop_index,
                        "crop_filename": crop_filename,
                        "coordinates": {
                            "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                            "width": x2 - x1, "height": y2 - y1,
                            "crop_x1": crop_x1, "crop_y1": crop_y1,
                            "crop_x2": crop_x2, "crop_y2": crop_y2
                        },
                        "confidence": float(box.conf.cpu().numpy()[0]),
                        "speed": r.speed,
                    }

                    crops_to_save.append({
                        'path': outp,
                        'image': crop,
                        'info': detection_info
                    })

                    crop_index += 1
                    plates_in_batch += 1

            if crops_to_save:
                saved_detections = await loop.run_in_executor(
                    executor, save_crops_batch, crops_to_save
                )
                all_detections.extend(saved_detections)
                
                if plates_in_batch > 0:
                    tqdm.write(f"Batch {batch_idx}: Found {plates_in_batch} license plates")

            pbar.update(len(batch_paths))
            
            pbar.refresh()

        pbar.close()
        executor.shutdown(wait=True)

        elapsed_time = time_module.time() - start_time
        print(f"\n{'='*60}")
        print(f"Detection completed successfully!")
        print(f"Statistics:")
        print(f"   • Total frames processed: {len(frame_paths)}")
        print(f"   • License plates found: {crop_index}")
        print(f"   • Time elapsed: {elapsed_time:.2f} seconds")
        print(f"   • Average speed: {len(frame_paths)/elapsed_time:.2f} fps")
        print(f"   • Crops saved to: {crops_data_dir}")
        print(f"{'='*60}")

        result_file = os.path.join(crops_data_dir, "yolo_detection_results.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(all_detections, f, indent=2, ensure_ascii=False)

        return {
            "status": "ok", 
            "crops_saved": crop_index,
            "frames_processed": len(frame_paths),
            "processing_time": elapsed_time,
            "crops_dir": crops_data_dir
        }

    # TODO: добавить увеличение на 30px если не было поворота
    @app.post("/api/recognizeLicensePlatesPaddle")
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
                    processed_img = preprocess_crop(cp, upscaler_manager, upscaler_name=upscaler_name)

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

    @app.post("/api/recognizeLicensePlatesFastPlate")
    async def recognize_license_plates_fast(req: RecognizeLicensePlatesRequest):
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

    return app
